#!/usr/bin/env python3
"""build_role_swap.py BASE.env OUT_DIR [--pair "0 8"]: the two overrides files of the role swap (autolab 032).

    032a_role_sync.env   BASE + the sync block: the two boxes of the pair copy each other's role data over the LAN
                         (deploy/inkling-fleet/fleet/role_sync.py, embedded byte for byte). Changes no role.
    032b_run.sh          fleet/run.sh with ROLE_SWAP="A B" (the repository's copy keeps it empty, so no other release can
                         repoint a rank by accident)
    032b_role_swap.env   BASE + the sync block (idempotent: data already verified) + the relay block: the entry box,
                         once it no longer plays rank 0, answers :8000 by relaying to the box that does
                         (fleet/api_relay.py, embedded). Goes out TOGETHER with a run.sh whose ROLE_SWAP names the pair,
                         and only after both boxes reported ready=1 (probe lines SW<box>).

Revert: publish the previous run.sh (ROLE_SWAP="") and BASE.env. Nothing on the boxes is deleted by any of this.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET = os.path.join(os.path.dirname(os.path.dirname(HERE)), "deploy", "inkling-fleet", "fleet")


def block_sync(pair, src):
    return '''
# autolab 032a (the boxes INSTALLED as ranks %(a)s and %(b)s): each gives the other the model files of its pipeline role
# (shells, experts, attention IRs, the embedding), sha256-checked, into the same model folder (other layer numbers:
# nothing collides, nothing is deleted or overwritten). 60 MB/s so the pipeline's frames keep the wire. When BOTH
# boxes hold each other's role, each writes role-swap/ready-<role>: the marker run.sh's ROLE_SWAP needs. Progress
# on "stage profile" lines (probe_read.py SW). Starts eight minutes after the worker so a model load is not disturbed.
_pair="%(a)s %(b)s"; _box="${BOX_RANK:-$RANK}"; _pa="${_pair%%%% *}"; _pb="${_pair##* }"; _peer=""
[ "$_box" = "$_pa" ] && _peer="$_pb"; [ "$_box" = "$_pb" ] && _peer="$_pa"
if [ -n "$_peer" ]; then
  mkdir -p /run/cascadia-inkling
  cat > /run/cascadia-inkling/role_sync.py <<'ROLE_SYNC_PY'
%(src)s
ROLE_SYNC_PY
  ( set +e +u; sleep 480
    exec /usr/bin/python3 /run/cascadia-inkling/role_sync.py --prefix "$PREFIX" --fleet "$FLEET" --box "$_box" --peer "$_peer" \\
      --per "$((LAYER_END - LAYER_START))" --mbps 60 --reserve-gb 70 --hours 6 ) &
fi
''' % dict(a=pair[0], b=pair[1], src=src.rstrip("\n"))


def block_clock(proxy):
    return '''
# autolab 032 (the entry box): a power loss resets this box's clock to its firmware date (three times on 2026-09-20,
# back to July). publish.py stamps the fleet's manifest with this clock and every updater refuses a manifest older
# than the one it applied: with the clock in the past NO release can roll out until someone sets it by hand. If the
# clock is provably wrong (earlier than a release this box already applied) take the time from an HTTPS Date header
# through the lab proxy. Forward only, this box only, and only in that case.
if [ "${BOX_RANK:-$RANK}" = "0" ]; then
  _last=$(sed -n 's/.*"version"[^0-9]*\\([0-9][0-9]*\\).*/\\1/p' "$PREFIX/update-state.json" 2>/dev/null | head -1)
  _was=$(date +%%s)
  if [ -n "$_last" ] && [ "$_was" -lt "$_last" ]; then
    _hdr=$(curl -sI --max-time 20 -x "%(proxy)s" https://www.google.com 2>/dev/null | sed -n 's/^[Dd]ate: *//p' | tr -d '\\r' | head -1)
    _now=$(date -u -d "$_hdr" +%%s 2>/dev/null || echo 0)
    if [ "$_now" -gt "$_last" ]; then
      date -u -s "@$_now" > /dev/null 2>&1 && { hwclock -w > /dev/null 2>&1; echo "CLK probe stage profile ph=1 clock_set=1 was_behind_s=$((_now - _was))"; }
    else
      echo "CLK probe stage profile ph=8 clock_set=0 was=$_was last=$_last got=$_now"
    fi
  fi
fi
''' % dict(proxy=proxy)


def block_relay(src):
    return '''
# autolab 032b (the entry box, when it no longer plays pipeline rank 0): the operator tunnel, the Tailscale address
# and every bookmark point at THIS box's port 8000; the API, the dashboard and the scheduler now run on the box that
# plays rank 0. A relay keeps the address working (streaming included) and answers /api/fleet/telemetry itself: that
# file is written by this box's beacon. BOX_RANK and ROLE_SWAP come from run.sh.
if [ "${BOX_RANK:-$RANK}" = "0" ] && [ "$RANK" != "0" ]; then
  _rp="${ROLE_SWAP:-}"; _ra="${_rp%% *}"; _rb="${_rp##* }"; _up="$_ra"; [ "$_ra" = "0" ] && _up="$_rb"
  mkdir -p /run/cascadia-inkling
  cat > /run/cascadia-inkling/api_relay.py <<'API_RELAY_PY'
%(src)s
API_RELAY_PY
  ( set +e +u; while :; do /usr/bin/python3 /run/cascadia-inkling/api_relay.py --listen 8000 --upstream "${FLEET}-rank-${_up}:8000"; sleep 5; done ) &
fi
''' % dict(src=src.rstrip("\n"))


def main():
    base, out = sys.argv[1], sys.argv[2]
    pair = (sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--pair" else "0 8").split()
    assert len(pair) == 2 and all(p.isdigit() for p in pair), pair
    s = open(base).read().rstrip("\n") + "\n"
    import re
    m = re.search(r"PX=(http://[^ ;\n]+)", s)   # the lab proxy, as the base file names it (redacted in repository copies)
    assert m, "no PX= in the base overrides"
    s += block_clock(m.group(1))
    sync = block_sync(pair, open(os.path.join(FLEET, "role_sync.py")).read())
    relay = block_relay(open(os.path.join(FLEET, "api_relay.py")).read())
    run = open(os.path.join(FLEET, "run.sh")).read()
    assert run.count('ROLE_SWAP=""\n') == 1, "run.sh must carry the empty default exactly once"
    p = os.path.join(out, "032b_run.sh")
    open(p, "w").write(run.replace('ROLE_SWAP=""\n', 'ROLE_SWAP="%s %s"\n' % tuple(pair)))
    os.chmod(p, 0o755); print(p, "(run.sh with ROLE_SWAP=\"%s %s\")" % tuple(pair))
    for name, body in (("032a_role_sync.env", s + sync), ("032b_role_swap.env", s + sync + relay)):
        assert "ROLE_SYNC_PY\n" not in open(os.path.join(FLEET, "role_sync.py")).read()
        p = os.path.join(out, name)
        open(p, "w").write(body)
        print(p, len(body), "bytes")


if __name__ == "__main__":
    main()
