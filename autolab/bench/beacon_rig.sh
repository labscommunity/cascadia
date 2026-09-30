#!/bin/bash
# Container rig for deploy/inkling-fleet/fleet/beacon.py. Runs ON THE MINER (docker), never on the fleet:
#
#   scp deploy/inkling-fleet/fleet/beacon.py miner:inkling-build/beacon.py
#   git show afa7da1b:deploy/inkling-fleet/fleet/beacon.py | ssh miner 'cat > inkling-build/beacon-rig/beacon_old.py'
#   git show HEAD:deploy/inkling-fleet/fleet/beacon.py     | ssh miner 'cat > inkling-build/beacon-rig/beacon_head.py'
#   ssh miner 'bash -s' < autolab/bench/beacon_rig.sh 2>&1 | tee autolab/bench/beacon_rig.last.log
#   ssh miner 'BASE=ubuntu:26.04 bash -s' < autolab/bench/beacon_rig.sh      # the fleet's OS: Ubuntu 26.04, Python 3.14
#
# "Boxes" are containers on a LAN segment of their own (a docker network made for the run, so the broadcasts
# reach nobody else). T1-T4 are the original rig (discovery, address rotation, duplicate rank, the real binary
# dialling by name); T5 rank 0's telemetry file, T6 rank 0's file-server supervision, T7 --show shape and
# old <-> new interop. Every check prints PASS/FAIL with the evidence next to it; exit code = number of FAILs.
#
# The whole body is one function run with stdin from /dev/null: the script itself arrives on stdin
# (`bash -s`), and one command that reads stdin would eat the rest of it.
main() {
set -u
B=$HOME/inkling-build; R=$B/beacon-rig; D="sudo -n docker"
IMG=inkling-beacon-rig; NET=brig-net; SUB=10.213.77; BASE=${BASE:-ubuntu:24.04}
ALL="brig0 brig3 brig4 brig5 brig4dup brig6old brigfs"
PASS=0; FAIL=0; FAILED=""
ok()  { PASS=$((PASS+1)); echo "PASS  $*"; }
bad() { FAIL=$((FAIL+1)); FAILED="$FAILED|$*"; echo "FAIL  $*"; }
is()  { if [ "$2" = "$3" ]; then ok "$1  [$2]"; else bad "$1  [got '$2', expected '$3']"; fi; }     # is "what" got expected
has() { if printf '%s\n' "$2" | grep -Eq -- "$3"; then ok "$1"; else bad "$1  [no match for /$3/ in: $(printf '%s' "$2" | head -c 300)]"; fi; }
cleanup() { $D rm -f $ALL > /dev/null 2>&1; $D network rm $NET > /dev/null 2>&1; $D image rm $IMG > /dev/null 2>&1; true; }
trap cleanup EXIT
for f in $B/beacon.py $R/beacon_old.py $R/beacon_head.py; do [ -s $f ] || { echo "missing $f"; exit 99; }; done
echo "beacon under test: $(sha256sum $B/beacon.py | cut -c1-16)  old(afa7da1b): $(sha256sum $R/beacon_old.py | cut -c1-16)  head: $(sha256sum $R/beacon_head.py | cut -c1-16)   $(date -u +%FT%TZ) on $(hostname)"

# ---------- helpers that run inside the containers (mounted read-only at /rig) ----------
cat > $R/fake-systemctl <<'EOS'
#!/bin/bash
# a worker that is up: what `systemctl show cascadia-inkling.service -p ActiveState -p NRestarts -p MainPID` prints
[ "${1:-}" = show ] && printf 'ActiveState=active\nNRestarts=2\nMainPID=1\n'
exit 0
EOS
cat > $R/fake-journalctl <<'EOS'
#!/bin/bash
# a worker's log whose "stage profile" line is a new one every 4 s (the beacon reads the log every 5 s)
slot=$(( $(date -u +%s) / 4 * 4 ))
echo "2026-01-01T00:00:00.000000Z  INFO cascadia: worker starting"
echo "2026-01-01T00:00:05.000000Z  INFO cascadia: entering relay loop"
echo "$(date -u -d @$slot +%Y-%m-%dT%H:%M:%S).000000Z  INFO cascadia_engine_sparse_moe::engine: stage profile rank=3 total=11 window_ms=4000 frames=$((slot % 1000)) rows=1648 opens=2 open_rows=96 wait_ms=1120 recv_ms=212 compute_ms=2312 max_compute_ms=38 prefill_ms=310 head_ms=0 send_ms=44 relay_ms=0 relays=0 emit_ms=0 replies=0 round_trip_ms=0 max_round_trip_ms=0 attn_ms=1210 mlp_ms=950 prefill_attn_ms=100 prefill_mlp_ms=205 ov_attn_ms=1100 ov_attn_calls=2472 ov_head_ms=0 cache_hits=51234 cache_misses=1200 cache_mib=41234 cache_cap_mib=45000"
EOS
chmod 755 $R/fake-systemctl $R/fake-journalctl
cat > $R/fsinfo.py <<'EOS'
# every `python -m http.server` process in this container: pid uid gid groups state port; and python zombies
import os
n = z = 0
for pid in sorted(int(p) for p in os.listdir("/proc") if p.isdigit()):
    try:
        argv = open("/proc/%d/cmdline" % pid, "rb").read().split(b"\0")
        st = dict(l.split(":\t", 1) for l in open("/proc/%d/status" % pid).read().splitlines() if ":\t" in l)
    except OSError:
        continue
    if st.get("State", "").startswith("Z") and "python" in st.get("Name", ""):
        z += 1
    if b"-m" in argv and argv[argv.index(b"-m") + 1:argv.index(b"-m") + 2] == [b"http.server"]:
        n += 1
        print("httpd pid=%d uid=%s gid=%s groups=[%s] state=%s sid_leader=%s args=%s" % (
            pid, st["Uid"].split()[0], st["Gid"].split()[0], ",".join(st.get("Groups", "").split()), st["State"].split()[0],
            os.getsid(pid) == pid, b" ".join(argv[3:]).decode().strip()))
print("httpd_count=%d zombies=%d" % (n, z))
EOS
cat > $R/fetch.py <<'EOS'
import sys, urllib.request as u
try:
    r = u.build_opener(u.ProxyHandler({})).open("http://127.0.0.1:%s/%s" % (sys.argv[1], sys.argv[2]), timeout=2)
    print(r.status, r.read().decode().strip())
except Exception as e:
    print("ERR", type(e).__name__)
EOS
cat > $R/sniff.py <<'EOS'
# what is on the wire: per sender the key set of its discovery packet, and the size of its telemetry packet
import json, select, socket, sys, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1); s.bind(("", 9099))
disc, tele, end = {}, {}, time.time() + float(sys.argv[1])
while time.time() < end:
    if select.select([s], [], [], 0.2)[0]:
        data, (src, _) = s.recvfrom(4096)
        try: m = json.loads(data.decode())
        except ValueError: continue
        if not isinstance(m, dict): continue
        if m.get("magic") == "cascadia-inkling-beacon-1": disc[m.get("rank")] = (sorted(m), len(data))
        elif m.get("magic") == "cascadia-inkling-tele-1": tele[m.get("rank")] = max(tele.get(m.get("rank"), 0), len(data))
for r in sorted(disc): print("disc rank=%s bytes=%d keys=%s" % (r, disc[r][1], ",".join(disc[r][0])))
for r in sorted(tele): print("tele rank=%s max_bytes=%d" % (r, tele[r]))
EOS
cat > $R/inject.py <<'EOS'
# a box that does not exist (rank 7) sends 100 distinct stage profiles; then everything a LAN can throw at port 9099
import json, socket, sys, time
sys.path.insert(0, "/"); import beacon
name, ip, brd = beacon.local_addrs()[0]
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1); s.bind((ip, 0))
T = beacon.TELE_MAGIC
if sys.argv[1] == "profiles":
    for i in range(100):
        m = {"magic": T, "fleet": "inkling", "rank": 7, "host": "ghost-7", "t": 12345.0 + i, "sys": {"cpu": i / 100.0, "mem_total": 64000},
             "prof": {"at": "T%03d" % i, "window_ms": 4000, "rows": i}}
        if i == 50: m["sys"]["static"] = {"ncpu": 16, "kernel": "ghost"}
        for _ in range(2 if i % 10 == 0 else 1):  # some twice: a repeated profile must not count twice
            s.sendto(json.dumps(m).encode(), (brd, 9099))
        time.sleep(0.01)
    print("sent 100 profiles for rank 7 from %s to %s" % (ip, brd))
else:
    head = '{"magic":"%s","fleet":"inkling",' % T
    bad = [b"", b"[]", b"1", b"null", b'"x"', b"\xff\xfe" + T.encode(), b"[" * 1900 + T.encode(), b"[" * 1900, b"x" * 3000,
           ('{"magic":"%s"}' % T).encode(), (head + '"rank":"x","sys":{}}').encode(), (head + '"rank":-1}').encode(),
           (head + '"rank":99999,"sys":{}}').encode(), (head + '"rank":true,"sys":{}}').encode(), (head + '"rank":2.5,"sys":{}}').encode(),
           (head + '"rank":40,"sys":[1,2],"prof":"x","t":"yesterday","host":{"a":1}}').encode(),
           (head + '"rank":41,"sys":{"cpu":NaN,"x":Infinity,"y":-Infinity},"prof":{"at":NaN,"v":NaN}}').encode(),
           ('{"magic":"%s","fleet":"other","rank":42,"sys":{"cpu":1}}' % T).encode(),
           b'{"magic":"cascadia-inkling-beacon-1","fleet":"inkling","rank":[1]}', b'{"magic":"cascadia-inkling-beacon-1","fleet":"inkling"}',
           b'{"magic":"cascadia-inkling-beacon-1","fleet":"inkling","rank":"nine","total":"x"}']
    for _ in range(3):
        for p in bad:
            s.sendto(p, (brd, 9099)); time.sleep(0.005)
    print("sent %d hostile packets x3 from %s to %s" % (len(bad), ip, brd))
EOS
cat > $R/telecheck.py <<'EOS'
# checks rank 0's aggregate; prints the evidence, then "VERDICT ok" or "VERDICT FAIL: ..."
import json, os, sys, time
P = os.environ.get("TELE", "/run/cascadia-inkling/telemetry.json")
def strict(path):
    raw = open(path).read()
    def const(c): raise ValueError("not JSON: " + c)
    return json.loads(raw, parse_constant=const), len(raw)
mode, args, errs = sys.argv[1], sys.argv[2:], []
def need(cond, what):
    if not cond: errs.append(what)
try:
    doc, size = strict(P)
    if mode == "basic":
        print("file %s  %d bytes  mode %o  owner uid %d  keys=%s  fleet=%s" % (P, size, os.stat(P).st_mode & 0o777, os.stat(P).st_uid, ",".join(sorted(doc)), doc.get("fleet")))
        need(set(doc) >= {"t", "fleet", "ranks", "worker"}, "top-level keys")
        need(os.stat(P).st_mode & 0o777 == 0o644, "mode 0644")
        need(abs(doc["t"] - time.time()) < 3, "t is rank 0's clock, fresh")
        need(sorted(doc["ranks"], key=int) == sorted(args, key=int), "ranks listed %s, expected %s" % (sorted(doc["ranks"], key=int), args))
        for r in args:
            e = doc["ranks"].get(r) or {}
            print("  rank %s host=%s ip=%s rt_age=%.2fs t-rt=%.2fs sys=%d figures (cpu=%s mem_avail=%s) static=%s ring=%d profs=%d" % (
                r, e.get("host"), e.get("ip"), doc["t"] - e.get("rt", 0), (e.get("t") or 0) - e.get("rt", 0), len(e.get("sys", {})), e.get("sys", {}).get("cpu"),
                e.get("sys", {}).get("mem_avail"), ",".join(sorted(e.get("static", {})))[:60], len(e.get("sys_ring", [])), len(e.get("profs", []))))
            need(set(e) == {"host", "ip", "rt", "t", "sys", "static", "sys_ring", "profs"}, "rank %s keys %s" % (r, sorted(e)))
            need(isinstance(e.get("sys"), dict) and len(e["sys"]) >= 8 and "cpu" in e["sys"] and "static" not in e["sys"], "rank %s sys" % r)
            need(isinstance(e.get("rt"), float) and 0 <= doc["t"] - e["rt"] < 3.5, "rank %s rt is a fresh rank-0 receive time" % r)
            need(isinstance(e.get("static"), dict) and "ncpu" in e["static"], "rank %s static" % r)
            need(1 <= len(e.get("sys_ring", [])) <= 12 and all("rt" in s and "sys" in s and "static" not in s["sys"] for s in e["sys_ring"]), "rank %s sys_ring" % r)
            need([s["rt"] for s in e["sys_ring"]] == sorted(s["rt"] for s in e["sys_ring"]) and e["sys_ring"][-1]["rt"] == e["rt"], "rank %s sys_ring oldest first" % r)
            need(r in doc["worker"] and set(doc["worker"][r]) == {"state", "restarts", "phase"}, "worker[%s]" % r)
        print("  worker=%s" % json.dumps(doc["worker"]))
        print("  disc=%s" % json.dumps(doc.get("disc")))
    elif mode == "ring":  # after > 14 s every live rank's ring is full, and never more than 12
        lens = {r: len(e["sys_ring"]) for r, e in doc["ranks"].items() if r in args}
        print("  sys_ring lengths %s" % lens)
        need(all(v == 12 for v in lens.values()) and len(lens) == len(args), "rings full at 12")
    elif mode == "profs":  # rank 3 has a fake worker: a new profile every ~5 s, repeated in every packet between
        e = doc["ranks"]["3"]; ats = [p["at"] for p in e["profs"]]; up = float(args[0])
        print("  rank 3: %d profiles in %.0f s of rank 0 uptime (%d packets heard); at=%s..%s; rt=%s" % (len(ats), up, up, ats[0], ats[-1], [p["rt"] for p in e["profs"]][-3:]))
        print("  last profile: %s" % json.dumps(e["profs"][-1])[:330])
        need(len(set(ats)) == len(ats), "distinct by at")
        need(2 <= len(ats) <= up / 4 + 2, "one entry per profile, not per packet")
        need(all(isinstance(p.get("rt"), float) for p in e["profs"]) and [p["rt"] for p in e["profs"]] == sorted(p["rt"] for p in e["profs"]), "rt on every profile, oldest first")
        need(e["profs"][-1].get("compute_ms") == 2312 and e["profs"][-1].get("window_ms") == 4000, "profile fields intact")
        need(doc["worker"]["3"] == {"state": "active", "restarts": 2, "phase": "serving"}, "worker[3] = %s" % doc["worker"].get("3"))
    elif mode == "inject":
        e = doc["ranks"].get("7") or {}; ats = [p["at"] for p in e.get("profs", [])]
        print("  rank 7 (injected): host=%s profs=%d at=%s..%s ring=%d latest cpu=%s static=%s" % (e.get("host"), len(ats), ats[:1], ats[-1:], len(e.get("sys_ring", [])), e.get("sys", {}).get("cpu"), e.get("static")))
        need(ats == ["T%03d" % i for i in range(70, 100)], "exactly the last 30 distinct profiles, oldest first")
        need(len(e.get("sys_ring", [])) == 12 and e["sys"]["cpu"] == 0.99, "ring 12, latest sys")
        need(e.get("static") == {"ncpu": 16, "kernel": "ghost"}, "static kept from the one packet that had it")
    elif mode == "hostile":
        print("  ranks now: %s   rank 40=%s   rank 41 sys=%s" % (sorted(doc["ranks"], key=int), json.dumps(doc["ranks"].get("40"))[:120], json.dumps(doc["ranks"].get("41", {}).get("sys"))))
        need(not ({"-1", "99999", "42", "1", "2", "x"} & set(doc["ranks"])), "junk ranks stayed out")
        need(doc["ranks"].get("40", {}).get("sys") == {} and doc["ranks"]["40"]["t"] is None and doc["ranks"]["40"]["profs"] == [], "odd types neutralised")
        need(doc["ranks"].get("41", {}).get("sys") == {"cpu": None, "x": None, "y": None}, "NaN/Infinity became null")
    elif mode == "atomic":  # a reader never sees half a file, and the file moves on
        n, ts = 0, set()
        end = time.time() + float(args[0])
        while time.time() < end:
            d, _ = strict(P); ts.add(d["t"]); n += 1; time.sleep(0.004)
        print("  %d reads in %s s, every one complete strict JSON; %d distinct versions seen" % (n, args[0], len(ts)))
        need(len(ts) >= float(args[0]) - 1.5, "rewritten about once a second")
    elif mode == "fs":
        print("  file_server=%s" % json.dumps(doc.get("file_server")))
        fsv = doc.get("file_server") or {}
        need(fsv.get("listening") is (args[0] == "True") and fsv.get("starts") == int(args[1]), "file_server listening=%s starts=%s" % (args[0], args[1]))
    elif mode == "empty":
        print("  %d bytes: %s" % (size, json.dumps(doc)[:260]))
        need(doc["ranks"] == {} and "0" in doc["worker"], "no senders: empty ranks, still a file")
except Exception as e:
    errs.append("%s: %s" % (type(e).__name__, e))
print("VERDICT ok" if not errs else "VERDICT FAIL: " + "; ".join(errs))
EOS

# ---------- the LAN and the boxes ----------
$D rm -f $ALL > /dev/null 2>&1; $D network rm $NET > /dev/null 2>&1
$D network create --subnet $SUB.0/24 $NET > /dev/null || { echo "cannot create the docker network"; exit 98; }
$D image rm $IMG > /dev/null 2>&1   # always from $BASE, never a leftover of another base
printf 'FROM %s\nRUN apt-get update -qq && apt-get install -y -qq iproute2 python3 mawk > /dev/null && rm -rf /var/lib/apt/lists/*\n' $BASE | $D build -q -t $IMG - > /dev/null || { echo "cannot build the image from $BASE"; exit 97; }
echo "boxes are: $($D run --rm $IMG sh -c '. /etc/os-release; echo "$PRETTY_NAME, $(python3 --version)"')"
box() { # box <container> <hostname> <beacon file> [more docker run arguments]
  local c=$1 h=$2 f=$3; shift 3
  $D run -d --init --name $c --hostname $h --network $NET --cap-add NET_ADMIN -v $f:/beacon.py:ro -v $R:/rig:ro "$@" $IMG sleep 3600 > /dev/null; }
serve() { # serve <container> <beacon args...>: the beacon as that box's service, log in /beacon.log, pid in /beacon.pid
  local c=$1; shift; $D exec -d $c bash -c "echo \$\$ > /beacon.pid; exec python3 /beacon.py $* > /beacon.log 2>&1"; }
alive() { $D exec $1 sh -c 'kill -0 $(cat /beacon.pid) 2>/dev/null && echo alive || echo DEAD'; }
addr()  { $D exec $1 bash -c "ip -4 -o addr show dev eth0 | awk '{print \$4}'"; }
restarts() { $D exec $1 sh -c 'cat /restarted 2>/dev/null | wc -l'; }
# T4 runs the real binary on box 3: mounted, not copied (350 MB of OpenVINO; the miner's disk is nearly full)
T4=""; DEBS=$HOME/inkling-ssd-update/inkling-deploy/gpu-debs
if [ -x $B/out/cascadia ] && [ -d $B/ov ] && [ -d $DEBS ]; then T4="-v $B/out/cascadia:/cascadia:ro -v $B/ov:/ov:ro -v $DEBS:/debs:ro"; fi
box brig3 box-3 $B/beacon.py $T4; for r in 4 5; do box brig$r box-$r $B/beacon.py; done
$D exec brig3 sh -c 'ln -s /rig/fake-systemctl /usr/local/bin/systemctl; ln -s /rig/fake-journalctl /usr/local/bin/journalctl'   # box 3 has a "worker"
$D exec brig5 sh -c 'mkdir -p /run/cascadia-inkling; echo "{\"version\": 1789000000}" > /run/cascadia-inkling/update.json'          # box 5 is enrolled with the updater
for r in 3 4 5; do serve brig$r "--rank $r --total 11 --on-change 'date +%T >> /restarted'"; done
sleep 5

echo; echo "##### T1. who is where, seen from box 3 (run next to its own beacon service)"
SHOW=$($D exec brig3 python3 /beacon.py --show --wait 3); echo "$SHOW" | grep -v "not heard"
is "T1 --show lists exactly ranks 3, 4, 5" "$(echo "$SHOW" | grep -E '^ +rank +[0-9]+ +[0-9]+\.' | awk '{print $2}' | tr '\n' ' ')" "3 4 5 "
echo "--- /etc/hosts block on box 5:"; HB=$($D exec brig5 sed -n '/>>> cascadia-inkling/,/<<< cascadia-inkling/p' /etc/hosts); echo "$HB"
is "T1 box 5's /etc/hosts block names the three ranks" "$(echo "$HB" | grep -c 'inkling-rank-[345]$')" 3
is "T1 box 3 resolves inkling-rank-4 to box 4's address" "$($D exec brig3 getent hosts inkling-rank-4 | awk '{print $1}')/24" "$(addr brig4)"

OLD=$(addr brig4); NEW=$SUB.$((200 + RANDOM % 40))
echo; echo "##### T2. box 4's address rotates: $OLD -> $NEW/24"
$D exec brig4 bash -c "ip addr flush dev eth0 && ip addr add $NEW/24 brd + dev eth0"; echo "box 4 now has: $(addr brig4)"
sleep 8
for r in 3 4 5; do echo "--- box $r log:"; $D exec brig$r grep -E "moved|restarting" /beacon.log | tail -2; done
is "T2 box 3 resolves rank 4 to the new address" "$($D exec brig3 getent hosts inkling-rank-4 | awk '{print $1}')" "$NEW"
is "T2 worker restarts triggered on box3 box4 box5 (rank 4 itself and both neighbours)" "$(restarts brig3) $(restarts brig4) $(restarts brig5)" "1 1 1"
$D exec brig5 bash -c "ip addr flush dev eth0 && ip addr add $SUB.250/24 brd + dev eth0"; sleep 8
is "T2 after box 5 rotates too: box3 restarts (5 is not 3's neighbour) / box4 restarts" "$(restarts brig3) $(restarts brig4)" "1 2"
is "T2 box 3 resolves rank 5 to its new address" "$($D exec brig3 getent hosts inkling-rank-5 | awk '{print $1}')" "$SUB.250"

echo; echo "##### T3. a second box wrongly installed as rank 4"
box brig4dup box-4 $B/beacon.py; serve brig4dup "--rank 4 --total 11"; sleep 6
W=$($D exec brig3 grep -E "WARNING" /beacon.log | tail -1 | cut -c1-200); echo "--- box 3 log: $W"
has "T3 box 3 warns that rank 4 is claimed twice" "$W" "WARNING: rank 4 is claimed by"
is "T3 box 3 still resolves rank 4 to the first box (no flapping), restarts on box 3 unchanged" "$($D exec brig3 getent hosts inkling-rank-4 | awk '{print $1}') $(restarts brig3)" "$NEW 1"
echo "--- --show:"; $D exec brig3 python3 /beacon.py --show --wait 3 | grep -E "rank  4" | cut -c1-260
is "T3 --show exit code (2 = duplicate ranks)" "$($D exec brig3 sh -c 'python3 /beacon.py --show --wait 3 > /dev/null; echo $?')" 2
$D rm -f brig4dup > /dev/null 2>&1

echo; echo "##### T4. the real cascadia binary dials its neighbour by name"
if [ -n "$T4" ]; then
  $D exec -d brig4 python3 -c "
import socket,time
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('',9104)); s.listen(1); c,a=s.accept(); open('/accepted','w').write(a[0]); time.sleep(30)"
  $D exec brig3 bash -c "dpkg -i /debs/ocl-icd-libopencl1_*.deb > /dev/null 2>&1; mkdir -p /model; set +u; . /ov/openvino_genai_*/setupvars.sh > /dev/null 2>&1; timeout 12 /cascadia worker --rank 3 --total 11 --engine sparse-moe --device CPU --model /model --layer-start 18 --layer-end 24 --listen :9103 --next inkling-rank-4:9104 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'downstream|connected|waiting|error|Error' | cut -c1-170 | head -5"
  is "T4 box 4's listener accepted a connection from box 3 (dialled as inkling-rank-4)" "$($D exec brig4 cat /accepted 2>/dev/null || echo nobody)/24" "$(addr brig3)"
else
  echo "SKIP  T4: no cascadia binary / OpenVINO runtime / gpu-debs on this machine"
fi

echo; echo "##### T5. rank 0 keeps the fleet's telemetry in /run/cascadia-inkling/telemetry.json"
box brig0 box-0 $B/beacon.py; serve brig0 "--rank 0 --total 11 --on-change 'date +%T >> /restarted'"; T0=$(date +%s)
# next to it, a rank 0 of ANOTHER fleet on another port: no telemetry senders at all, a file server that is already there (T6)
box brigfs box-fs $B/beacon.py
$D exec brigfs sh -c 'useradd -m devcloud; mkdir -p /home/devcloud/inkling-files; echo foreign > /home/devcloud/inkling-files/hello.txt'
$D exec -d brigfs sh -c 'cd /home/devcloud/inkling-files && exec python3 -m http.server 8088 > /dev/null 2>&1'
serve brigfs "--rank 0 --total 2 --fleet fsonly --port 9098 --no-telemetry --telemetry-file /tmp/made/by/beacon/t.json"
sleep 4
is "T5 the file appears within 4 s of rank 0 starting" "$($D exec brig0 sh -c 'test -s /run/cascadia-inkling/telemetry.json && echo yes || echo no')" yes
sleep 14
OUT=$($D exec brig0 python3 /rig/telecheck.py basic 0 3 4 5); echo "$OUT"; has "T5 valid JSON; ranks 0 3 4 5 each with sys, static, rt (rank 0's clock), sys_ring; worker + disc" "$OUT" "^VERDICT ok"
OUT=$($D exec brig0 python3 /rig/telecheck.py atomic 5); echo "$OUT"; has "T5 rewritten ~once a second, atomically (no reader ever sees half a file)" "$OUT" "^VERDICT ok"
OUT=$($D exec brig0 python3 /rig/telecheck.py ring 0 3 4 5); echo "$OUT"; has "T5 sys_ring holds 12 samples per rank, no more" "$OUT" "^VERDICT ok"
OUT=$($D exec brig0 python3 /rig/telecheck.py profs $(( $(date +%s) - T0 ))); echo "$OUT"; has "T5 stage profiles: one entry per distinct profile (not per packet), each with rt; worker state of box 3" "$OUT" "^VERDICT ok"
is "T5 only rank 0 writes the file (box 3, box 4 have none)" "$($D exec brig3 sh -c 'ls /run/cascadia-inkling/ | tr "\n" " "')|$($D exec brig4 sh -c 'ls /run/cascadia-inkling/ | tr "\n" " "')" "fleet.json |fleet.json "
OUT=$($D exec -e TELE=/tmp/made/by/beacon/t.json brigfs python3 /rig/telecheck.py empty); echo "$OUT"; has "T5 a rank 0 that hears no telemetry at all still writes a (small) file, at --telemetry-file, directory made" "$OUT" "^VERDICT ok"
$D exec brig4 python3 /rig/inject.py profiles; sleep 2
OUT=$($D exec brig0 python3 /rig/telecheck.py inject); echo "$OUT"; has "T5 100 profiles from one rank: the last 30 distinct kept, oldest first; ring 12; static kept" "$OUT" "^VERDICT ok"
$D exec brig4 python3 /rig/inject.py hostile; sleep 3
is "T5 every beacon survived the hostile packets (boxes 0 3 4 5)" "$(alive brig0) $(alive brig3) $(alive brig4) $(alive brig5)" "alive alive alive alive"
OUT=$($D exec brig0 python3 /rig/telecheck.py hostile); echo "$OUT"; has "T5 hostile telemetry: junk ranks ignored, odd types neutralised, NaN -> null, file still strict JSON" "$OUT" "^VERDICT ok"
OUT=$($D exec brig0 python3 /rig/telecheck.py atomic 3); echo "$OUT"; has "T5 the file keeps moving after the hostile packets" "$OUT" "^VERDICT ok"
SHOW=$($D exec brig3 python3 /beacon.py --show --wait 3 | grep -v "not heard"); is "T5 discovery unharmed: --show still lists 0 3 4 5" "$(echo "$SHOW" | grep -E '^ +rank +[0-9]+ +[0-9]+\.' | awk '{print $2}' | tr '\n' ' ')" "0 3 4 5 "
echo "--- telemetry.json size with 4 live ranks + 3 injected: $($D exec brig0 stat -c %s /run/cascadia-inkling/telemetry.json) bytes"
P0=$($D exec brig0 cat /beacon.pid); C1=$($D exec brig0 awk '{print $14+$15}' /proc/$P0/stat); sleep 10; C2=$($D exec brig0 awk '{print $14+$15}' /proc/$P0/stat)
echo "--- rank 0 beacon's own CPU: $((C2 - C1)) ticks of 1000 in 10 s (children, i.e. the ip/systemctl calls every beacon makes, not counted)"
echo "--- rank 0 beacon log:"; $D exec brig0 cat /beacon.log | cut -c1-200 | head -12

echo; echo "##### T6. rank 0 keeps the fleet's file server (port 8088) alive"
is "T6 before there is a user or a folder: nothing started, nothing listening" "$($D exec brig0 python3 /rig/fsinfo.py | tail -1) $($D exec brig0 python3 /rig/fetch.py 8088 hello.txt)" "httpd_count=0 zombies=0 ERR URLError"
echo "--- rank 0 log so far: $($D exec brig0 grep 'file server' /beacon.log | cut -c1-160)"
for c in brig0 brig3; do $D exec $c sh -c 'useradd -m devcloud && mkdir -p /home/devcloud/inkling-files && echo "hello from $(hostname)" > /home/devcloud/inkling-files/hello.txt && chown -R devcloud: /home/devcloud/inkling-files'; done
UIDG=$($D exec brig0 sh -c 'echo uid=$(id -u devcloud) gid=$(id -g devcloud)'); T1=$(date +%s); GOT=""
for i in $(seq 1 45); do GOT=$($D exec brig0 python3 /rig/fetch.py 8088 hello.txt); [ "$GOT" = "200 hello from box-0" ] && break; sleep 1; done
is "T6 port 8088 serves hello.txt within ~40 s of the folder appearing (took $(( $(date +%s) - T1 )) s)" "$GOT" "200 hello from box-0"
INFO=$($D exec brig0 python3 /rig/fsinfo.py); echo "$INFO"; echo "    devcloud is $UIDG"
has "T6 it runs as devcloud, not root: uid, gid AND supplementary groups" "$INFO" "^httpd pid=[0-9]+ $UIDG groups=\[$($D exec brig0 id -g devcloud)\] state=S sid_leader=True args=8088 --directory /home/devcloud/inkling-files --bind 0.0.0.0"
is "T6 exactly one server" "$(echo "$INFO" | tail -1)" "httpd_count=1 zombies=0"
PID1=$(echo "$INFO" | sed -n 's/^httpd pid=\([0-9]*\).*/\1/p' | head -1); $D exec brig0 kill $PID1; sleep 1
is "T6 killed it (pid $PID1): port 8088 is dead" "$($D exec brig0 python3 /rig/fetch.py 8088 hello.txt)" "ERR URLError"
T1=$(date +%s); for i in $(seq 1 45); do GOT=$($D exec brig0 python3 /rig/fetch.py 8088 hello.txt); [ "$GOT" = "200 hello from box-0" ] && break; sleep 1; done
is "T6 it is back by itself (took $(( $(date +%s) - T1 )) s)" "$GOT" "200 hello from box-0"
INFO=$($D exec brig0 python3 /rig/fsinfo.py); echo "$INFO"
PID2=$(echo "$INFO" | sed -n 's/^httpd pid=\([0-9]*\).*/\1/p' | head -1)
is "T6 a new process (pid $PID1 -> $PID2), still exactly one, the dead one reaped (no zombie)" "$([ "$PID1" != "$PID2" ] && echo new) $(echo "$INFO" | tail -1)" "new httpd_count=1 zombies=0"
has "T6 still as devcloud after the restart" "$INFO" "^httpd pid=$PID2 $UIDG groups="
echo "--- waiting 35 s: one more supervision round must NOT start a second server"; sleep 35
is "T6 after another round: still one server, the same pid" "$($D exec brig0 python3 /rig/fsinfo.py | sed -n 's/^httpd pid=\([0-9]*\).*/\1/p' | tr '\n' ' ')" "$PID2 "
echo "--- rank 0 log:"; $D exec brig0 grep 'file server' /beacon.log | cut -c1-200
is "T6 one log line per start (2 starts)" "$($D exec brig0 grep -c 'file server: nothing was listening on port 8088, started' /beacon.log)" 2
OUT=$($D exec brig0 python3 /rig/telecheck.py fs True 2); echo "$OUT"; has "T6 telemetry.json reports the file server: listening, 2 starts" "$OUT" "^VERDICT ok"
echo "--- the beacon service restarts (an update of beacon.py does that). Without systemd here, both outcomes by hand:"
$D exec brig0 sh -c 'kill $(cat /beacon.pid)'; sleep 1; serve brig0 "--rank 0 --total 11 --on-change 'date +%T >> /restarted'"; sleep 6
is "T6 beacon restarted, its file server survived: the new beacon adopts it (same pid, no second one, 0 starts logged)" "$($D exec brig0 python3 /rig/fsinfo.py | sed -n 's/^httpd pid=\([0-9]*\).*/\1/p' | tr '\n' ' ')$($D exec brig0 grep -c 'file server' /beacon.log)" "$PID2 0"
$D exec brig0 sh -c "kill \$(cat /beacon.pid) $PID2"; sleep 1; serve brig0 "--rank 0 --total 11 --on-change 'date +%T >> /restarted'"; T1=$(date +%s); GOT=""
for i in $(seq 1 12); do GOT=$($D exec brig0 python3 /rig/fetch.py 8088 hello.txt); [ "$GOT" = "200 hello from box-0" ] && break; sleep 0.5; done
is "T6 beacon restarted and the file server died with it (what systemd's control-group stop does): back $(( $(date +%s) - T1 )) s after the beacon" "$GOT $($D exec brig0 python3 /rig/fsinfo.py | tail -1)" "200 hello from box-0 httpd_count=1 zombies=0"
is "T6 neither restart of the beacon restarted the worker (no address changed), telemetry file is current again" "$(restarts brig0) $($D exec brig0 python3 /rig/telecheck.py atomic 3 | tail -1)" "0 VERDICT ok"
is "T6 a box that is NOT rank 0 (box 3: same user, same folder) starts none" "$($D exec brig3 python3 /rig/fsinfo.py | tail -1) $($D exec brig3 python3 /rig/fetch.py 8088 hello.txt) $($D exec brig3 grep -c 'file server' /beacon.log)" "httpd_count=0 zombies=0 ERR URLError 0"
INFO=$($D exec brigfs python3 /rig/fsinfo.py); echo "--- the other rank 0, where somebody's server (root's) was listening before the beacon started, $(( $(date +%s) - T0 )) s ago:"; echo "$INFO"
is "T6 a server that is already listening is left alone: still one, still root's, none started" "$(echo "$INFO" | sed -n 's/^httpd pid=[0-9]* \(uid=[0-9]*\).*/\1/p' | tr '\n' ' ')$(echo "$INFO" | tail -1) $($D exec brigfs python3 /rig/fetch.py 8088 hello.txt) $($D exec brigfs grep -c 'file server' /beacon.log)" "uid=0 httpd_count=1 zombies=0 200 foreign 0"
OUT=$($D exec -e TELE=/tmp/made/by/beacon/t.json brigfs python3 /rig/telecheck.py fs True 0); echo "$OUT"; has "T6 ... and its telemetry.json says listening, 0 starts" "$OUT" "^VERDICT ok"
is "T6 discovery went on through all of it (boxes 0 3 4 5 alive)" "$(alive brig0) $(alive brig3) $(alive brig4) $(alive brig5)" "alive alive alive alive"

echo; echo "##### T7. --show keeps its shape; the beacon before telemetry (afa7da1b) and this one discover each other"
box brig6old box-6 $R/beacon_old.py; serve brig6old "--rank 6 --total 11"; sleep 7
RX='^\s+rank\s+[0-9]+\s+\S+\s+\S+\s+files'
SHOW=$($D exec brig3 python3 /beacon.py --show --wait 3); echo "--- new --show on box 3:"; echo "$SHOW" | grep -v "not heard" | cut -c1-200
HEARD=$(echo "$SHOW" | grep -E '^ +rank' | grep -vc "not heard")
is "T7 every heard rank line matches the parser's regex (heard: $HEARD)" "$(echo "$SHOW" | grep -Ec "$RX")" "$HEARD"
is "T7 new --show lists 0 3 4 5 and the old beacon's rank 6" "$(echo "$SHOW" | grep -E "$RX" | awk '{print $2}' | tr '\n' ' ')" "0 3 4 5 6 "
has "T7 exact shape, enrolled box + worker status (rank 5)" "$SHOW" '^  rank  5  10\.213\.77\.250    box-5            files [0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}  \| worker \?, 0 restarts: $'
has "T7 exact shape, not enrolled + a worker that serves (rank 3, run on box 3 itself)" "$SHOW" '^  rank  3  10\.213\.77\.[0-9]+ +box-3            files: not enrolled  \| worker active, 2 restarts: serving$'
A=$($D exec brig3 sh -c 'python3 /beacon.py --show --wait 3 > /tmp/new.txt & python3 /rig/beacon_head.py --show --wait 3 > /tmp/head.txt; wait; diff /tmp/new.txt /tmp/head.txt > /tmp/show.diff && echo identical || cat /tmp/show.diff')
is "T7 --show output of this beacon and of the committed one (HEAD), run side by side on the same packets" "$A" identical
OSHOW=$($D exec brig6old python3 /beacon.py --show --wait 3); echo "--- OLD --show on box 6:"; echo "$OSHOW" | grep -v "not heard" | cut -c1-200
is "T7 the old beacon's --show parses the new beacons' packets: lists 0 3 4 5 6" "$(echo "$OSHOW" | grep -E "$RX" | awk '{print $2}' | tr '\n' ' ')" "0 3 4 5 6 "
echo "--- old beacon's log:"; $D exec brig6old cat /beacon.log | cut -c1-160
is "T7 old service heard the 4 new ranks, wrote them to /etc/hosts, and is alive" "$($D exec brig6old grep -c 'inkling-rank-[0345]$' /etc/hosts) $(alive brig6old)" "4 alive"
is "T7 new services resolve the old box's rank" "$($D exec brig0 getent hosts inkling-rank-6 | awk '{print $1}')/24 $($D exec brig5 getent hosts inkling-rank-6 | awk '{print $1}')/24" "$(addr brig6old) $(addr brig6old)"
WIRE=$($D exec brig3 python3 /rig/sniff.py 3); echo "--- on the wire:"; echo "$WIRE"
is "T7 discovery packet: same keys from the old (rank 6) and the new (rank 3) beacon" "$(echo "$WIRE" | sed -n 's/^disc rank=3 .*keys=//p')" "$(echo "$WIRE" | sed -n 's/^disc rank=6 .*keys=//p')"
is "T7 telemetry packets fit one frame (<= 1400 bytes) and the service's 2048-byte buffer" "$(echo "$WIRE" | sed -n 's/^tele .*max_bytes=//p' | awk '$1 > 1400' | wc -l)" 0
has "T7 rank 0's file: a box without telemetry (old beacon) is in worker/disc, not in ranks" "$($D exec brig0 python3 -c "
import json; d = json.load(open('/run/cascadia-inkling/telemetry.json')); print('6' in d['ranks'], '6' in d['worker'], '6' in d['disc'], d['disc']['5']['files'])")" "^False True True 1789000000$"
echo "--- teeth check (not a PASS/FAIL of the new code): the same hostile packets against the OLD beacon"
$D exec brig4 python3 /rig/inject.py hostile > /dev/null; sleep 3
echo "    old beacon (rank 6): $(alive brig6old)   $($D exec brig6old tail -1 /beacon.log | cut -c1-120)"
is "T7 ... while every new beacon is still alive" "$(alive brig0) $(alive brig3) $(alive brig4) $(alive brig5)" "alive alive alive alive"

echo; echo "##### result: $PASS passed, $FAIL failed"
[ $FAIL -gt 0 ] && echo "$FAILED" | tr '|' '\n' | sed '/^$/d; s/^/   FAILED: /'
echo "containers, network and image are removed on exit"
exit $FAIL
}
main "$@" < /dev/null
