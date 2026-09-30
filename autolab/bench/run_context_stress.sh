#!/bin/bash
# autolab 047, the context STRESS test (hours), as ONE command, only with the owner's explicit go-ahead:
#
#     AUTOLAB_OPERATOR=<tag> BUDGET_S=14400 bash autolab/bench/run_context_stress.sh
#
# Steps up the context (1k, 2k, ... 256k) with real prompts, repeats, and a 4-stream pass, for as long as BUDGET_S
# allows (default 4 h for the measurements; publish/settle before and the revert after add ~20 min).
# 1. publishes the test binary + overrides and waits until the fleet is steady (about 10 min: every rank runs the
#    context probe while loading; the probe lines are saved every 20 s meanwhile);  2. reads the probe lines once
# more;  3. output gates;  4. the real-prompt scan inside a 13-minute budget (results written after every size);
# 5. publishes the revert set (what the fleet ran before) and waits for it;  6. gates again.
#
# Built so that a box dying in the middle loses as little as possible: everything measured is on disk the moment it
# is measured (autolab/experiments/034_context_scan/probe_cx.json, context_scan.json, run.log), and every exit,
# planned or not (Ctrl-C included), prints the numbers gathered so far. If the entry box (installed rank 0, the box
# with the tunnel) dies, the API and telemetry go with it; the revert release is still published here and the boxes
# apply it by themselves when that box is back, so the fleet ends on today's configuration either way.
set -u
cd "$(dirname "$0")/../.." || exit 1
EXP=047_context_stress
TEST_BIN=$HOME/inkling-release/builds/cascadia-6430d70e-ctx2
TEST_ENV=$HOME/inkling-release/autolab-overrides/047_context_stress.env
BACK_BIN=$HOME/inkling-release/builds/cascadia-639f0c02-streams
BACK_ENV=$HOME/inkling-release/autolab-overrides/040_counts_041_attn.env
LAB="/usr/bin/python3 autolab/bench/lab.py"
OUT=autolab/experiments/$EXP; mkdir -p "$OUT"
log() { echo "$(date +%T) $*" | tee -a "$OUT/run.log"; }
for f in "$TEST_BIN" "$TEST_ENV" "$BACK_BIN" "$BACK_ENV"; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
: "${AUTOLAB_OPERATOR:?set AUTOLAB_OPERATOR to the tag in ~/inkling-release/publisher.lock/owner}"

save_probe() {   # the probe lines as they stand now (one record per rank and size), quietly
  /usr/bin/python3 autolab/bench/probe_read.py CX "$OUT/probe_cx.json.tmp" > /dev/null 2>&1 && mv -f "$OUT/probe_cx.json.tmp" "$OUT/probe_cx.json"
}

summary() {      # everything gathered so far, from the files on disk
  echo "================ numbers so far ($(date +%T)) ================"
  /usr/bin/python3 - "$OUT" <<'PY'
import json, os, sys
d = sys.argv[1]
p = os.path.join(d, "probe_cx.json")
if os.path.exists(p):
    rows = json.load(open(p))
    if rows:
        print("PROBE (per box, per context size; installed rank 0 = the box that plays rank 8, installed 8 = the box that plays rank 0):")
        print("  rank  ctx        fits  need_mb  avail_mb  decode_ms  attn_ms  mlp_ms")
        for r in sorted(rows, key=lambda r: (int(r.get("rank", 0)), int(r.get("ctx", 0)))):
            print("  %4s  %-9s  %-4s  %7s  %8s  %9s  %7s  %6s" % (r.get("rank"), r.get("ctx"), r.get("fits"), r.get("need_mb", ""),
                                                                r.get("avail_mb", ""), r.get("decode_ms", ""), r.get("attn_ms", ""), r.get("mlp_ms", "")))
    else:
        print("PROBE: no lines received yet")
else:
    print("PROBE: nothing saved yet")
p = os.path.join(d, "context_stress.json")
if os.path.exists(p):
    s = json.load(open(p))
    print("STRESS (%s%s):" % (s.get("note", ""), (", stopped: " + s["stopped_reason"]) if s.get("stopped_reason") else ""))
    print("  | streams | context (tokens) | first token (s) | prefill tok/s | decode tok/s | aggregate | needle | note |")
    for r in s.get("results", []):
        if r.get("skipped"):
            print("  | %d | %d | - | - | - | - | - | stopped: %d s predicted, %d s left |" % (r["streams"], r["size"], r["predicted_s"], r["left_s"]))
        elif r.get("in_flight"):
            print("  | %d | %d | (in flight since %s) | | | | | |" % (r["streams"], r["size"], r["started_at"]))
        else:
            print("  | %d | %s | %s | %s | %s | %s | %s | %s |" % (r["streams"], r.get("prompt_tokens") or r["size"], r.get("ttft_s"), r.get("prefill_tok_s"),
                                                             r.get("decode_tok_s"), r.get("aggregate_decode_tok_s"), "yes" if r.get("needle_found") else "NO", r.get("error") or ""))
else:
    print("STRESS: not started")
PY
  echo "files: $OUT/context_stress.json  $OUT/context_stress.csv  $OUT/run.log"
}

REVERTED=0
revert() {
  [ "$REVERTED" = 1 ] && return
  REVERTED=1
  log "REVERT: publishing what the fleet ran before ($(basename "$BACK_BIN") + $(basename "$BACK_ENV"))"
  $LAB publish --cap 1500 --note "autolab 047 revert: the binary and overrides the fleet ran before the context scan" \
    cascadia="$BACK_BIN" fleet-overrides.env="$BACK_ENV" 2>&1 | tail -2 | tee -a "$OUT/run.log"
  if $LAB settle --cap 900 2>&1 | tail -1 | tee -a "$OUT/run.log" | grep -q "steady 3/3"; then
    $LAB gate 2>&1 | tail -1 | tee -a "$OUT/run.log"
  else
    log "the fleet has not settled on the revert (a box down?): the revert release is published and applies by itself when the box is back"
  fi
}
on_exit() { summary | tee -a "$OUT/run.log"; }
trap 'log "interrupted"; revert; exit 130' INT TERM
trap on_exit EXIT

T0=$(date +%s)
log "1/6 publish: $(basename "$TEST_BIN") + $(basename "$TEST_ENV")"
$LAB publish --dry-run --note t cascadia="$TEST_BIN" fleet-overrides.env="$TEST_ENV" > /dev/null || { log "guards refused the test release"; exit 1; }
$LAB publish --cap 1500 --note "autolab 047: context scan (ONE-SHOT, reverted by the same script): max_seq 1M, 16 slots, API caps up, 64-row prefill windows, per-rank context probe at load" \
  cascadia="$TEST_BIN" fleet-overrides.env="$TEST_ENV" 2>&1 | tail -2 | tee -a "$OUT/run.log"
# the probe lines travel through the entry box: keep a copy of whatever has arrived, every 20 s, while the fleet settles
( while :; do save_probe; sleep 20; done ) & PROBE_POLL=$!
if ! $LAB settle --cap 600 2>&1 | tail -1 | tee -a "$OUT/run.log" | grep -q "steady 3/3"; then
  kill $PROBE_POLL 2>/dev/null; save_probe
  log "the fleet did not settle: no traffic; reverting"; revert; exit 2
fi
kill $PROBE_POLL 2>/dev/null
log "2/6 (no probe in this release: 034 holds the per-box context probe)"
log "3/6 gates"
if ! $LAB gate 2>&1 | tee -a "$OUT/run.log" | tail -1 | grep -q "GATE PASS"; then
  log "gate failed on the test release: reverting"; revert; exit 3
fi
log "4/6 stress (budget ${BUDGET_S:-14400} s; results after every request)"
/usr/bin/python3 autolab/bench/context_stress.py "$EXP" --sizes 1024,2048,4096,8192,16384,32768,65536,131072,262144 --repeats 3,2,1 --streams 1,4 --new-tokens 96 --budget-s "${BUDGET_S:-14400}" 2>&1 | tee -a "$OUT/run.log" | tail -30
log "stress done at $(( ($(date +%s) - T0) / 60 )) min from the start"
log "5/6 revert"; revert
log "6/6 done at $(( ($(date +%s) - T0) / 60 )) min."
