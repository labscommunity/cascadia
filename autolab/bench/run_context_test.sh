#!/bin/bash
# autolab 034, the context scan, as ONE command (only with the owner's explicit go-ahead; see the hypothesis):
#
#     AUTOLAB_OPERATOR=<the tag in ~/inkling-release/publisher.lock/owner> bash autolab/bench/run_context_test.sh
#
# 1. publishes the test binary + overrides and waits until the fleet is steady (about 10 min: every rank runs the
#    context probe while loading);  2. reads the probe lines;  3. output gates;  4. the real-prompt scan inside a
# 13-minute budget;  5. publishes the revert set (what the fleet ran before) and waits for it;  6. gates again.
# Stops at the first thing that is not right (a gate failure, a fleet that does not settle) and, whatever happened
# after step 1, always ends with the revert. Wall clock: about 25 min to the end of the scan, ~35 with the revert.
set -u
cd "$(dirname "$0")/../.." || exit 1
EXP=034_context_scan
TEST_BIN=$HOME/inkling-release/builds/cascadia-83cefc97-ctx
TEST_ENV=$HOME/inkling-release/autolab-overrides/034_context_scan.env
BACK_BIN=$HOME/inkling-release/builds/cascadia-639f0c02-streams
BACK_ENV=$HOME/inkling-release/autolab-overrides/040_counts_041_attn.env
LAB="/usr/bin/python3 autolab/bench/lab.py"
OUT=autolab/experiments/$EXP; mkdir -p "$OUT"
log() { echo "$(date +%T) $*" | tee -a "$OUT/run.log"; }
for f in "$TEST_BIN" "$TEST_ENV" "$BACK_BIN" "$BACK_ENV"; do [ -s "$f" ] || { echo "missing $f"; exit 1; }; done
: "${AUTOLAB_OPERATOR:?set AUTOLAB_OPERATOR to the tag in ~/inkling-release/publisher.lock/owner}"

revert() {
  log "REVERT: publishing what the fleet ran before ($(basename "$BACK_BIN") + $(basename "$BACK_ENV"))"
  $LAB publish --cap 1500 --note "autolab 034 revert: the binary and overrides the fleet ran before the context scan" \
    cascadia="$BACK_BIN" fleet-overrides.env="$BACK_ENV" 2>&1 | tail -2 | tee -a "$OUT/run.log"
  $LAB settle --cap 900 2>&1 | tail -1 | tee -a "$OUT/run.log"
  $LAB gate 2>&1 | tail -1 | tee -a "$OUT/run.log"
}

T0=$(date +%s)
log "1/6 publish: $(basename "$TEST_BIN") + $(basename "$TEST_ENV")"
$LAB publish --dry-run --note t cascadia="$TEST_BIN" fleet-overrides.env="$TEST_ENV" > /dev/null || { log "guards refused the test release"; exit 1; }
$LAB publish --cap 1500 --note "autolab 034: context scan (ONE-SHOT, reverted by the same script): max_seq 1M, 16 slots, API caps up, per-rank context probe at load" \
  cascadia="$TEST_BIN" fleet-overrides.env="$TEST_ENV" 2>&1 | tail -2 | tee -a "$OUT/run.log"
if ! $LAB settle --cap 600 2>&1 | tail -1 | tee -a "$OUT/run.log" | grep -q "steady 3/3"; then
  log "the fleet did not settle: no traffic; reverting"; revert; exit 2
fi
log "2/6 probe lines (per rank and size; fits=0 = that size does not fit in the box's free memory)"
/usr/bin/python3 autolab/bench/probe_read.py CX "$OUT/probe_cx.json" 2>&1 | tee -a "$OUT/run.log" | tail -80
log "3/6 gates"
if ! $LAB gate 2>&1 | tee -a "$OUT/run.log" | tail -1 | grep -q "GATE PASS"; then
  log "gate failed on the test release: reverting"; revert; exit 3
fi
log "4/6 scan (13-minute budget; sizes that would not fit in it are skipped)"
/usr/bin/python3 autolab/bench/context_scan.py "$EXP" --sizes 1024,8192,32768,65536,102400 --budget-s 780 --new-tokens 32 2>&1 | tee -a "$OUT/run.log" | tail -12
log "scan done at $(( ($(date +%s) - T0) / 60 )) min from the start"
log "5/6 revert"; revert
log "6/6 done at $(( ($(date +%s) - T0) / 60 )) min. Results: $OUT/context_scan.json, $OUT/probe_cx.json, $OUT/run.log"
