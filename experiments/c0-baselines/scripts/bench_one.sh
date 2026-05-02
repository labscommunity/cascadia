#!/bin/bash
# Run one tahoma worker bench, capture wallclock + stdout, parse tok/s.
# Usage: bench_one.sh <id> <ssh_host> <ps_command>
# The PS command is the full powershell -Command body to run.
set -euo pipefail
ID=$1
HOST=$2
PSCMD=$3
LOG="$(dirname "$0")/../logs/${ID}.log"
RESULTS="$(dirname "$0")/../logs/${ID}.results.json"

START=$(python3 -c "import time; print(time.time())")
ssh "cascadia@${HOST}" "powershell -Command \"${PSCMD}\"" 2>&1 | tee "$LOG"
END=$(python3 -c "import time; print(time.time())")

python3 - "$LOG" "$RESULTS" "$ID" "$HOST" "$START" "$END" <<'PY'
import json, re, sys
log_path, out_path, exp_id, host, start, end = sys.argv[1:]
text = open(log_path, encoding="utf-8", errors="replace").read()
# Pull the "task ... done: N tokens" line if present.
m = re.search(r"done: (\d+) tokens(?:, (\d+) steps,\s*accept=([\d.]+))?", text)
tokens = int(m.group(1)) if m else None
steps = int(m.group(2)) if m and m.group(2) else None
accept = float(m.group(3)) if m and m.group(3) else None
# Active timestamp + done timestamp
def _ts(pat):
    mm = re.search(pat, text)
    if not mm: return None
    return mm.group(1) + " " + mm.group(2)
import datetime as dt
fmt = "%Y-%m-%d %H:%M:%S,%f"
t_active = _ts(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2},\d+).*active")
t_done   = _ts(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2},\d+).*done:")
delta = None
if t_active and t_done:
    a = dt.datetime.strptime(t_active, fmt)
    b = dt.datetime.strptime(t_done, fmt)
    delta = (b - a).total_seconds()
tok_s = (tokens / delta) if (tokens and delta) else None
json.dump({
    "id": exp_id, "host": host,
    "wallclock_s": float(end) - float(start),
    "tokens": tokens, "steps": steps, "accept": accept,
    "decode_time_s": delta, "tok_s": tok_s,
}, open(out_path, "w"), indent=2)
print(f"[{exp_id}] tokens={tokens} decode_s={delta} tok_s={tok_s}")
PY
