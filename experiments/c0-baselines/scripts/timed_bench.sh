#!/bin/bash
# Run a tahoma worker bench with wallclock + per-token timing.
# Usage: timed_bench.sh <id> <ssh_host> <model_path_or_id> <engine> [extra_args...]
set -euo pipefail
ID=$1; HOST=$2; MODEL=$3; ENGINE=$4; shift 4
EXTRA="$*"
LOG_DIR="$(cd "$(dirname "$0")/../logs" && pwd)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${ID}.log"
RESULT="$LOG_DIR/${ID}.json"

# We control the prompt + max-tokens here; each bench is the same workload.
PROMPT="What is the capital of France?"
MAX=64

ssh "cascadia@${HOST}" "powershell -Command \"\$env:PYTHONIOENCODING='utf-8'; Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 1; Set-Location C:\\tahoma; \$sw=[System.Diagnostics.Stopwatch]::StartNew(); '${PROMPT}' | python -m tahoma worker --rank 0 --total 1 --engine ${ENGINE} --device GPU --model ${MODEL} --max-tokens ${MAX} ${EXTRA} 2>&1; \$sw.Stop(); Write-Output \\\"WALL_CLOCK_S=\$(\$sw.Elapsed.TotalSeconds)\\\"\"" > "$LOG" 2>&1

python3 - "$LOG" "$RESULT" "$ID" "$HOST" "$ENGINE" "$MODEL" "$MAX" <<'PY'
import json, re, sys, datetime as dt
log_path, out_path, exp_id, host, engine, model, max_tok = sys.argv[1:]
text = open(log_path, encoding="utf-8", errors="replace").read()
m_wall = re.search(r"WALL_CLOCK_S=([\d.]+)", text)
wall = float(m_wall.group(1)) if m_wall else None

fmt = "%Y-%m-%d %H:%M:%S,%f"
def _ts(s, label):
    m = re.search(rf"(\d{{4}}-\d{{2}}-\d{{2}}) (\d{{2}}:\d{{2}}:\d{{2}},\d+).*{label}", s)
    return dt.datetime.strptime(m.group(1) + " " + m.group(2), fmt) if m else None

t_first = re.search(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2},\d+)", text)
t_first_ts = dt.datetime.strptime(
    t_first.group(1) + " " + t_first.group(2), fmt,
) if t_first else None
t_active = _ts(text, "active")
t_ready = _ts(text, "runner ready")

tokens = decode_s = tok_s = accept = None
m_done = re.search(r"task .* done: (\d+) tokens(?:, (\d+) steps,\s*accept(?:_rate)?=([\d.]+))?", text, re.S)
if m_done:
    tokens = int(m_done.group(1))
    accept = float(m_done.group(3)) if m_done.group(3) else None
    m_d_ts = re.search(rf"(\d{{4}}-\d{{2}}-\d{{2}}) (\d{{2}}:\d{{2}}:\d{{2}},\d+) .*done: {tokens}", text)
    if m_d_ts and t_active:
        b = dt.datetime.strptime(m_d_ts.group(1) + " " + m_d_ts.group(2), fmt)
        decode_s = (b - t_active).total_seconds()
elif wall is not None and t_active and t_first_ts:
    # ov-optimum / pytorch — no per-task done log. Decode = wall - (active - first_log).
    # Assume the run produced max_tokens (output text length suggests yes).
    pre = (t_active - t_first_ts).total_seconds()
    decode_s = wall - pre
    tokens = int(max_tok)

if tokens and decode_s:
    tok_s = tokens / decode_s

load_s = (t_ready - t_first_ts).total_seconds() if t_ready and t_first_ts else None

json.dump({
    "id": exp_id, "host": host, "engine": engine, "model": model,
    "wallclock_s": wall, "load_s": load_s, "tokens": tokens,
    "decode_s": decode_s, "tok_s": tok_s, "accept": accept,
}, open(out_path, "w"), indent=2)
print(f"[{exp_id}] tok_s={tok_s} tokens={tokens} decode_s={decode_s} load_s={load_s} wall={wall}")
PY
