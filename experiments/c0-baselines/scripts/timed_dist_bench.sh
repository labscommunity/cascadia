#!/bin/bash
# Distributed bench: start a worker on the second node, then run the driver
# on the first node. Times the driver wallclock + parses the engine's
# task-done log.
# Usage: timed_dist_bench.sh <id> <driver_host> <worker_host> \
#                            <driver_tb_ip> <worker_tb_ip> \
#                            <model_dir_driver> <model_dir_worker> \
#                            <engine> [extra_driver_args...]
set -euo pipefail
ID=$1; DRIVER=$2; WORKER=$3; DRIVER_IP=$4; WORKER_IP=$5
MODEL_D=$6; MODEL_W=$7; ENGINE=$8; shift 8
EXTRA="$*"

LOG_DIR="$(cd "$(dirname "$0")/../logs" && pwd)"
mkdir -p "$LOG_DIR"
DRIVER_LOG="$LOG_DIR/${ID}-driver.log"
WORKER_LOG="$LOG_DIR/${ID}-worker.log"
RESULT="$LOG_DIR/${ID}.json"

PROMPT="What is the capital of France?"
MAX=64

# Start worker (rank 1) on WORKER_HOST, listening on WORKER_IP:9100.
ssh "cascadia@${WORKER}" "powershell -Command \"\$env:PYTHONIOENCODING='utf-8'; Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 1; Set-Location C:\\tahoma; python -m tahoma worker --rank 1 --total 2 --engine ${ENGINE} --device GPU --model ${MODEL_W} --listen ${WORKER_IP}:9100 2>&1 | Tee-Object -FilePath ${ID}-w.log\"" > "$WORKER_LOG" 2>&1 &
WPID=$!

# Wait for worker bind.
for i in $(seq 1 30); do
  if ssh "cascadia@${WORKER}" "powershell -Command \"Get-Content C:\\tahoma\\${ID}-w.log -ErrorAction SilentlyContinue\"" 2>/dev/null | grep -q "ActivationServer listening"; then
    echo "[bind ok at ${i}s]"
    break
  fi
  sleep 2
done

# Run driver (rank 0) on DRIVER_HOST.
ssh "cascadia@${DRIVER}" "powershell -Command \"\$env:PYTHONIOENCODING='utf-8'; Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 1; Set-Location C:\\tahoma; \$sw=[System.Diagnostics.Stopwatch]::StartNew(); '${PROMPT}' | python -m tahoma worker --rank 0 --total 2 --engine ${ENGINE} --device GPU --model ${MODEL_D} --next ${WORKER_IP}:9100 --max-tokens ${MAX} ${EXTRA} 2>&1; \$sw.Stop(); Write-Output \\\"WALL_CLOCK_S=\$(\$sw.Elapsed.TotalSeconds)\\\"\"" > "$DRIVER_LOG" 2>&1

# Tear down worker SSH (worker python dies with it).
ssh "cascadia@${WORKER}" "powershell -Command \"Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force\"" >/dev/null 2>&1
kill $WPID 2>/dev/null || true

# Parse driver log for tok/s.
python3 - "$DRIVER_LOG" "$RESULT" "$ID" "$DRIVER" "$ENGINE" "$MODEL_D" "$MAX" <<'PY'
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
t_first_ts = dt.datetime.strptime(t_first.group(1) + " " + t_first.group(2), fmt) if t_first else None
t_active = _ts(text, "active")
t_ready = _ts(text, "runner ready")

tokens = decode_s = tok_s = accept = steps = None
m_done = re.search(r"task .* done: (\d+) tokens(?:, (\d+) steps,\s*accept(?:_rate)?=([\d.]+))?", text, re.S)
if m_done:
    tokens = int(m_done.group(1))
    steps = int(m_done.group(2)) if m_done.group(2) else None
    accept = float(m_done.group(3)) if m_done.group(3) else None
    m_d_ts = re.search(rf"(\d{{4}}-\d{{2}}-\d{{2}}) (\d{{2}}:\d{{2}}:\d{{2}},\d+) .*done: {tokens}", text)
    if m_d_ts and t_active:
        b = dt.datetime.strptime(m_d_ts.group(1) + " " + m_d_ts.group(2), fmt)
        decode_s = (b - t_active).total_seconds()
elif wall and t_active and t_first_ts:
    pre = (t_active - t_first_ts).total_seconds()
    decode_s = wall - pre
    tokens = int(max_tok)

if tokens and decode_s:
    tok_s = tokens / decode_s
load_s = (t_ready - t_first_ts).total_seconds() if t_ready and t_first_ts else None

json.dump({
    "id": exp_id, "host": host, "engine": engine, "model": model,
    "wallclock_s": wall, "load_s": load_s, "tokens": tokens, "steps": steps,
    "decode_s": decode_s, "tok_s": tok_s, "accept": accept,
}, open(out_path, "w"), indent=2)
print(f"[{exp_id}] tok_s={tok_s} tokens={tokens} steps={steps} accept={accept} decode_s={decode_s} wall={wall}")
PY
