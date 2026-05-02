#!/bin/bash
# c1.<n>: LLMPipeline bench on a remote node.
# Usage: run.sh <id> <ssh_host> <model_dir> [extra args to bench_llmpipeline.py...]
set -euo pipefail
ID=$1; HOST=$2; MODEL=$3; shift 3
EXTRA="$*"
LOG_DIR="$(cd "$(dirname "$0")/../logs" && pwd)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${ID}.log"
RESULT="$LOG_DIR/${ID}.json"

# Push the bench script.
scp "$(dirname "$0")/bench_llmpipeline.py" "cascadia@${HOST}:C:/tahoma/bench_llmpipeline.py" >/dev/null 2>&1

ssh "cascadia@${HOST}" "powershell -Command \"\$env:PYTHONIOENCODING='utf-8'; Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep 1; cd C:\\tahoma; python bench_llmpipeline.py --model ${MODEL} --device GPU --max-tokens 64 ${EXTRA}\"" > "$LOG" 2>&1

# Pull the RESULT={...} line.
python3 - "$LOG" "$RESULT" "$ID" "$HOST" <<'PY'
import json, re, sys
log_path, out_path, exp_id, host = sys.argv[1:]
text = open(log_path, encoding="utf-8", errors="replace").read()
m = re.search(r"RESULT=(\{.+\})", text)
if not m:
    print(f"[{exp_id}] NO RESULT line — see {log_path}")
    json.dump({"id": exp_id, "host": host, "error": "no_result_line"}, open(out_path, "w"), indent=2)
    sys.exit(1)
data = json.loads(m.group(1))
data["id"] = exp_id
data["host"] = host
json.dump(data, open(out_path, "w"), indent=2)
print(f"[{exp_id}] tok_s={data.get('tok_s'):.2f if data.get('tok_s') else None} tokens={data.get('tokens')} decode_s={data.get('decode_s'):.3f if data.get('decode_s') else None}")
PY
