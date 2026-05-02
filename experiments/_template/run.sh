#!/bin/bash
# Experiment runner — keep self-contained so any future re-run is a one-liner.
# Conventions:
#   - Time the wallclock; log to logs/run.log
#   - Emit results.json at the end with at least: tok_s, tokens, time_s, model, hw, hash
#   - Exit non-zero on any failure
set -euo pipefail
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$EXP_DIR/logs"
# (fill in the actual command(s))
echo "TODO: implement"
