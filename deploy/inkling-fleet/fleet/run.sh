#!/bin/bash
# Runs this box's Inkling rank (started by the cascadia-inkling systemd unit).
set -u
PREFIX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$PREFIX/rank.env"; set +a
# side-by-side OpenVINO runtime for this process only
if [ -n "${OVDIR:-}" ] && [ -f "$OVDIR/setupvars.sh" ]; then set +u; source "$OVDIR/setupvars.sh" > /dev/null; set -u; fi
args=(worker --rank "$RANK" --total "$TOTAL" --engine sparse-moe --device CPU --model "$PREFIX/model"
      --layer-start "$LAYER_START" --layer-end "$LAYER_END")
[ "$RANK" -gt 0 ] && args+=(--listen ":$((9100 + RANK))")
[ -n "${NEXT:-}" ] && args+=(--next "$NEXT")
[ "$RANK" = 0 ] && args+=(--api ":8000")
exec "$PREFIX/cascadia" "${args[@]}"
