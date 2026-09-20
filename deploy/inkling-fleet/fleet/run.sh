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
# What the dashboard shows for this box. --device stays CPU (the engine's own device: routing and the CPU-side
# expert layers); these two flags only label the card, and only with what this rank is really set up to use.
if [ "${CASCADIA_INKLING_OV_ATTN:-0}" = 1 ]; then
  args+=(--advertise-device iGPU)
  [ -n "${CASCADIA_INKLING_OV_MOE_LAYERS:-}" ] && args+=(--advertise-engines "sparse-moe,fused-moe")
fi
# Rank 0 is the only rank that dials without first waiting for anyone, and it starts instantly. After a
# "systemctl restart" it would reach rank 1 while that process is still exiting (its listener is open for a
# moment longer), then sit on a dead link until the first request makes it re-dial - and until then the ranks
# behind it wait unloaded. Give them time to come back first (they restart 5 s after they notice).
[ "$RANK" = 0 ] && sleep 8
exec "$PREFIX/cascadia" "${args[@]}"
