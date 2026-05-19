#!/bin/bash
# K2.6 eyeball-eval CLI helper — print baseline vs feature outputs
# side-by-side from a quality-eval run dir.
#
# Why this exists:
#   iter 037 showed substring eval can pass garbage. The other end of
#   the spectrum is "just let a human look". This script prints both
#   responses next to each other with a divergence marker, so quality
#   regressions that aren't caught by any metric still surface in the
#   reviewer's eyes.
#
# Usage:
#   bash autolab/bench/k26_eyeball_eval.sh /tmp/k26-qeval
#   bash autolab/bench/k26_eyeball_eval.sh /tmp/k26-qeval --idx 4
#   bash autolab/bench/k26_eyeball_eval.sh /tmp/k26-qeval --width 60
#
# Expects:
#   $1/comparison.jsonl  produced by k26_quality_eval.sh
#
# Output: stdout. Each prompt prints:
#   ----
#   idx N | prompt: ...
#   first divergence at word M, byte K
#
#   BASELINE                       | FEATURE
#   <wrapped>                      | <wrapped>
#   ...

set -u

DIR=""
ONLY_IDX=""
WIDTH=60

while [ $# -gt 0 ]; do
    case "$1" in
        --idx)   ONLY_IDX="$2"; shift 2;;
        --width) WIDTH="$2";    shift 2;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *) DIR="$1"; shift;;
    esac
done

if [ -z "$DIR" ]; then
    echo "usage: $0 DIR [--idx N] [--width W]" >&2
    exit 1
fi
CMP="$DIR/comparison.jsonl"
if [ ! -f "$CMP" ]; then
    echo "ERROR: $CMP not found — run k26_quality_eval.sh first" >&2
    exit 1
fi

# Render via python (cleanest for column wrapping + unicode-safe widths).
python3 - "$CMP" "$WIDTH" "${ONLY_IDX:--1}" <<'PYEOF'
import json, sys, textwrap

cmp_path = sys.argv[1]
width = int(sys.argv[2])
only_idx = int(sys.argv[3])

with open(cmp_path) as f:
    rows = [json.loads(l) for l in f if l.strip()]

for r in rows:
    if only_idx >= 0 and r['idx'] != only_idx:
        continue
    print('=' * (2 * width + 5))
    print(f"idx {r['idx']:3d} | flag={r['flag']:<14s} | div_word={r['first_divergence_words']:>4d} div_byte={r['first_divergence_bytes']:>4d}")
    print(f"prompt: {r['prompt']}")
    print(f"bytes:    baseline={r['baseline_bytes']:>5d}  feature={r['feature_bytes']:>5d}  len_frac={r['length_fraction']}")
    print('-' * (2 * width + 5))

    b_wrap = textwrap.wrap(r['baseline_content'] or '<empty>', width) or ['<empty>']
    f_wrap = textwrap.wrap(r['feature_content']  or '<empty>', width) or ['<empty>']
    n = max(len(b_wrap), len(f_wrap))
    for i in range(n):
        bl = b_wrap[i] if i < len(b_wrap) else ''
        fl = f_wrap[i] if i < len(f_wrap) else ''
        print(f"{bl:<{width}}  |  {fl:<{width}}")
    print()
PYEOF
