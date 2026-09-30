#!/usr/bin/env python3
"""Add a bounded, one-rank attention benchmark; serving IRs remain untouched."""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(base):
    if 'LAB-PROXY:PORT' in base or 'AUTOLAB_ATTENTION_BENCH_PY' in base:
        raise ValueError('use real latest overrides without a benchmark block')
    result = base + '''
# autolab 041: rank 5 only, benchmark int8/int4 projections before loading.
if [ "$RANK" = 5 ]; then
  ( set +e +u
    _m="$PREFIX/attention-bench-041"
    mkdir -p "$_m" /run/cascadia-inkling
    _n=$(( $(cat "$_m/starts" 2>/dev/null || echo 0) + 1 ))
    echo "$_n" > "$_m/starts"
    _died=0
    if [ -e "$_m/trying" ]; then
      _died=$(journalctl -u cascadia-inkling --since "@$(stat -c %Y "$_m/trying")" -q --no-pager 2>/dev/null | grep -c -E 'code=dumped|code=killed|status=11/SEGV|status=6/ABRT|status=101')
    fi
    if [ "$_n" -gt 12 ] || [ "${_died:-0}" -gt 0 ]; then touch "$_m/off"; fi
    [ -e "$_m/trying" ] || [ -e "$_m/off" ] || {
      touch "$_m/trying"
'''
    for path, tag in [(HERE.parents[1] / 'tools/inkling_attn_ov.py', 'AUTOLAB_ATTENTION_IR_PY'),
                      (HERE / 'attention_quant_bench.py', 'AUTOLAB_ATTENTION_BENCH_PY')]:
        result += "      cat > /run/cascadia-inkling/%s <<'%s'\n%s\n%s\n" % (path.name, tag, path.read_text().rstrip(), tag)
    result += '''      PYTHONPATH="$PREFIX/pylib" timeout 600 /usr/bin/python3 /run/cascadia-inkling/attention_quant_bench.py --model "$PREFIX/model" --layers 30,31 --out "$_m/result.json"
      _rc=$?
      [ "$_rc" = 0 ] || touch "$_m/off"
      echo "AQ-exit probe stage profile rc=$_rc"
    }
  )
fi
'''
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('base'); ap.add_argument('out')
    a = ap.parse_args()
    Path(a.out).write_text(build(Path(a.base).read_text()))
