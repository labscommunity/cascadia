#!/usr/bin/env python3
"""Prepare a reversible, role-5-only int4 attention quality canary.

Requires the owner's numerical-policy decision before publishing. Original
IRs are untouched; an override selects the separate candidate directory.
"""
import argparse
from pathlib import Path

GENERATOR = Path(__file__).resolve().parents[2] / 'tools/inkling_attn_ov.py'


def build(base, generator=None):
    if 'LAB-PROXY:PORT' in base or 'AUTOLAB_ATTENTION_CANARY_043' in base:
        raise ValueError('use latest real overrides without an existing 043 block')
    source = GENERATOR.read_text() if generator is None else generator
    return base + '''
# AUTOLAB_ATTENTION_CANARY_043: only role 5; keep original int8 IRs.
CASCADIA_INKLING_EXPERT_COUNTS=
CASCADIA_STREAMS_CAPTURE_FINAL=
CASCADIA_STREAMS_CAPTURE_BOUNDARY=
if [ "$RANK" = 5 ]; then
  ( set +e +u
    _m="$PREFIX/attention-canary-043"
    mkdir -p "$_m" /run/cascadia-inkling
    _n=$(( $(cat "$_m/starts" 2>/dev/null || echo 0) + 1 ))
    echo "$_n" > "$_m/starts"
    _died=0
    if [ -e "$_m/trying" ]; then
      _died=$(journalctl -u cascadia-inkling --since "@$(stat -c %Y "$_m/trying")" -q --no-pager 2>/dev/null | grep -c -E 'code=dumped|code=killed|status=11/SEGV|status=6/ABRT|status=101')
    fi
    if [ "$_n" -gt 12 ] || [ "${_died:-0}" -gt 0 ]; then touch "$_m/off"; fi
    if [ ! -e "$_m/trying" ] && [ ! -e "$_m/off" ]; then
      touch "$_m/trying"
      cat > /run/cascadia-inkling/attention_canary_generator.py <<'AUTOLAB_ATTENTION_CANARY_PY'
''' + source.rstrip() + '''
AUTOLAB_ATTENTION_CANARY_PY
      PYTHONPATH="$PREFIX/pylib" timeout 600 /usr/bin/python3 /run/cascadia-inkling/attention_canary_generator.py --src "$PREFIX/model" --out "$PREFIX/model" --layers 30,31,32,33,34,35 --weights int4 --dir-name attn_ov_int4_043
      _rc=$?
      if [ "$_rc" = 0 ]; then
        for _l in 30 31 32 33 34 35; do
          for _part in qkvr o; do
            for _ext in xml bin; do
              [ -s "$PREFIX/model/attn_ov_int4_043/layer_$_l/$_part/openvino_model.$_ext" ] || _rc=1
            done
          done
        done
      fi
      if [ "$_rc" = 0 ]; then touch "$_m/ready"; else touch "$_m/off"; fi
    fi
  )
  if [ -e "$PREFIX/attention-canary-043/ready" ] && [ ! -e "$PREFIX/attention-canary-043/off" ]; then
    CASCADIA_INKLING_OV_ATTN_DIR=attn_ov_int4_043
    echo "AC043-on probe stage profile enabled=1 layers=6"
  else
    echo "AC043-off probe stage profile enabled=0"
  fi
fi
'''


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('base'); ap.add_argument('out')
    a = ap.parse_args()
    Path(a.out).write_text(build(Path(a.base).read_text()))
