#!/usr/bin/env python3
"""Prepare a once-only role-5 CPU/device overlap probe; default serving unchanged."""
import argparse
from pathlib import Path


def build(base):
    if 'LAB-PROXY:PORT' in base or 'AUTOLAB_CPU_OVERLAP_044' in base:
        raise ValueError('use real latest overrides without an existing 044 block')
    return base + '''
# AUTOLAB_CPU_OVERLAP_044: only role 5, at most one attempted diagnostic.
CASCADIA_INKLING_EXPERT_COUNTS=
CASCADIA_STREAMS_CAPTURE_FINAL=
CASCADIA_STREAMS_CAPTURE_BOUNDARY=
CASCADIA_INKLING_CPU_OVERLAP_BENCH=0
if [ "$RANK" = 5 ]; then
  ( set +e +u
    _m="$PREFIX/cpu-overlap-044"
    mkdir -p "$_m"
    _n=$(( $(cat "$_m/starts" 2>/dev/null || echo 0) + 1 ))
    echo "$_n" > "$_m/starts"
    _died=0
    if [ -e "$_m/trying" ]; then
      _died=$(journalctl -u cascadia-inkling --since "@$(stat -c %Y "$_m/trying")" -q --no-pager 2>/dev/null | grep -c -E 'code=dumped|code=killed|status=11/SEGV|status=6/ABRT|status=101')
    fi
    if [ "$_n" -gt 12 ] || [ "${_died:-0}" -gt 0 ]; then touch "$_m/off"; fi
    if [ ! -e "$_m/trying" ] && [ ! -e "$_m/off" ]; then
      touch "$_m/trying" && touch "$_m/run-once"
    fi
  )
  if [ -e "$PREFIX/cpu-overlap-044/run-once" ] && [ ! -e "$PREFIX/cpu-overlap-044/off" ]; then
    rm -f "$PREFIX/cpu-overlap-044/run-once"
    CASCADIA_INKLING_CPU_OVERLAP_BENCH=1
  fi
fi
'''


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('base'); ap.add_argument('out')
    a = ap.parse_args()
    Path(a.out).write_text(build(Path(a.base).read_text()))
