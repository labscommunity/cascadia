#!/usr/bin/env python3
"""Extend the last published overrides with read-only fleet state capture."""
import argparse
from pathlib import Path
import re

FLEET = Path(__file__).resolve().parents[2] / 'deploy/inkling-fleet/fleet'


def build(base, boundary_roles=(), max_mb=2048):
    if 'LAB-PROXY:PORT' in base:
        raise ValueError('use the real last-published overrides, not a redacted copy')
    if 'AUTOLAB_CAPTURE_SERVER_PY' in base:
        raise ValueError('capture block already present; edit the existing block')
    relay = (FLEET / 'api_relay.py').read_text().rstrip()
    pattern = r"(?s)(<<'API_RELAY_PY'\n).*?(\nAPI_RELAY_PY\n)"
    base, count = re.subn(pattern, lambda m: m[1] + relay + m[2], base)
    if count != 1:
        raise ValueError('expected exactly one existing API relay block')
    target = '--upstream "${FLEET}-rank-${_up}:8000"'
    if base.count(target) != 1:
        raise ValueError('expected exactly one existing relay invocation')
    base = base.replace(target, target + ' --capture-fleet "$FLEET" --capture-port 9204')
    roles = sorted(set(boundary_roles) | {10})
    conditions = ' || '.join('[ "$RANK" = "%d" ]' % r for r in roles)
    source = (FLEET / 'capture_server.py').read_text().rstrip()
    return base + '''
# autolab 037: bounded residual files, read-only via the existing entry relay.
if %(conditions)s; then
  CASCADIA_STREAMS_CAPTURE_MAX_MB=%(max_mb)d
  if [ "$RANK" = 10 ]; then
    CASCADIA_STREAMS_CAPTURE_FINAL="$PREFIX/autolab-capture"
  else
    CASCADIA_STREAMS_CAPTURE_BOUNDARY="$PREFIX/autolab-capture"
  fi
  mkdir -p "$PREFIX/autolab-capture" /run/cascadia-inkling
  cat > /run/cascadia-inkling/capture_server.py <<'AUTOLAB_CAPTURE_SERVER_PY'
%(source)s
AUTOLAB_CAPTURE_SERVER_PY
  ( set +e +u
    while :; do
      /usr/bin/python3 /run/cascadia-inkling/capture_server.py --dir "$PREFIX/autolab-capture" --port 9204
      sleep 5
    done ) &
fi
''' % dict(conditions=conditions, max_mb=max_mb, source=source)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('base'); ap.add_argument('out')
    ap.add_argument('--boundary', default='', help='comma-separated pipeline roles besides the final rank')
    ap.add_argument('--max-mb', type=int, default=2048)
    a = ap.parse_args()
    roles = [int(r) for r in a.boundary.split(',') if r]
    if any(r not in range(11) for r in roles) or a.max_mb <= 0:
        ap.error('roles must be 0..10 and max-mb positive')
    Path(a.out).write_text(build(Path(a.base).read_text(), roles, a.max_mb))
    print(a.out)


if __name__ == '__main__':
    main()
