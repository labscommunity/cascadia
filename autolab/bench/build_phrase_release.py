#!/usr/bin/env python3
"""Add an idempotent, verified phrase-table transfer to the last published overrides."""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(base):
    if 'LAB-PROXY:PORT' in base or 'AUTOLAB_PHRASE_TRANSFER_PY' in base:
        raise ValueError('expected real base overrides without a phrase-transfer block')
    merged = base + '\n# autolab 038: preserve what both pipeline-head boxes learned.\n'
    merged += 'if [ "${BOX_RANK:-$RANK}" = 0 ] || [ "$RANK" = 0 ]; then\n  mkdir -p /run/cascadia-inkling\n'
    for filename, tag in [('ngram_merge.py', 'AUTOLAB_NGRAM_MERGE_PY'), ('phrase_transfer.py', 'AUTOLAB_PHRASE_TRANSFER_PY')]:
        merged += "  cat > /run/cascadia-inkling/%s <<'%s'\n%s\n%s\n" % (filename, tag, (HERE / filename).read_text().rstrip(), tag)
    merged += '''  if [ "${BOX_RANK:-$RANK}" = 0 ]; then
    ( set +e +u
      /usr/bin/python3 /run/cascadia-inkling/phrase_transfer.py serve "$PREFIX/spec-ngrams.bin" /run/cascadia-inkling/ngram-transfer --port 9205
    ) &
  elif [ "$RANK" = 0 ]; then
    # Bounded to 90 seconds, before the worker loads its in-memory table.
    ( set +e +u
      /usr/bin/python3 /run/cascadia-inkling/phrase_transfer.py fetch "$PREFIX/spec-ngrams.bin" "http://${FLEET}-rank-0:9205" "$PREFIX/ngram-transfer" || echo 'NG8-failed probe stage profile transferred=0 failed=1'
    )
  fi
fi
'''
    return merged


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('base'); ap.add_argument('out')
    a = ap.parse_args()
    Path(a.out).write_text(build(Path(a.base).read_text()))
    print(a.out)
