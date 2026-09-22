#!/usr/bin/env python3
"""Extract only non-sensitive cumulative expert-use summaries from telemetry."""
import argparse
import json
from pathlib import Path


def summaries(records):
    layers = {}
    for record in records:
        for p in record.get('profs', []):
            if not str(p.get('at', '')).startswith('EU') or 'layer' not in p:
                continue
            layer = int(p['layer'])
            if p.get('rows', 0) <= layers.get(layer, {}).get('rows', 0):
                continue
            layers[layer] = {k: int(p[k]) for k in
                             ('layer', 'rows', 'top16_ppm', 'top32_ppm', 'top64_ppm', 'max_ppm', 'distinct')}
    return [layers[k] for k in sorted(layers)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('telemetry', type=Path, help='lab.py JSONL outside the repository')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    rows = summaries(json.loads(line) for line in a.telemetry.read_text().splitlines() if line.strip())
    a.out.write_text(json.dumps(rows, indent=2) + '\n')
    print('layers with records:', len(rows), '; top-64 share over 60%:', sum(r['top64_ppm'] > 600000 for r in rows))
    if rows:
        shares = sorted(r['top64_ppm'] / 10000 for r in rows)
        print('top-64 share %, min / median / max:', shares[0], shares[len(shares)//2], shares[-1])
        print('routing rows per layer, min / max:', min(r['rows'] for r in rows), max(r['rows'] for r in rows))
    if {r['layer'] for r in rows} != set(range(2, 66)):
        raise SystemExit('incomplete: require records from all 64 MoE layers')
