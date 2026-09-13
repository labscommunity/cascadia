#!/usr/bin/env python3
"""Run one Autolab campaign, reject invalid output, and save portable results.

The current agent drives hypothesis/code revisions; Autolab executes and resumes
parameter grids. No second LLM process or separate API credential is required.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import math
from pathlib import Path

from autolab.core.campaign import Campaign
from autolab.core.loop import ResearchLoop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    # Only one campaign at a time on this single benchmark host.
    with (root / '.campaign.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Campaign(args.campaign.resolve())
        loop = ResearchLoop(root / 'results.db')
        loop.run_campaign(args.campaign.resolve())
        history = loop.db.load_history(campaign.name)
        destination = root / 'results' / f'{campaign.name}.json'
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps(history, indent=2) + '\n')
        expected = campaign.defaults.get('expected_hash')
        for result in history:
            m = result.get('metrics', {})
            rate = m.get('layer_tokens_per_s', 0)
            if (result['status'] != 'completed' or not math.isfinite(rate) or rate <= 0
                    or not m.get('output_hash')):
                raise SystemExit(f"Invalid result: {result['experiment_name']}; inspect {destination}")
            if expected and m['output_hash'] != expected:
                raise SystemExit(f"Correctness gate failed: {result['experiment_name']}")
        if len(history) != campaign.experiment_count():
            raise SystemExit('Incomplete sweep: do not promote a winner')
        best = max(history, key=lambda r: r['metrics']['layer_tokens_per_s'])
        print('VERIFIED_BEST=' + json.dumps({'experiment': best['experiment_name'], 'metrics': best['metrics']}))


if __name__ == '__main__':
    main()
