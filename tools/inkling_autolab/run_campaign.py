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
from datetime import datetime, timezone
from pathlib import Path

from autolab.core.campaign import Campaign
from autolab.core.loop import ResearchLoop


def full_model_target_met(campaign: Campaign, metrics: dict) -> bool:
    """Only a repeated, verified full large-model decode may satisfy 25 tok/s."""
    return (
        campaign.config.get('measurement_scope') == 'full_large_model_decode'
        and campaign.primary_metric == 'decode_tokens_per_s'
        and campaign.metric_direction == 'maximize'
        and bool(campaign.defaults.get('expected_hash'))
        and metrics.get('output_hash') == campaign.defaults['expected_hash']
        and metrics.get('full_model') == 1
        and metrics.get('correctness_verified') == 1
        and metrics.get('decode_steps_min', 0) >= 32
        and metrics.get('repetitions', 0) >= 3
        and math.isfinite(metrics.get('decode_tokens_per_s', 0))
        and metrics.get('decode_tokens_per_s', 0) >= 25
    )


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
        destination = root / 'results' / f'{campaign.name}.json'
        destination.parent.mkdir(exist_ok=True)
        state_dir = root / '.autolab'
        state_dir.mkdir(exist_ok=True)
        state_path = state_dir / 'state.json'

        target_reached = False

        def checkpoint(status: str) -> None:
            history = loop.db.load_history(campaign.name)
            temp = destination.with_suffix('.json.tmp')
            temp.write_text(json.dumps(history, indent=2) + '\n')
            temp.replace(destination)
            state_path.write_text(json.dumps({
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'last_campaign': campaign.name,
                'campaign_status': status,
                'experiments_recorded': len(history),
                'experiments_planned': campaign.experiment_count(),
                'target_full_model_decode_tokens_per_s': 25,
                'full_model_target_reached': target_reached,
                'measurement_scope': campaign.config.get('measurement_scope', 'synthetic_resident_layer'),
                'full_checkpoint_available': any(r.get('metrics', {}).get('full_model') == 1 for r in history),
            }, indent=2) + '\n')

        checkpoint('running')
        try:
            loop.run_campaign(args.campaign.resolve())
        except BaseException:
            checkpoint('interrupted')
            raise
        checkpoint('executed')
        history = loop.db.load_history(campaign.name)
        expected = campaign.defaults.get('expected_hash')
        for result in history:
            m = result.get('metrics', {})
            rate = m.get(campaign.primary_metric, 0)
            if (result['status'] != 'completed' or not math.isfinite(rate) or rate <= 0
                    or (expected and not m.get('output_hash'))):
                raise SystemExit(f"Invalid result: {result['experiment_name']}; inspect {destination}")
            validation = campaign.config.get('validation_metric')
            if validation and m.get(validation) != 1:
                raise SystemExit(f"Numerical oracle failed: {result['experiment_name']}")
            if expected and m['output_hash'] != expected:
                raise SystemExit(f"Correctness gate failed: {result['experiment_name']}")
        if len(history) != campaign.experiment_count():
            raise SystemExit('Incomplete sweep: do not promote a winner')
        choose = min if campaign.metric_direction == 'minimize' else max
        best = choose(history, key=lambda r: r['metrics'][campaign.primary_metric])
        target_reached = full_model_target_met(campaign, best['metrics'])
        checkpoint('verified')
        print('VERIFIED_BEST=' + json.dumps({'experiment': best['experiment_name'], 'metrics': best['metrics']}))


if __name__ == '__main__':
    main()
