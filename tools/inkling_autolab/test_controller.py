"""The controller must retain results on interruption and reject false wins."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location('controller', Path(__file__).with_name('run_campaign.py'))
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def run(tmp_path, monkeypatch, metrics, *, interrupted=False, scope='layer'):
    campaign = tmp_path/'campaigns/test.yaml'
    campaign.parent.mkdir()
    config = {'name': 'test', 'grid': {'setting': [1]}, 'metrics': {'primary': 'rate', 'direction': 'maximize'}}
    if scope == 'layer':
        config['defaults'] = {'expected_hash': 'abcdef'}
    else:
        config['validation_metric'] = 'validation_passed'
    campaign.write_text(yaml.safe_dump(config))
    record = {'experiment_name': 'test_setting=1', 'campaign_name': 'test',
              'status': 'completed', 'metrics': metrics}

    class DB:
        def load_history(self, name):
            return [record]

    class Loop:
        def __init__(self, path):
            self.db = DB()

        def run_campaign(self, path):
            if interrupted:
                raise KeyboardInterrupt

    monkeypatch.setattr(controller, '__file__', str(tmp_path/'run_campaign.py'))
    monkeypatch.setattr(controller, 'ResearchLoop', Loop)
    monkeypatch.setattr(sys, 'argv', ['run_campaign.py', str(campaign)])
    controller.main()


@pytest.mark.parametrize('metrics', [
    {'rate': float('nan'), 'output_hash': 'abcdef'},
    {'rate': 0, 'output_hash': 'abcdef'},
    {'rate': 999, 'output_hash': 'wrong'},
    {'rate': 999},
])
def test_invalid_metrics_cannot_be_promoted(tmp_path, monkeypatch, metrics):
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch, metrics)


def test_interruption_preserves_raw_records_and_target_status(tmp_path, monkeypatch):
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, monkeypatch, {'rate': 10, 'output_hash': 'abcdef'}, interrupted=True)
    assert len(json.loads((tmp_path/'results/test.json').read_text())) == 1
    state = json.loads((tmp_path/'.autolab/state.json').read_text())
    assert state['campaign_status'] == 'interrupted'
    assert state['full_model_target_reached'] is False


def test_fast_component_is_not_a_full_model_target_result(tmp_path, monkeypatch):
    run(tmp_path, monkeypatch, {'rate': 100, 'validation_passed': 1}, scope='expert')
    state = json.loads((tmp_path/'.autolab/state.json').read_text())
    assert state['campaign_status'] == 'verified'
    assert state['full_model_target_reached'] is False


def test_failed_numerical_oracle_rejects_fast_component(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run(tmp_path, monkeypatch, {'rate': 100, 'validation_passed': 0}, scope='expert')


def test_target_requires_full_model_proof(tmp_path):
    path = tmp_path/'full.yaml'
    path.write_text(yaml.safe_dump({
        'name': 'full', 'measurement_scope': 'full_large_model_decode',
        'defaults': {'expected_hash': 'abcdef'},
        'metrics': {'primary': 'decode_tokens_per_s', 'direction': 'maximize'},
    }))
    campaign = controller.Campaign(path)
    valid = {'output_hash': 'abcdef', 'full_model': 1, 'correctness_verified': 1,
             'decode_steps_min': 63, 'repetitions': 3, 'decode_tokens_per_s': 25.1}
    assert controller.full_model_target_met(campaign, valid)
    for missing in valid:
        reduced = dict(valid)
        del reduced[missing]
        assert not controller.full_model_target_met(campaign, reduced), missing
    for key, value in [('full_model', 0), ('decode_steps_min', 31), ('repetitions', 2),
                       ('decode_tokens_per_s', 24.9), ('decode_tokens_per_s', float('inf')),
                       ('output_hash', 'wrong')]:
        assert not controller.full_model_target_met(campaign, dict(valid, **{key: value}))
