import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from train_feature_head import FeatureHead, load_sequences, split_sequences, validate_disjoint
from train_mtp_adapter import AdaptedDepth
from mtp_score import rms


class FleetTrainingTests(unittest.TestCase):
    def test_actual_samples_align_with_two_step_labels_and_next_features(self):
        with tempfile.TemporaryDirectory() as directory:
            ids = torch.tensor([10, 11, 12, 13, 14, 15])
            raw = torch.arange(20, dtype=torch.float32).reshape(5, 4) + 1
            embedding = torch.arange(24, dtype=torch.float32).reshape(6, 4)
            samples = torch.tensor([-1, 12, 13, 14, 15])
            path = str(Path(directory) / 'p0042.safetensors')
            data = dict(tokens=ids, final_out=raw, embed_out=embedding, argmax=samples)
            save_file(data, path, metadata=dict(i='42', family='6', prompt_len='2'))
            rows = load_sequences(directory, torch.ones(4))
            self.assertEqual(rows[0]['ids'].tolist(), [13, 14, 15])
            torch.testing.assert_close(rows[0]['h'][0], rms(raw[1], torch.ones(4)))
            torch.testing.assert_close(rows[0]['target'][0], rms(raw[2], torch.ones(4)))
            torch.testing.assert_close(rows[0]['e'][0], embedding[2])
            data['argmax'] = torch.tensor([-1, 12, 999, 14, 15])
            save_file(data, path, metadata=dict(i='42', family='6', prompt_len='2'))
            with self.assertRaisesRegex(ValueError, 'alignment'):
                load_sequences(directory, torch.ones(4))

    def test_no_prompt_or_id_leakage_and_stratified_selection(self):
        records = [dict(i=i, family=i % 12, prompt=(100+i,)) for i in range(60)]
        train, valid = split_sequences(records)
        self.assertEqual(len(valid), 12)
        self.assertEqual({r['family'] for r in valid}, set(range(12)))
        validate_disjoint(train, valid)
        with self.assertRaisesRegex(ValueError, 'ID overlap'):
            validate_disjoint(records, records[:1])
        with self.assertRaisesRegex(ValueError, 'prompt overlap'):
            validate_disjoint(records, [dict(i=999, prompt=(100,))])

    def test_feature_predictor_learns_a_known_transition(self):
        torch.manual_seed(42)
        model = FeatureHead(4, 16)
        h, e = torch.randn(32, 4), torch.randn(32, 4)
        target = h + 0.2 * e
        initial = float((model(h,e)-target).square().mean().detach())
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(100):
            opt.zero_grad(); loss = (model(h,e)-target).square().mean()
            loss.backward(); opt.step()
        self.assertLess(float(loss.detach()), initial * 0.05)

    def test_zero_adapter_preserves_base_and_backward_reaches_adapter(self):
        class Base:
            hidden_norm = torch.ones(4)
            embed_norm = torch.ones(4)
            input_proj = torch.randn(4, 8)
            def block(self, x, flags):
                return torch.tanh(x)
        base = Base(); model = AdaptedDepth(base, rank=2)
        h,e = torch.randn(7,4), torch.randn(7,4)
        expected = base.block(torch.cat((rms(h,base.hidden_norm),rms(e,base.embed_norm)),-1) @ base.input_proj.t(), ())
        torch.testing.assert_close(model(h,e), expected, rtol=0, atol=0)
        model(h,e).square().mean().backward()
        self.assertGreater(float(model.up.weight.grad.abs().sum()), 0)
        self.assertIsNone(base.input_proj.grad)


if __name__ == '__main__':
    unittest.main()
