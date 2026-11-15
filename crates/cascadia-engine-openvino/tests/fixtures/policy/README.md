# Policy fixtures

Copied, unmodified:

- `intel-pi05-libero-manifest.json`: `manifest.json` of Hugging Face `OpenVINO/pi05-libero-fp16-ov`.
- `intel-smolvla-libero-manifest.json`: `manifest.json` of Hugging Face `OpenVINO/smolvla-libero-fp16-ov`.
- `pi05-droid-placeholder-manifest.json`: our `lerobot/pi05_droid` export (physicalai-train 0.3.0),
  whose stats are identity placeholders; the loader must refuse it.
- `smolvla-libero-noise-manifest.json`: our export of `HuggingFaceVLA/smolvla_libero` with
  physical-ai-studio `8f132416` and the physicalai runtime `e3dd944`, dtype float32 with fp16
  weights. New style: `noise` is a model input (`[1, 50, 32]`) and cameras are named
  `images.images.<name>`.
- `pi0-libero-manifest.json`: our export of `lerobot/pi0_libero_finetuned_v044` (Intel's π0.5
  exporter at physical-ai-studio `8f132416`, patched with LeRobot's π0 action expert), f32 with
  fp16 weights, `noise` as an input, `policy.name` set to `pi0`.

Generated:

- `smolvla-libero-noise-reference-actions.json`: the served actions for the fixed inputs of
  `tests/policy_openvino.rs`, from an independent numpy + OpenVINO Python pipeline (own
  tokenizer call, image scaling, normalize, xoshiro noise and denormalize) on CPU at f32.

Fixtures generated from Intel's `physicalai` runtime by `tools/vla_reference.py` land here too.
