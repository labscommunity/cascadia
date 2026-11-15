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
- `xvla-libero-manifest.json`: our export of `lerobot/xvla-libero` (LeRobot 0.7.0 modules in
  Intel's manifest format), f32 with fp16 weights, `noise` as an input, `image2` flipped on the
  host by `image_flip`, actions served 7 wide by `xvla_rotation_6d_to_axis_angle`.
- `vlajepa-libero-manifest.json`: our export of `lerobot/VLA-JEPA-LIBERO` (Intel's draft VLA-JEPA
  exporter, physical-ai-studio PR 1416), f32 with fp16 weights, the prompt padded right to 288
  tokens for a static graph; state normalization and action denormalization are graph constants.

Generated:

- `xvla-rotation-reference.json`: rows of LeRobot 0.7.0's X-VLA `rotate6d_to_axis_angle`
  (`lerobot/policies/xvla/utils.py`, numpy 1.26.4) on float32 6-D rotations built from
  axis-angles near π and below, plus a scaled non-orthogonal row, the identity and a near-identity
  row.

- `smolvla-libero-noise-reference-actions.json`: the served actions for the fixed inputs of
  `tests/policy_openvino.rs`, from an independent numpy + OpenVINO Python pipeline (own
  tokenizer call, image scaling, normalize, xoshiro noise and denormalize) on CPU at f32.

Fixtures generated from Intel's `physicalai` runtime by `tools/vla_reference.py` land here too.
