# Policy fixtures

Copied, unmodified:

- `intel-pi05-libero-manifest.json`: `manifest.json` of Hugging Face `OpenVINO/pi05-libero-fp16-ov`.
- `intel-smolvla-libero-manifest.json`: `manifest.json` of Hugging Face `OpenVINO/smolvla-libero-fp16-ov`.
- `pi05-droid-placeholder-manifest.json`: our `lerobot/pi05_droid` export (physicalai-train 0.3.0),
  whose stats are identity placeholders; the loader must refuse it.

Fixtures generated from Intel's `physicalai` runtime by `tools/vla_reference.py` land here too.
