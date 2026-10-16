#!/usr/bin/env python3
"""Compose a VLM export's text embeddings into its language model: a text-only
draft for ov-dist-spec.

optimum-intel exports a vision-language model (e.g. Ministral-3-3B, a
Mistral3ForConditionalGeneration) as separate IRs; its language model
(`openvino_language_model.xml`) takes `inputs_embeds`, so it cannot serve as
an ov-dist-spec draft, which is fed token ids. This tool wires the text
embedding IR (`openvino_text_embeddings_model.xml`: ids -> rows) in front of
the language model and saves one stateful `openvino_model.xml` that takes
`input_ids`, `attention_mask`, `position_ids` and `beam_idx` and keeps the
language model's KV state and logits.

The draft never sees images: ov-dist-spec feeds it the target's expanded
prompt ids, image soft tokens included as ordinary tokens. Only the target's
verification decides the output, so the draft affects the acceptance rate,
never the text.

Usage:
  python make_text_draft.py --model <OV VLM export> --output-dir <draft dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import openvino as ov
from openvino import opset13 as ops

LM_XML = "openvino_language_model.xml"
EMBED_XML = "openvino_text_embeddings_model.xml"
OUT_XML = "openvino_model.xml"

COPY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "chat_template.json",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _names(port) -> set[str]:
    return set(port.get_names())


def _param_named(model: ov.Model, name: str):
    for p in model.get_parameters():
        if name in _names(p.output(0)) or p.get_friendly_name() == name:
            return p
    return None


def compose(embed: ov.Model, lm: ov.Model) -> ov.Model:
    """`lm` with its `inputs_embeds` Parameter replaced by `embed`'s rows of a
    new `input_ids` Parameter. Results, KV sinks and the other inputs are the
    language model's own."""
    if len(embed.get_parameters()) != 1 or len(embed.outputs) != 1:
        raise SystemExit(
            f"{EMBED_XML} should map one ids input to one output; it has "
            f"{len(embed.get_parameters())} inputs and {len(embed.outputs)} outputs")
    embeds_param = _param_named(lm, "inputs_embeds")
    if embeds_param is None:
        raise SystemExit(f"{LM_XML} has no inputs_embeds input; is this a VLM language model?")
    if _param_named(lm, "input_ids") is not None:
        raise SystemExit(f"{LM_XML} already takes input_ids")

    old_ids = embed.get_parameters()[0]
    ids = ops.parameter(ov.PartialShape([-1, -1]), ov.Type.i64, name="input_ids")
    ids.output(0).set_names({"input_ids"})
    src = ids.output(0)
    if old_ids.get_element_type() != ov.Type.i64:
        src = ops.convert(src, old_ids.get_element_type()).output(0)
    for tgt in list(old_ids.output(0).get_target_inputs()):
        tgt.replace_source_output(src)

    rows = embed.get_results()[0].input_value(0)
    want = embeds_param.get_element_type()
    if rows.get_element_type() != want:
        rows = ops.convert(rows, want).output(0)
    for tgt in list(embeds_param.output(0).get_target_inputs()):
        tgt.replace_source_output(rows)

    params = [ids] + [p for p in lm.get_parameters() if p is not embeds_param]
    return ov.Model(list(lm.get_results()), list(lm.get_sinks()), params,
                    "text_draft")


def flat_text_config(cfg: dict) -> dict:
    tc = dict(cfg.get("text_config") or cfg)
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if k not in tc and k in cfg:
            tc[k] = cfg[k]
    return tc


def run(model_dir: str, out_dir: str) -> None:
    for fn in (LM_XML, EMBED_XML):
        if not os.path.exists(os.path.join(model_dir, fn)):
            raise SystemExit(f"{model_dir} lacks {fn}; not an OpenVINO VLM export")
    if os.path.abspath(model_dir) == os.path.abspath(out_dir):
        raise SystemExit("--output-dir must differ from --model")
    core = ov.Core()
    lm = core.read_model(os.path.join(model_dir, LM_XML))
    embed = core.read_model(os.path.join(model_dir, EMBED_XML))
    model = compose(embed, lm)
    os.makedirs(out_dir, exist_ok=True)
    ov.save_model(model, os.path.join(out_dir, OUT_XML), compress_to_fp16=False)
    inputs = [p.get_friendly_name() for p in model.get_parameters()]
    log(f"draft: inputs={inputs} states={len(model.get_sinks())} "
        f"outputs={[o.get_any_name() for o in model.outputs]}")

    for fn in COPY_FILES:
        src = os.path.join(model_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fn))
    cfg_path = os.path.join(model_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        shutil.copy2(cfg_path, os.path.join(out_dir, "config.vlm.json"))
        # Text tower at the top level: ov-dist-spec reads vocab_size here to
        # refuse prompt ids the draft cannot embed.
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(flat_text_config(cfg), f, indent=2)
    log(f"DRAFT DONE -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="OpenVINO VLM export directory")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    run(args.model, args.output_dir)


if __name__ == "__main__":
    main()
