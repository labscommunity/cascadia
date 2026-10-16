#!/usr/bin/env python3
"""Cut an OpenVINO VLM export's language model into ov-runtime stages.

optimum-intel exports a vision-language model as three IRs: the text
embedding table (`openvino_text_embeddings_model.xml`), the vision encoder
(`openvino_vision_embeddings_model.xml`) and a stateful language model that
takes `inputs_embeds`, `attention_mask`, `position_ids` and `beam_idx`
(`openvino_language_model.xml`). This tool slices the language model at
decoder-layer boundaries into the v3 tree ov-runtime loads: stage 0 keeps
`inputs_embeds` (rank 0 embeds the prompt and splices image rows itself),
later stages take `hidden_states`, the last ends in last-position logits.

The slicing is the Gemma 4 surgery's `extract_stage` (each stage owns the KV
state of its own layers; the cut is the residual stream entering a layer's
input RMSNorm), which does not depend on the architecture as long as the
layers are named `...layers.N.input_layernorm` and no KV is shared across
layers.

Families (the `vision.family` rank 0 serves):
  pixtral   LLaVA layout, Mistral text tower, Pixtral encoder (e.g.
            OpenVINO/pixtral-12b-int4-ov). Causal attention, so no
            token_type_ids frame.
  mistral3  Mistral3ForConditionalGeneration (e.g. Ministral-3, exported with
            optimum-intel): the Pixtral encoder plus a 2x2 patch merger in
            openvino_multi_modal_projector_model.xml, which rank 0 runs.

Usage:
  python cut_vlm_lm.py --model <OV VLM dir> --output-dir <tree> --num-stages N \
      --family pixtral [--stage I]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import openvino as ov

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "gemma4_surgery"))
import export_gemma4_text as g  # noqa: E402  (extract_stage, stage_ranges)

EXPORT_VERSION = "vlm_lm_surgery_v1"

# family -> (model_type check, files rank 0 needs beside the stages)
FAMILIES = {
    "pixtral": {
        "model_types": ("llava",),
        "vision_types": ("pixtral",),
        "token_type_ids": False,
    },
    # Mistral 3 (e.g. Ministral-3): Pixtral tower + 2x2 patch merger; rank 0
    # also runs openvino_multi_modal_projector_model.xml.
    "mistral3": {
        "model_types": ("mistral3",),
        "vision_types": ("pixtral",),
        "token_type_ids": False,
    },
}

ROOT_FILES = (
    "openvino_text_embeddings_model.xml",
    "openvino_text_embeddings_model.bin",
    "openvino_vision_embeddings_model.xml",
    "openvino_vision_embeddings_model.bin",
    "openvino_multi_modal_projector_model.xml",
    "openvino_multi_modal_projector_model.bin",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "chat_template.json",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def flat_text_config(cfg: dict) -> dict:
    """The text tower's config at the top level (rotary, eos and layer counts
    for ov-runtime), keeping the ids the VLM wrapper carries."""
    tc = dict(cfg["text_config"])
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if k not in tc and k in cfg:
            tc[k] = cfg[k]
    return tc


def check_family(cfg: dict, family: str) -> None:
    spec = FAMILIES[family]
    mt = cfg.get("model_type", "")
    vt = (cfg.get("vision_config") or {}).get("model_type", "")
    if mt not in spec["model_types"] or vt not in spec["vision_types"]:
        raise SystemExit(
            f"--family {family} expects model_type in {spec['model_types']} with a "
            f"{spec['vision_types']} vision tower; this export is {mt!r} / {vt!r}"
        )
    if (cfg.get("text_config") or {}).get("num_kv_shared_layers"):
        raise SystemExit("cross-layer KV sharing cannot be cut by this tool")


def run(model_dir: str, out_dir: str, num_stages: int, family: str,
        suffix: str, only_stage: int | None) -> None:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    check_family(cfg, family)
    text = flat_text_config(cfg)
    num_layers = int(text["num_hidden_layers"])
    hidden = int(text["hidden_size"])
    lm_xml = os.path.join(model_dir, "openvino_language_model.xml")
    required = ["openvino_language_model.xml", "openvino_text_embeddings_model.xml",
                "openvino_vision_embeddings_model.xml"]
    if family == "mistral3":
        # Rank 0 reads spatial_merge_size from processor_config.json and runs
        # the projector on the merged rows; without them the tree would serve
        # as plain Pixtral and fail only at image time.
        required += ["openvino_multi_modal_projector_model.xml", "processor_config.json"]
    for fn in required:
        if not os.path.exists(os.path.join(model_dir, fn)):
            raise SystemExit(f"{model_dir} lacks {fn}; not an OpenVINO {family} VLM export")

    if num_stages > num_layers:
        raise SystemExit(f"--num-stages {num_stages} exceeds the {num_layers} decoder layers")
    if only_stage is not None and not 0 <= only_stage < num_stages:
        raise SystemExit(f"--stage {only_stage} is not in 0..{num_stages - 1}")
    os.makedirs(out_dir, exist_ok=True)
    # Removed up front and written last, so a run that dies (a whole cut or a
    # one-stage re-cut) never leaves a tree that reads as complete.
    marker = os.path.join(out_dir, "pipeline_config.json")
    if os.path.exists(marker):
        os.remove(marker)
    ranges = g.stage_ranges(num_stages, num_layers)
    stages_meta = []
    for i, (a, b) in enumerate(ranges):
        first, last = i == 0, i == len(ranges) - 1
        meta = {
            "stage": i, "layer_start": a, "layer_end": b + 1,
            "has_embed": first, "has_head": last, "stateful": True,
            "num_kv_heads": text.get("num_key_value_heads"),
            "head_dim": text.get("head_dim") or hidden // int(text["num_attention_heads"]),
            "export_version": EXPORT_VERSION,
        }
        stages_meta.append(meta)
        if only_stage is not None and i != only_stage:
            continue
        t0 = time.time()
        stage, state_vars, _ = g.extract_stage(lm_xml, a, b, first, last, suffix, True)
        sdir = os.path.join(out_dir, f"stage_{i}")
        os.makedirs(sdir, exist_ok=True)
        ov.save_model(stage, os.path.join(sdir, "openvino_model.xml"), compress_to_fp16=False)
        meta["inputs"] = [p.get_friendly_name() for p in stage.get_parameters()]
        meta["state_vars"] = len(state_vars)
        with open(os.path.join(sdir, "stage_config.json"), "w") as f:
            json.dump(meta, f, indent=2)
        log(f"stage_{i}: layers {a}..{b} saved in {time.time() - t0:.0f}s "
            f"inputs={meta['inputs']} states={len(state_vars)}")

    def put(src: str, dst: str) -> None:
        # Replace, never write through: dst may be a hardlink in another tree.
        if os.path.exists(dst):
            os.remove(dst)
        shutil.copy2(src, dst)

    for fn in ROOT_FILES:
        src = os.path.join(model_dir, fn)
        if os.path.exists(src):
            put(src, os.path.join(out_dir, fn))
    put(os.path.join(model_dir, "config.json"), os.path.join(out_dir, "config.vlm.json"))
    cfg_out = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_out):
        os.remove(cfg_out)
    with open(cfg_out, "w") as f:
        json.dump(text, f, indent=2)
    tok = os.path.join(out_dir, "tokenizer")
    os.makedirs(tok, exist_ok=True)
    for fn in TOKENIZER_FILES:
        src = os.path.join(model_dir, fn)
        if os.path.exists(src):
            put(src, os.path.join(tok, fn))
    put(cfg_out, os.path.join(tok, "config.json"))
    if not os.path.exists(os.path.join(tok, "tokenizer.json")):
        raise SystemExit(f"{model_dir} has no tokenizer.json; ov-runtime needs it")

    pipeline = {
        "model_id": os.path.basename(os.path.abspath(model_dir)),
        "num_stages": num_stages,
        "num_layers": num_layers,
        "hidden_size": hidden,
        "export_version": EXPORT_VERSION,
        "stages": stages_meta,
        "vision": {
            "family": family,
            "text_embeddings": "openvino_text_embeddings_model.xml",
            "vision_embeddings": "openvino_vision_embeddings_model.xml",
            "token_type_ids": FAMILIES[family]["token_type_ids"],
            "image_token_id": cfg.get("image_token_index"),
        },
    }
    with open(marker, "w") as f:
        json.dump(pipeline, f, indent=2)
    log(f"CUT DONE -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="OpenVINO VLM export directory")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-stages", type=int, default=2)
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--stage", type=int, default=None, help="cut only this stage")
    ap.add_argument("--boundary-suffix", default=g.DEFAULT_LAYERNORM_SUFFIX)
    args = ap.parse_args()
    # One stage serves a single box through ov-runtime: GenAI's own Pixtral
    # preprocessing differs from HF's (see docs/rfcs/vlm-support.md).
    if args.num_stages < 1:
        raise SystemExit("--num-stages must be >= 1")
    run(args.model, args.output_dir, args.num_stages, args.family,
        args.boundary_suffix, args.stage)


if __name__ == "__main__":
    main()
