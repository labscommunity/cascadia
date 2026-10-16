#!/usr/bin/env python3
"""Cut an OpenVINO VLM export's language model into ov-runtime stages.

optimum-intel exports a vision-language model as three IRs: the text
embedding table (`openvino_text_embeddings_model.xml`), the vision encoder
(`openvino_vision_embeddings_model.xml`) and a stateful language model that
takes `inputs_embeds`, `attention_mask`, `position_ids` and `beam_idx`
(`openvino_language_model.xml`). This tool slices the language model at
decoder-layer boundaries into the v3 tree ov-runtime loads: stage 0 keeps
`inputs_embeds` (rank 0 embeds the prompt and splices image rows itself),
later stages take `hidden_states`, the last ends in last-position logits
(ov-runtime samples only the last row). `--all-logits` keeps the head's logits
for every fed position instead, which ov-dist-spec needs as a target: it
verifies each drafted token from its own row. The last stage's
`stage_config.json` records which with `"logits": "all" | "last"`.

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
  gemma3    Gemma3ForConditionalGeneration (e.g. OpenVINO/gemma-3-12b-it-int4-ov):
            SigLIP encoder whose IR already pools and projects to 256 rows
            per image. The language model takes token_type_ids (each image is
            attended both ways in its full and sliding masks), so every stage
            keeps them and they ride the wire as a third frame.
  llava_next LLaVA-NeXT / llava-v1.6 (e.g. OpenVINO/llava-v1.6-mistral-7b-hf-int4-ov):
            CLIP tower whose IR already ends in the projector, anyres tiles;
            rank 0 packs the tiles and inserts `image_newline`, which the
            export keeps in config.json (copied to config.vlm.json).
            Causal attention.
  phi3_v    Phi-3.5-vision (e.g. OpenVINO/Phi-3.5-vision-instruct-int4-ov):
            Phi-3 text tower (config at the top level), CLIP tower in the
            vision IR, MLP projector in openvino_vision_projection_model.xml;
            rank 0 does the HD crops and the glb_GN/sub_GN assembly between
            them (the rows are in config.vlm.json). The soft-token id is
            `<|image|>`; the tree stops on `<|end|>` as well as
            `<|endoftext|>`.
  internvl_chat
            InternVL2 with a Phi-3 text tower (e.g.
            OpenVINO/InternVL2-4B-int4-ov; text config under `llm_config`):
            the vision IR holds InternViT, the pixel shuffle and mlp1; rank 0
            tiles the image. Soft token `<IMG_CONTEXT>`.
  qwen2_5_vl  Qwen2.5-VL (e.g. OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov). Every
            stage takes M-RoPE `position_ids [3, B, T]`, which rank 0 computes
            and sends with each hidden; the stages record `position_rows: 3`
            and the pipeline `vision.mrope_rows: 3`. Rank 0 runs the patch
            embedding and the merger IR (window attention).
  qwen3_vl  Qwen3-VL (e.g. OpenVINO/Qwen3-VL-4B-Instruct-int4-ov): as
            qwen2_5_vl, plus DeepStack. Its language model adds the vision
            merger's `deepstack_feature_lists` row k to the image tokens after
            decoder layer k (k < len(deepstack_visual_indexes) = 3) through
            two inputs, `visual_pos_masks` and `deepstack_visual_embeds`. Those
            must stay in stage 0 (rank 0 holds the rows), so stage 0 must
            cover layers 0..2; `visual_pos_masks` becomes an i64 input
            converted to the IR's boolean inside the graph (the runtime feeds
            no boolean tensors).

Phi-3's su/longrope RoPE is traced into these exports with a constant
inv_freq (no switch to the long factors past
original_max_position_embeddings), so a cut serves them exactly as the
export does: HF-exact up to 4096 positions.

Usage:
  python cut_vlm_lm.py --model <OV VLM dir> --output-dir <tree> --num-stages N \
      --family pixtral|mistral3|gemma3|llava_next|phi3_v|internvl_chat|qwen2_5_vl|qwen3_vl \
      [--stage I] [--all-logits]
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
    # Phi-3.5-vision: the vision tower is named in img_processor, not
    # vision_config; rank 0 also runs openvino_vision_projection_model.xml.
    "phi3_v": {
        "model_types": ("phi3_v",),
        "vision_types": ("clip_vision_model",),
        "token_type_ids": False,
        "text_key": None,
        "requires": ("openvino_vision_projection_model.xml", "preprocessor_config.json"),
    },
    # InternVL2 (Phi-3 text tower): the vision IR ends in the projector.
    "internvl_chat": {
        "model_types": ("internvl_chat",),
        "vision_types": ("intern_vit_6b",),
        "token_type_ids": False,
        "text_key": "llm_config",
    },
    # Gemma 3: SigLIP tower; projector and 4x4 pool are inside the vision IR.
    "gemma3": {
        "model_types": ("gemma3",),
        "vision_types": ("siglip_vision_model",),
        "token_type_ids": True,
    },
    # LLaVA-NeXT: one IR (CLIP tower + projector); image_newline lives in
    # config.json, and rank 0 does the tiling and packing.
    "llava_next": {
        "model_types": ("llava_next",),
        "vision_types": ("clip_vision_model",),
        "token_type_ids": False,
    },
    "qwen2_5_vl": {
        "model_types": ("qwen2_5_vl",),
        "vision_types": ("qwen2_5_vl",),
        "token_type_ids": False,
        "mrope_rows": 3,
        "requires": ("openvino_vision_embeddings_merger_model.xml", "preprocessor_config.json"),
    },
    "qwen3_vl": {
        "model_types": ("qwen3_vl",),
        "vision_types": ("qwen3_vl",),
        "token_type_ids": False,
        "mrope_rows": 3,
        "requires": ("openvino_vision_embeddings_merger_model.xml",
                     "openvino_vision_embeddings_pos_model.xml", "preprocessor_config.json"),
    },
}


# Top-level phi3_v keys that are the vision side, not the text tower.
PHI3V_VISION_KEYS = ("glb_GN", "sub_GN", "img_processor", "embd_layer", "auto_map")


# Qwen3-VL's DeepStack inputs; they belong to stage 0 only.
DEEPSTACK_INPUTS = ("visual_pos_masks", "deepstack_visual_embeds")

ROOT_FILES = (
    "openvino_text_embeddings_model.xml",
    "openvino_text_embeddings_model.bin",
    "openvino_vision_embeddings_model.xml",
    "openvino_vision_embeddings_model.bin",
    "openvino_multi_modal_projector_model.xml",
    "openvino_multi_modal_projector_model.bin",
    "openvino_vision_embeddings_merger_model.xml",
    "openvino_vision_embeddings_merger_model.bin",
    "openvino_vision_embeddings_pos_model.xml",
    "openvino_vision_embeddings_pos_model.bin",
    "openvino_vision_projection_model.xml",
    "openvino_vision_projection_model.bin",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
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


def flat_text_config(cfg: dict, family: str = "pixtral") -> dict:
    """The text tower's config at the top level (rotary, eos and layer counts
    for ov-runtime), keeping the ids the VLM wrapper carries."""
    key = FAMILIES[family].get("text_key", "text_config")
    if key is None:
        # phi3_v keeps the text config at the top level, next to the vision
        # settings and the (large) separator rows.
        return {k: v for k, v in cfg.items() if k not in PHI3V_VISION_KEYS}
    tc = dict(cfg[key])
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if k not in tc and k in cfg:
            tc[k] = cfg[k]
    return tc


def check_family(cfg: dict, family: str) -> None:
    spec = FAMILIES[family]
    mt = cfg.get("model_type", "")
    if family == "phi3_v":
        vt = (cfg.get("img_processor") or {}).get("name", "")
    else:
        vt = (cfg.get("vision_config") or {}).get("model_type", "")
    if mt not in spec["model_types"] or vt not in spec["vision_types"]:
        raise SystemExit(
            f"--family {family} expects model_type in {spec['model_types']} with a "
            f"{spec['vision_types']} vision tower; this export is {mt!r} / {vt!r}"
        )
    if (cfg.get("text_config") or {}).get("num_kv_shared_layers"):
        raise SystemExit("cross-layer KV sharing cannot be cut by this tool")


def deepstack_layers(cfg: dict, family: str) -> int:
    """Decoder layers that take DeepStack rows (Qwen3-VL: one per vision
    feature level, added after layers 0..n-1), 0 for other families."""
    if family != "qwen3_vl":
        return 0
    n = len((cfg.get("vision_config") or {}).get("deepstack_visual_indexes") or [])
    if n == 0:
        raise SystemExit("qwen3_vl export has no vision_config.deepstack_visual_indexes")
    return n


def param_by_name(model, name: str):
    for p in model.get_parameters():
        if name in p.output(0).get_names() or p.get_friendly_name() == name:
            return p
    return None


def check_mrope_positions(stage, rows: int, idx: int) -> None:
    """Every stage of an M-RoPE family must take `position_ids [rows, ?, ?]`."""
    p = param_by_name(stage, "position_ids")
    if p is None:
        raise SystemExit(f"stage_{idx} has no position_ids input; not an M-RoPE language model")
    ps = p.get_partial_shape()
    if ps.rank.get_length() != 3 or not ps[0].is_static or ps[0].get_length() != rows:
        raise SystemExit(f"stage_{idx} position_ids is {ps}, expected [{rows}, ?, ?]")


def fix_deepstack_inputs(stage, first: bool, idx: int):
    """Stage 0 keeps the DeepStack inputs, with `visual_pos_masks` turned into
    an i64 input converted to the IR's boolean (the runtime feeds no boolean
    tensors); any later stage holding one is refused: its rows live on rank 0."""
    have = [n for n in DEEPSTACK_INPUTS if param_by_name(stage, n) is not None]
    if not first:
        if have:
            raise SystemExit(
                f"stage_{idx} takes {have}: DeepStack rows feed the first decoder layers, "
                "which must all sit in stage 0; use fewer stages or a later first cut")
        return stage
    if len(have) != len(DEEPSTACK_INPUTS):
        raise SystemExit(f"stage_0 lacks DeepStack inputs (has {have}); not a Qwen3-VL export")
    mask = param_by_name(stage, "visual_pos_masks")
    if mask.get_element_type() == ov.Type.boolean:
        from openvino import opset13 as ops
        new = ops.parameter(mask.get_partial_shape(), ov.Type.i64, name="visual_pos_masks")
        new.output(0).set_names({"visual_pos_masks"})
        conv = ops.convert(new.output(0), ov.Type.boolean)
        for tgt in list(mask.output(0).get_target_inputs()):
            tgt.replace_source_output(conv.output(0))
        stage.replace_parameter(stage.get_parameter_index(mask), new)
        stage.validate_nodes_and_infer_types()
    return stage
def special_token_id(model_dir: str, token: str) -> int:
    """The id of an added token in the export's tokenizer.json."""
    with open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8") as f:
        tok = json.load(f)
    for t in tok.get("added_tokens", []):
        if t.get("content") == token:
            return int(t["id"])
    raise SystemExit(f"{model_dir}/tokenizer.json has no {token} token")


def image_token_id(model_dir: str, cfg: dict, family: str):
    """The soft-token id rank 0 expands."""
    if family == "phi3_v":
        return special_token_id(model_dir, "<|image|>")
    if family == "internvl_chat":
        return cfg.get("img_context_token_id")
    if family in ("qwen2_5_vl", "qwen3_vl"):
        return cfg.get("image_token_id")
    return cfg.get("image_token_index")


def add_stop_token(path: str, token_id: int) -> None:
    """Add `token_id` to a generation_config.json's eos_token_id list."""
    with open(path, encoding="utf-8") as f:
        gen = json.load(f)
    eos = gen.get("eos_token_id")
    eos = [] if eos is None else (list(eos) if isinstance(eos, list) else [eos])
    if token_id not in eos:
        eos.append(token_id)
    gen["eos_token_id"] = eos
    os.remove(path)  # may be a hardlink into another tree
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gen, f, indent=2)


def check_llava_next(cfg: dict, hidden: int) -> None:
    """What rank 0's LLaVA-NeXT front end needs from config.json."""
    nl = cfg.get("image_newline")
    if not isinstance(nl, list) or len(nl) != hidden:
        raise SystemExit(
            f"config.json has no {hidden}-wide image_newline (optimum-intel writes it there); "
            "re-export the model with optimum-intel")
    strategy = cfg.get("vision_feature_select_strategy", "default")
    if strategy != "default":
        raise SystemExit(f"vision_feature_select_strategy {strategy!r} is not supported "
                         "(rank 0 expects the CLS row dropped)")
    if cfg.get("image_token_index") is None:
        raise SystemExit("config.json lacks image_token_index")


def run(model_dir: str, out_dir: str, num_stages: int, family: str,
        suffix: str, only_stage: int | None, all_logits: bool = False) -> None:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    check_family(cfg, family)
    text = flat_text_config(cfg, family)
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
    required += list(FAMILIES[family].get("requires", ()))
    if family in ("gemma3", "llava_next"):
        # Rank 0 reads the image processor settings from it (Gemma 3's 896x896
        # SigLIP resize; LLaVA-NeXT's tiling pinpoints and image_newline packing).
        required += ["preprocessor_config.json"]
    for fn in required:
        if not os.path.exists(os.path.join(model_dir, fn)):
            raise SystemExit(f"{model_dir} lacks {fn}; not an OpenVINO {family} VLM export")
    if family == "llava_next":
        check_llava_next(cfg, hidden)

    if FAMILIES[family]["token_type_ids"]:
        lm_inputs = [p.get_friendly_name() for p in ov.Core().read_model(lm_xml).get_parameters()]
        if "token_type_ids" not in lm_inputs:
            raise SystemExit(f"--family {family}: openvino_language_model.xml takes no "
                             f"token_type_ids ({lm_inputs}); images would be attended "
                             "causally. Re-export it with a newer optimum-intel")

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
    mrope_rows = FAMILIES[family].get("mrope_rows")
    n_deepstack = deepstack_layers(cfg, family)
    if n_deepstack and ranges[0][1] + 1 < n_deepstack:
        raise SystemExit(
            f"stage 0 would hold layers 0..{ranges[0][1]}, but Qwen3-VL adds DeepStack rows "
            f"after layers 0..{n_deepstack - 1} and only rank 0 has them; use fewer stages")
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
        if last:
            meta["logits"] = "all" if all_logits else "last"
        if mrope_rows:
            meta["position_rows"] = mrope_rows
        stages_meta.append(meta)
        if only_stage is not None and i != only_stage:
            continue
        t0 = time.time()
        stage, state_vars, _ = g.extract_stage(lm_xml, a, b, first, last, suffix,
                                               not all_logits)
        if mrope_rows:
            check_mrope_positions(stage, mrope_rows, i)
        if family == "qwen3_vl":
            stage = fix_deepstack_inputs(stage, first, i)
        sdir = os.path.join(out_dir, f"stage_{i}")
        os.makedirs(sdir, exist_ok=True)
        ov.save_model(stage, os.path.join(sdir, "openvino_model.xml"), compress_to_fp16=False)
        meta["inputs"] = [p.get_friendly_name() for p in stage.get_parameters()]
        if FAMILIES[family]["token_type_ids"] and "token_type_ids" not in meta["inputs"]:
            raise SystemExit(f"stage_{i} lost token_type_ids ({meta['inputs']}); every stage "
                             "must take them for bidirectional image attention")
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
    if family == "phi3_v":
        # The chat template closes every turn with <|end|>; the export's
        # generation_config stops only on <|endoftext|>, one token later.
        # InternVL2's Phi-3 tower lists both.
        end = special_token_id(model_dir, "<|end|>")
        for d in (out_dir, tok):
            gen = os.path.join(d, "generation_config.json")
            if os.path.exists(gen):
                add_stop_token(gen, end)

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
            "image_token_id": image_token_id(model_dir, cfg, family),
        },
    }
    if mrope_rows:
        pipeline["vision"]["mrope_rows"] = mrope_rows
    if n_deepstack:
        pipeline["vision"]["deepstack_layers"] = n_deepstack
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
    ap.add_argument("--all-logits", action="store_true",
                    help="keep logits for every fed position (an ov-dist-spec target)")
    args = ap.parse_args()
    # One stage serves a single box through ov-runtime: GenAI's own Pixtral
    # preprocessing differs from HF's (see docs/rfcs/vlm-support.md).
    if args.num_stages < 1:
        raise SystemExit("--num-stages must be >= 1")
    run(args.model, args.output_dir, args.num_stages, args.family,
        args.boundary_suffix, args.stage, args.all_logits)


if __name__ == "__main__":
    main()
