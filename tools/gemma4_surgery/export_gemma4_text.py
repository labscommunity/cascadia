#!/usr/bin/env python3
"""Gemma-4 multimodal VLM (OpenVINO int4 IR) -> cascadia text-only artifacts.

Productionizes a validated prototype pipeline that took a gemma-4-E2B VLM
int4 IR -> grafted text-only LM -> cascadia ov-genai -> coherent "Paris"
@ ~14 tok/s on a Lunar Lake iGPU.

No torch. No RAM wall: everything is OpenVINO graph surgery on memory-mapped
IR (``core.read_model`` / ``ov.save_model``); model weights are never
materialized into Python. Two modes selected by ``--num-stages``:

N == 1  (whole IR, for the ov-genai engine)
    Graft the token-embedding front-end into the language model and emit a
    single ``openvino_model.xml`` with the classic causal-LM signature
    ``input_ids, attention_mask, position_ids, beam_idx -> logits`` (stateful
    KV Assign sinks preserved). Alongside it: a BOS-regenerated tokenizer /
    detokenizer, a flattened text-gen ``config.json`` (VLM -> Gemma4ForCausalLM,
    ``text_config`` promoted), and copied ``chat_template.jinja`` /
    ``generation_config.json`` / ``tokenizer.json``.  This is the EXACT
    pipeline proven on-node.

N > 1   (sliced stages, for the ov-runtime distributed engine)
    Graft as above, then slice the grafted IR at decoder-layer boundaries into
    N per-stage stateful shards in the v3 on-disk layout the ov-runtime
    engine loads (``crates/cascadia-engine-openvino/src/runtime.rs``): a root
    ``pipeline_config.json`` + per-stage ``stage_{i}/openvino_model.xml`` +
    ``stage_{i}/stage_config.json`` — carrying every key ov-runtime's
    ``read_pipeline_config`` / ``read_stage_config`` deserialize (the full
    key sets otherwise differ from ``tools/export_shards.py``'s). The boundary + sink-ownership algorithm
    mirrors ``tools/qwen36_surgery/export_qwen36_moe.py``, so a gemma-4 too big
    for one node (31B) runs pipeline-parallel. Stage 0 keeps the grafted
    ``input_ids -> embeddings`` front-end; the last stage keeps the ``logits``
    output.

    Slicing is REFUSED for models with ``num_kv_shared_layers > 0`` (E2B=20,
    E4B=18): those layers reuse an earlier layer's KV cache, and this tool does
    not emit cross-stage KV passing (that lives in ``export_gemma4.py``).
    Override with ``--allow-kv-share`` only if you know the boundary keeps each
    sharing group inside one stage.

The graft (``graft_text_frontend``) and the N==1 aux steps
(``regenerate_tokenizer_bos``, ``flatten_config``) are faithful ports of that
validated prototype (graft, tokenizer-BOS regen, config flatten). The N>1
slice (``slice_stages`` / ``extract_stage`` / ``_validate``) is NEW code,
adapted from the qwen36 surgery; run ``--validate`` on-node to gate it.

Usage (on a node with the model dir + openvino / openvino_tokenizers):

    # whole text IR for ov-genai
    python export_gemma4_text.py \
        --model  C:\\cascadia\\models\\gemma-4-E2B-it-int4-ov \
        --output-dir C:\\cascadia\\models\\gemma-4-E2B-text-llm

    # 2-stage split for ov-runtime distributed (e.g. gemma-4-31B) + parity gate
    python export_gemma4_text.py \
        --model  C:\\cascadia\\models\\gemma-4-31b-it-int4-ov \
        --output-dir C:\\cascadia\\models\\gemma-4-31b-stages \
        --num-stages 2 --validate
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time

import openvino as ov
from openvino import opset13 as ops

# Files copied verbatim from the source VLM IR dir into the text output dir.
# (Same list the validated prototype copies, minus the sub-IRs we graft away.)
AUX_COPY = [
    "openvino_tokenizer.xml", "openvino_tokenizer.bin",
    "openvino_detokenizer.xml", "openvino_detokenizer.bin",
    "config.json", "generation_config.json", "chat_template.jinja",
    "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
]

# Default decoder-layer boundary op suffix (proven for HF-exported OV IR; used
# by qwen36_surgery). ``input_value(0)`` of this Power op is the residual-stream
# hidden state ENTERING the layer's input RMSNorm — the clean cut point.
DEFAULT_LAYERNORM_SUFFIX = "input_layernorm/aten::pow/Power"

# Tensor name stamped on the grafted inputs_embeds Output so the slice path can
# find and re-route its (shape-only) consumers off mid/last stages. Only added
# in the N>1 path — the N==1 output stays byte-faithful to step1_surgery.py.
GRAFTED_EMBEDS_NAME = "grafted_inputs_embeds"

# Fixed short prompt for the N>1 parity gate (leading BOS=2 for gemma-4).
VALIDATE_PROMPT_IDS = [2, 651, 6037, 603, 578, 3311]

# Stamped into the N>1 v3 layout (pipeline_config.json +
# stage_config.json) that ov-runtime loads, so the engine + on-node operator
# can identify which exporter produced these shards.
EXPORT_VERSION = "gemma4_text_surgery_v1"


def log(msg: str) -> None:
    print(msg, flush=True)


# ===========================================================================
# GRAFT  (VERBATIM-EQUIVALENT to step1_surgery.py / step1_surgery_26b.py)
# ===========================================================================

def find_param(model: ov.Model, key: str):
    """Parameter whose tensor names or friendly name contain ``key``."""
    for p in model.get_parameters():
        names = set(p.output(0).get_names()) | {p.get_friendly_name()}
        if any(key in n for n in names):
            return p
    return None


def rewire_param_to_source(param, new_source_output, label: str) -> int:
    """Redirect all consumers of ``param`` to ``new_source_output``."""
    n = 0
    for t in list(param.output(0).get_target_inputs()):
        t.replace_source_output(new_source_output)
        n += 1
    log(f"  [{label}] rewired {n} consumer(s) of param "
        f"{sorted(param.output(0).get_names())}")
    return n


def maybe_convert(src_out, want_type, label: str):
    """Insert a Convert if source element-type != target (prototype behavior)."""
    if src_out.get_element_type() != want_type:
        log(f"  [{label}] element-type mismatch: {src_out.get_element_type()} "
            f"-> {want_type}; inserting Convert")
        return ops.convert(src_out, want_type).output(0)
    return src_out


def graft_text_frontend(src_dir: str, tag_inputs_embeds: bool = False):
    """Assemble an ``input_ids``-based text-only LM from the gemma-4 VLM IR.

    Faithful port of ``step1_surgery.py`` (+ the ``step1_surgery_26b.py``
    degenerate-``per_layer`` auto-drop). Chains, in-graph and torch-free::

        text_embeddings(input_ids)           -> inputs_embeds
        text_embeddings_per_layer(input_ids) -> per_layer_inputs  (if consumed)

    onto ``openvino_language_model.xml``, preserving every stateful Assign sink
    and a single shared ``input_ids`` Parameter.

    ``tag_inputs_embeds`` (set only by the N>1 slice path) stamps the grafted
    inputs_embeds Output with ``GRAFTED_EMBEDS_NAME`` so the slicer can drop it
    from mid stages. It is left False for N==1 so that path's serialized IR
    stays byte-identical to the validated prototype output.

    Returns ``(grafted_model, info)`` where ``info`` records
    ``per_layer_dropped`` and the surviving input names.
    """
    core = ov.Core()
    log("=== read sub-IRs ===")

    def _read_sub_ir(fname):
        path = os.path.join(src_dir, fname)
        if not os.path.exists(path):
            raise RuntimeError(
                f"required gemma-4 sub-IR missing: {path} (is --model a "
                f"complete optimum-intel gemma-4 VLM OpenVINO IR dir?)"
            )
        return core.read_model(path)

    lm = _read_sub_ir("openvino_language_model.xml")
    emb = _read_sub_ir("openvino_text_embeddings_model.xml")
    pl = _read_sub_ir("openvino_text_embeddings_per_layer_model.xml")

    # --- locate LM params ---
    p_embeds = find_param(lm, "inputs_embeds")
    p_perlayer = find_param(lm, "per_layer")
    p_attn = find_param(lm, "attention_mask")
    p_pos = find_param(lm, "position_ids")
    p_beam = find_param(lm, "beam_idx")
    log("LM params:")
    for nm, p in [("inputs_embeds", p_embeds), ("per_layer", p_perlayer),
                  ("attention_mask", p_attn), ("position_ids", p_pos),
                  ("beam_idx", p_beam)]:
        if p is None:
            log(f"  {nm}: <MISSING>")
        else:
            log(f"  {nm}: names={sorted(p.output(0).get_names())} "
                f"shape={p.get_partial_shape()} dtype={p.get_element_type()} "
                f"n_consumers={len(list(p.output(0).get_target_inputs()))}")

    if p_embeds is None:
        raise RuntimeError(
            "no `inputs_embeds` Parameter in openvino_language_model.xml — "
            "is this an optimum-intel VLM IR (text_embeddings + language_model "
            "sub-IRs)?")

    # --- embedding subgraph (input_ids -> inputs_embeds) ---
    emb_ids = find_param(emb, "input_ids") or emb.get_parameters()[0]
    emb_out = emb.get_results()[0].input_value(0)
    log(f"emb: input names={sorted(emb_ids.output(0).get_names())} "
        f"shape={emb_ids.get_partial_shape()} dtype={emb_ids.get_element_type()}")
    log(f"emb: output dtype={emb_out.get_element_type()} "
        f"shape={emb_out.get_partial_shape()}")

    # --- per-layer subgraph (input_ids -> per_layer_inputs) ---
    pl_ids = find_param(pl, "input_ids") or pl.get_parameters()[0]
    pl_out = pl.get_results()[0].input_value(0)
    log(f"pl:  input names={sorted(pl_ids.output(0).get_names())} "
        f"shape={pl_ids.get_partial_shape()} dtype={pl_ids.get_element_type()}")
    log(f"pl:  output dtype={pl_out.get_element_type()} "
        f"shape={pl_out.get_partial_shape()}")

    # --- single shared input_ids Parameter ---
    shared_ids = ops.parameter(emb_ids.get_partial_shape(),
                               emb_ids.get_element_type(), name="input_ids")
    shared_ids.output(0).set_names({"input_ids"})
    rewire_param_to_source(emb_ids, shared_ids.output(0), "emb.input_ids")
    rewire_param_to_source(pl_ids, shared_ids.output(0), "pl.input_ids")

    # --- neutralize VLM-only token_type_ids (present in 31B / transformers>=5
    #     exports; absent on E2B). Text-only => every token is text (type 0).
    #     Feed a dynamic all-zeros tensor (in the Parameter's own element
    #     type) shaped like input_ids so the Parameter can be DROPPED
    #     (ov-genai LLMPipeline never feeds it). ---
    p_tok_type = find_param(lm, "token_type_ids")
    if p_tok_type is not None:
        import numpy as _np
        n_tt = len(list(p_tok_type.output(0).get_target_inputs()))
        # Match the zero-constant dtype to token_type_ids' element type: an
        # i32-typed token_type_ids would throw at graft time against a hardcoded
        # i64 constant (Multiply requires matching operand element types).
        _tt_np_dtype = p_tok_type.get_element_type().to_dtype()
        zeros_tt = ops.multiply(
            maybe_convert(shared_ids.output(0), p_tok_type.get_element_type(),
                          "ttids->tok_type"),
            ops.constant(_np.array(0, dtype=_tt_np_dtype)))
        rewire_param_to_source(p_tok_type, zeros_tt.output(0),
                               "lm.token_type_ids(zeros)")
        log(f"  [lm.token_type_ids] neutralized -> zeros ({n_tt} consumer(s)),"
            f" param dropped")

    # --- graft emb_out -> inputs_embeds consumers ---
    emb_src = maybe_convert(emb_out, p_embeds.get_element_type(),
                            "emb->inputs_embeds")
    if tag_inputs_embeds:
        names = set(emb_src.get_names())
        names.add(GRAFTED_EMBEDS_NAME)
        emb_src.set_names(names)
    rewire_param_to_source(p_embeds, emb_src, "lm.inputs_embeds")

    # --- graft pl_out -> per_layer consumers, unless degenerate ---
    # E2B/E4B: real per_layer_inputs, e.g. [?,?,35,256], >0 consumers -> graft.
    # 26B/31B: degenerate [?,?,?,0] with 0 consumers -> DROP (auto-detected,
    # never hardcoded — exactly step1_surgery_26b.py).
    per_layer_dropped = False
    if p_perlayer is None:
        log("  [lm.per_layer] no per_layer Parameter present -> nothing to graft")
        per_layer_dropped = True
    elif len(list(p_perlayer.output(0).get_target_inputs())) == 0:
        log("  [lm.per_layer] param has 0 consumers -> DEGENERATE, per_layer "
            "subgraph dropped")
        per_layer_dropped = True
    else:
        pl_src = maybe_convert(pl_out, p_perlayer.get_element_type(),
                               "pl->per_layer")
        rewire_param_to_source(p_perlayer, pl_src, "lm.per_layer")

    # --- assemble: keep LM results + all Assign sinks; new param list ---
    results = list(lm.get_results())
    sinks = list(lm.get_sinks())
    params = [shared_ids]
    for p in (p_attn, p_pos, p_beam):
        if p is not None:
            params.append(p)

    grafted = ov.Model(results, sinks, params, "gemma4_text_llm")

    inputs = [sorted(p.output(0).get_names()) for p in grafted.get_parameters()]
    outputs = [sorted(o.get_names()) for o in grafted.outputs]
    log(f"\ngrafted inputs: {inputs}")
    log(f"grafted outputs: {outputs}")
    log(f"grafted sinks (stateful KV): {len(grafted.get_sinks())}")

    info = {
        "per_layer_dropped": per_layer_dropped,
        "has_beam_idx": p_beam is not None,
        "inputs": [n[0] if n else "" for n in inputs],
        "num_sinks": len(grafted.get_sinks()),
    }
    return grafted, info


# ===========================================================================
# N == 1  aux  (VERBATIM-EQUIVALENT to regen_tok.py / flatten_config.py)
# ===========================================================================

def copy_aux(src_dir: str, dst_dir: str) -> None:
    """Copy tokenizer / detokenizer / config / chat template into ``dst_dir``."""
    for fn in AUX_COPY:
        s = os.path.join(src_dir, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst_dir, fn))
            log(f"copied {fn}")
        else:
            log(f"skip (absent) {fn}")


# HF tokenizer files ov-runtime loads from ``<out>/tokenizer/`` (see
# runtime.rs:12 doc + loader: ``tokenizer/tokenizer.json`` for the Rust
# ``tokenizers`` crate, plus ``config.json`` / ``generation_config.json`` for
# rotary + eos lookup). Mirrors ``tools/export_shards.py::copy_tokenizer``.
TOKENIZER_SUBDIR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "config.json",
    "generation_config.json",
    "added_tokens.json",
    # ov-runtime's chat formatter (cascadia-api::load_chat_template_config)
    # checks the ``chat_template`` field in ``tokenizer/tokenizer_config.json``
    # FIRST (gemma-4 does NOT embed one), then falls back to
    # ``tokenizer/chat_template.jinja`` — and considers the jinja only when
    # ``tokenizer/tokenizer_config.json`` is present and parsable, which is why
    # that file must also ship in this list. Without a template rank 0 uses legacy
    # "role: content" formatting and the instruction-tuned model degenerates
    # (observed: "la la la ..." instead of a coherent answer). Ship the jinja
    # into the subdir so the turn markers are applied.
    "chat_template.jinja",
)


def copy_tokenizer_subdir(model_dir: str) -> None:
    """Populate ``<model_dir>/tokenizer/`` from the finalized root files.

    ov-runtime (the N>1 engine) expects the HF tokenizer + config under a
    ``tokenizer/`` subdir, NOT at the model root (ov-runtime's ``load()``
    joins ``pipeline_dir/tokenizer`` first). Run AFTER ``flatten_config`` and
    ``regenerate_tokenizer_bos`` so the flat text-gen ``config.json`` and the
    BOS/transformers-5-coerced ``tokenizer_config.json`` are the versions that
    ship. Root files are left in place (N==1 ov-genai + the regen step read
    them there), so this is additive. Mirrors
    ``tools/export_shards.py::copy_tokenizer``.
    """
    tok_dir = os.path.join(model_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)
    copied, missing = [], []
    for fn in TOKENIZER_SUBDIR_FILES:
        src = os.path.join(model_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(tok_dir, fn))
            copied.append(fn)
        else:
            missing.append(fn)
    log(f"copied {len(copied)} tokenizer file(s) into {tok_dir}"
        + (f" (absent: {', '.join(missing)})" if missing else ""))
    # ov-runtime cannot load a stage tree without the HF tokenizer — fail at
    # export time instead of at worker startup, far from the cause.
    if "tokenizer.json" in missing:
        raise SystemExit(
            f"tokenizer.json is missing from {model_dir} — ov-runtime cannot "
            f"load the stage tree without tokenizer/tokenizer.json. The "
            f"source IR should ship it; re-download or restore it.")
    if "chat_template.jinja" in missing:
        log("  WARNING: chat_template.jinja absent — rank 0 falls back to "
            "legacy 'role: content' prompt formatting, and instruction-tuned "
            "gemma-4 DEGENERATES without its turn markers (observed: "
            "'la la la ...'). Restore it in the source IR before serving.")


def regenerate_tokenizer_bos(model_dir: str) -> None:
    """Regenerate the OV tokenizer/detokenizer with a leading BOS.

    Faithful port of ``regen_tok.py``. The stock VLM tokenizer omits BOS;
    gemma-4 needs a leading ``<bos>`` (id 2) or greedy decode drifts. Requires
    ``transformers`` + ``openvino_tokenizers`` (present on-node with optimum).
    Backs up the originals to ``*.orig`` before overwriting.
    """
    import numpy as np
    from transformers import AutoTokenizer
    from openvino_tokenizers import convert_tokenizer

    # transformers-5 VLM exports write tokenizer_config.json with
    # ``extra_special_tokens`` as a LIST; AutoTokenizer.from_pretrained then
    # raises (older transformers expected a dict). Text-only regen only needs
    # BOS, so coerce a list-valued field to {} in the output-dir copy (backing
    # the original up as .t5.orig, matching the tool's *.orig backup convention)
    # before loading. No-op when the field is already a dict or absent (E2B /
    # older exports), so the proven BOS regen (Hello -> [2, 9259]) is unchanged.
    tok_cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        with open(tok_cfg_path, encoding="utf-8") as f:
            tok_cfg = json.load(f)
        if isinstance(tok_cfg.get("extra_special_tokens"), list):
            if not os.path.exists(tok_cfg_path + ".t5.orig"):
                shutil.copy2(tok_cfg_path, tok_cfg_path + ".t5.orig")
            tok_cfg["extra_special_tokens"] = {}
            with open(tok_cfg_path, "w", encoding="utf-8") as f:
                json.dump(tok_cfg, f, indent=2)
            log("  coerced transformers-5 extra_special_tokens list -> {} in "
                "tokenizer_config.json (backed up .t5.orig)")

    hf = AutoTokenizer.from_pretrained(model_dir, add_bos_token=True)
    log(f"loaded hf tokenizer: {type(hf).__name__} bos={hf.bos_token} "
        f"{hf.bos_token_id}")
    ids = hf("Hello")["input_ids"]
    log(f"hf encode 'Hello' -> {ids} (expect leading {hf.bos_token_id})")

    ov_tok, ov_detok = convert_tokenizer(hf, with_detokenizer=True)
    for fn in ("openvino_tokenizer.xml", "openvino_tokenizer.bin",
               "openvino_detokenizer.xml", "openvino_detokenizer.bin"):
        p = os.path.join(model_dir, fn)
        if os.path.exists(p) and not os.path.exists(p + ".orig"):
            shutil.copy2(p, p + ".orig")
    ov.save_model(ov_tok, os.path.join(model_dir, "openvino_tokenizer.xml"))
    ov.save_model(ov_detok, os.path.join(model_dir, "openvino_detokenizer.xml"))
    log("saved regenerated openvino_tokenizer/detokenizer")

    # verify via openvino_genai Tokenizer (proven check)
    try:
        import openvino_genai as ovg
        t = ovg.Tokenizer(model_dir)
        enc = t.encode("Hello")
        log(f"ovg encode 'Hello' -> {np.array(enc.input_ids.data).tolist()} "
            f"(want leading 2)")
    except Exception as e:  # noqa: BLE001 - verification only, non-fatal
        log(f"  (openvino_genai verify skipped: {e})")


def flatten_config(config_path: str) -> None:
    """VLM ``config.json`` -> flat text-gen ``Gemma4ForCausalLM`` config.

    Faithful port of ``flatten_config.py``: promote ``text_config`` to the top
    level, set text architecture/model_type, carry BOS/EOS/PAD ids. Backs up
    the original to ``*.vlm.orig``.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if "text_config" not in cfg:
        log("  config already flat (no text_config) — leaving as-is")
        return
    tc = cfg["text_config"]
    flat = dict(tc)  # promote text_config
    flat["architectures"] = ["Gemma4ForCausalLM"]
    flat["model_type"] = tc.get("model_type", "gemma4_text")
    flat["bos_token_id"] = 2
    flat["eos_token_id"] = cfg.get("eos_token_id", [1, 106])
    flat["pad_token_id"] = tc.get("pad_token_id", 0)
    flat["torch_dtype"] = "bfloat16"
    flat["transformers_version"] = cfg.get("transformers_version")
    if not os.path.exists(config_path + ".vlm.orig"):
        shutil.copy2(config_path, config_path + ".vlm.orig")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(flat, f, indent=2)
    log(f"wrote flat config. keys: {sorted(flat.keys())}")
    log(f"  num_hidden_layers={flat.get('num_hidden_layers')} "
        f"num_key_value_heads={flat.get('num_key_value_heads')} "
        f"head_dim={flat.get('head_dim')} "
        f"num_attention_heads={flat.get('num_attention_heads')} "
        f"num_kv_shared_layers={flat.get('num_kv_shared_layers')} "
        f"sliding_window={flat.get('sliding_window')}")


def save_whole(model: ov.Model, src_dir: str, out_dir: str,
               skip_tokenizer_regen: bool = False) -> None:
    """N==1: save the grafted IR + BOS tokenizer + flat config + aux.

    This is the proven ov-genai artifact set: ``openvino_model.xml`` plus a
    text-gen tokenizer/config the ov-genai pipeline loads directly.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_xml = os.path.join(out_dir, "openvino_model.xml")
    log(f"saving whole text IR -> {out_xml}")
    ov.save_model(model, out_xml, compress_to_fp16=False)

    copy_aux(src_dir, out_dir)
    flatten_config(os.path.join(out_dir, "config.json"))
    if skip_tokenizer_regen:
        log("skipping BOS tokenizer regen (--skip-tokenizer-regen)")
    else:
        try:
            regenerate_tokenizer_bos(out_dir)
        except ImportError as e:
            # Deliberate off-node tolerance: transformers/openvino_tokenizers
            # may be absent there; the copied VLM tokenizer may omit BOS —
            # rerun regen on-node. Any OTHER failure is a real bug and must
            # fail the export, not ship a BOS-less tokenizer with exit 0.
            log(f"  WARNING: tokenizer regen skipped ({e}); the copied VLM "
                f"tokenizer may omit BOS — rerun regen on-node")
    log("WHOLE-IR (N=1) DONE")


# ===========================================================================
# N > 1  slice  (NEW — adapted from qwen36_surgery/export_qwen36_moe.py)
# ===========================================================================

# variable_id patterns for per-layer KV state (optimum VLM IR + gemma4
# present.N/past_key_values.N naming). Permissive on purpose; a sample of the
# discovered variable_ids (and of any fallback attributions) is logged so the
# on-node operator can spot-check.
_LAYER_RE = re.compile(
    r"(?:past_key_values|present|past|layers?|blocks?|decoder)[._](\d+)")
# Cache kind (key/value/conv/ssm) taken as the delimited token immediately
# after the layer index — avoids matching the "key"/"value" inside the literal
# "key_values" substring (which is present in EVERY KV variable_id).
_KIND_RE = re.compile(r"[._]\d+[._](value|key|conv|ssm)")


def sink_layer_index(variable_id: str):
    """Layer number embedded in a KV Assign/ReadValue variable_id, or None.

    NOTE: optimum may sequence this number per ATTENTION TYPE, not per global
    layer (see the GLOBAL-LAYER attribution block below) — callers use it only
    as the fallback when no ``layers.{idx}`` op scope is reachable, where it is
    empirically correct.
    """
    m = _LAYER_RE.search(variable_id)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", variable_id)
    return int(m2.group(1)) if m2 else None


def sink_kind(variable_id: str) -> str:
    """Cache 'kind' (key/value/...) for same-kind orphan-read substitution.

    Uses the delimited token AFTER the layer index. Fallbacks test the
    unambiguous kinds (value/conv/ssm) before ``key`` so a value cache is never
    misread as ``key`` via the ``key_values`` substring.
    """
    m = _KIND_RE.search(variable_id)
    if m:
        return m.group(1)
    for k in ("value", "conv", "ssm"):
        if re.search(rf"[._]{k}(?:[._]|cache|$)", variable_id):
            return k
    if re.search(r"[._]key(?:[._]|cache|$)", variable_id):
        return "key"
    return "state"


def stage_ranges(total: int, num_layers: int):
    """Even decoder-layer ranges; remainder folded into the last stage."""
    per = num_layers // total
    ranges, start = [], 0
    for i in range(total):
        end = start + per - 1 if i < total - 1 else num_layers - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def layer_entry_output(model: ov.Model, idx: int, suffix: str):
    """Residual-stream Output ENTERING decoder layer ``idx``.

    ``input_value(0)`` of ``layers.{idx}.{suffix}`` (the input-RMSNorm Power).
    This defines BOTH boundaries: stage input = entry(a); stage output =
    entry(b+1) (== hidden state leaving layer b). Deriving the output cut from
    the next layer's entry (rather than a residual-add op name) makes the slice
    robust to gemma4's extra post-norms.
    """
    target = f"layers.{idx}.{suffix}"
    for op in model.get_ops():
        if op.get_friendly_name().endswith(target):
            return op.input_value(0)
    return None


def find_output_by_name(model: ov.Model, name: str):
    """First op Output whose tensor names contain ``name`` (post re-read)."""
    for op in model.get_ops():
        for out in op.outputs():
            if name in out.get_names():
                return out
    return None


# --- GLOBAL-LAYER attribution for KV state sinks ---------------------------
# The stage boundary is cut by GLOBALLY-indexed op scopes (``layers.{idx}`` —
# see layer_entry_output). optimum, however, may number a stateful model's KV
# variable_ids by ATTENTION-TYPE SEQUENCE, not global layer: gemma-4 interleaves
# sliding/local and global attention with DIFFERENT KV geometry (the
# num_global_key_value_heads / k_eq_v split, PR #72 — sliding is 8x256, global
# is 2x512). Attributing an Assign to a stage by the number inside its
# variable_id therefore uses the WRONG index space, splitting a layer's
# key/value ReadValue<->Assign pair across the boundary. The genuine KV read is
# then orphaned, and the by-kind orphan rewire wires a wrong-geometry cache onto
# it, so OpenVINO rejects the past(+)current Concat at serialize (the observed
# 26B --num-stages 2 crash). Attributing each Assign to the ``layers.{idx}``
# scope of the ops that FEED it keeps the pair together in the SAME index space
# the boundary uses — the same globally-indexed contract qwen36 encodes with its
# explicit per-layer variable_id map (layer_state_vids). For the few sinks the
# bounded scope-BFS cannot reach, the variable_id number is the fallback, and
# is empirically correct there (see ``_nearest_scope_layer``).
_LAYER_SCOPE_RE = re.compile(r"layers\.(\d+)")


def _op_scope_layer(op):
    """Global decoder-layer index from an op's ``layers.{idx}`` friendly-name
    scope, or None."""
    m = _LAYER_SCOPE_RE.search(op.get_friendly_name())
    return int(m.group(1)) if m else None


def _nearest_scope_layer(root, max_ops: int = 64):
    """Shallowest ``layers.{idx}`` scope reachable backward from ``root``
    (inclusive), or None. For a KV Assign the value stored is the layer's
    concatenated cache, produced by ``layers.{idx}.self_attn/aten::cat`` — so
    the nearest scope IS the layer that owns the Assign, in the same index space
    the boundary cut uses. Bounded so it never walks the whole graph.

    ``max_ops=64`` is validated (31B dense + 26B-A4B-it distributed coherent).
    Do NOT raise it blindly: a review round widened it to 256 to "harden"
    attribution, which mis-attributed some Assigns into an ADJACENT layer's
    ``layers.{idx}`` scope and was reverted. When the BFS legitimately returns
    None, the caller falls back to the variable_id index (``sink_layer_index``),
    which is correct for the sinks 64 hops can't reach. (Separately: if a
    surgered stage tree generates degenerate ``thought``-loop output, suspect a
    BASE source IR, not attribution — the instruction-tuned ``-it`` variant is
    required; base ``gemma-4-*-int4-ov`` loops on a chat prompt on ANY tool.)"""
    from collections import deque

    seen, q, visited = set(), deque([root]), 0
    while q and visited < max_ops:
        node = q.popleft()
        iid = node.get_instance_id()
        if iid in seen:
            continue
        seen.add(iid)
        visited += 1
        layer = _op_scope_layer(node)
        if layer is not None:
            return layer
        for iv in node.input_values():
            q.append(iv.get_node())
    return None


def _readvalue_scope_layer(rv_op):
    """Best-effort global layer index for a KV ReadValue, from a consumer's
    ``layers.{idx}`` scope (falls back to the variable_id parse). Diagnostics
    only — used to name the offending layer in error messages."""
    for out in rv_op.outputs():
        for tgt in out.get_target_inputs():
            layer = _op_scope_layer(tgt.get_node())
            if layer is not None:
                return layer
    return sink_layer_index(rv_op.get_variable_id())


def _feeds_attention_concat(rv_op) -> bool:
    """True if a ReadValue directly feeds a self-attention KV Concat
    (past(+)current) — i.e. a GENUINE KV read, not a shape-only bookkeeping
    read. Such a read must never be substituted by another layer's cache
    (silent garbage): its Assign should have been owned by this stage."""
    for out in rv_op.outputs():
        for tgt in out.get_target_inputs():
            c = tgt.get_node()
            if (c.get_type_name() == "Concat"
                    and "self_attn" in c.get_friendly_name()):
                return True
    return False


def _shapes_compatible(ps_a, ps_b) -> bool:
    """KV-cache PartialShape compatibility for orphan substitution: mergeable rank
    and, on every axis where BOTH dims are static, equal length. The dynamic
    sequence axis is a wildcard; the static head-count and head-dim axes must
    match, so a global-layer cache ([?,2,?,512]) can never be substituted onto a
    sliding-layer read ([?,8,?,256])."""
    try:
        # OpenVINO's own check: dynamic dims are wildcards, static dims must
        # be equal, ranks must be mergeable — exactly the semantics we want.
        return ps_a.compatible(ps_b)
    except Exception:  # noqa: BLE001 - older OV: structural fallback
        return str(ps_a) == str(ps_b)


def extract_stage(grafted_xml: str, a: int, b: int, first: bool, last: bool,
                  suffix: str, last_logits_only: bool):
    """Cut the grafted IR into the stateful shard for layers ``a..b``.

    Adapted from qwen36_surgery.extract_stage. Fresh ``read_model`` per stage
    (mmap; the surgery mutates the graph) so no RAM wall.
    """
    core = ov.Core()
    model = core.read_model(grafted_xml)

    params_new = []
    if not first:
        entry = layer_entry_output(model, a, suffix)
        if entry is None:
            raise RuntimeError(
                f"stage input boundary not found: layers.{a}.{suffix} "
                f"(override with --boundary-suffix; check op names on-node)")
        # ov-runtime feeds the inter-stage activation as f16 under the tensor
        # name ``hidden_states`` (runtime.rs wire-format doc,
        # ``input_named("hidden_states")``, the ShimDType::F16 ``set_input``
        # feed) — the exact contract ``tools/export_shards.py`` emits (its
        # whole model is traced at torch.float16, and it names the non-first
        # stage's Parameter ``hidden_states``). Match BOTH: name
        # the Parameter ``hidden_states`` and give it element type f16 so the
        # F16 feed is accepted. The grafted residual-stream boundary may be a
        # wider dtype (the graft is saved compress_to_fp16=False), so insert a
        # Convert from the f16 input to the boundary's native element type
        # before rewiring consumers (a no-op when the boundary is already f16).
        param = ops.parameter(entry.get_partial_shape(),
                              ov.Type.f16, name="hidden_states")
        param.output(0).set_names({"hidden_states"})
        hidden_src = maybe_convert(param.output(0), entry.get_element_type(),
                                   "hidden_states->stage-boundary")
        for tgt in list(entry.get_target_inputs()):
            tgt.replace_source_output(hidden_src)
        params_new.append(param)

        # Drop the token-embedding matrix from mid/last stages. The grafted
        # inputs_embeds Output (tagged GRAFTED_EMBEDS_NAME) still feeds two
        # things: (a) layer 0's scaled residual path — pruned in a non-first
        # stage — and (b) mask/position ShapeOf chains that read it for SHAPE
        # only. hidden_states has the identical [?,?,hidden] shape, so redirect
        # ALL its consumers onto hidden_states; the embedding subgraph (and its
        # input_ids Parameter, unless per_layer keeps it) then drops out.
        emb_src_out = find_output_by_name(model, GRAFTED_EMBEDS_NAME)
        if emb_src_out is None:
            # Fallback (untagged graft): layer-0 entry is the closest proxy.
            emb_src_out = layer_entry_output(model, 0, suffix)
        if emb_src_out is not None and emb_src_out is not entry:
            n = 0
            for tgt in list(emb_src_out.get_target_inputs()):
                tgt.replace_source_output(hidden_src)
                n += 1
            if n:
                log(f"  rewired {n} grafted-inputs_embeds shape-consumer(s) "
                    f"onto hidden_states")

        # Sever the input_ids-rooted attention-mask padding branch from mid/last
        # stages. gemma-4's mask reconstruction feeds a SECOND consumer off the
        # input_ids Parameter (Multiply_30620 -> Pad -> Equal(pad_id) -> ... ->
        # aten::masked_fill/Select_1 -> SDPA mask) that the inputs_embeds rewire
        # above does NOT touch — leaving input_ids alive in a stage ov-runtime
        # never feeds it (mid stages get hidden_states/attention_mask/position_
        # ids/beam_idx only). That is the "[CPU] Select ... dim index 2 mismatch"
        # compile failure: the branch's query seq-dim comes from input_ids while
        # the causal side comes from arange(hidden_states/attention_mask), so the
        # masked_fill Select cannot broadcast.
        #
        # The branch is VALUE-independent: its root op multiplies input_ids by a
        # 0 constant (Multiply_30620, Constant=[0]), so it depends on input_ids
        # ONLY for its [batch, query_seq] shape. hidden_states carries that exact
        # [batch, query_seq] in its first two axes and is the same source the
        # residual stream + SDPA query use. Rebuild the branch root as
        # zeros[batch, query_seq] (i64) from ShapeOf(hidden_states) and redirect
        # every input_ids consumer onto it: values are byte-identical (still
        # zeros), the query seq-dim now flows from hidden_states so the Select
        # broadcasts, and input_ids drops to zero consumers -> it is not reached
        # by the results/sinks walk below and falls out of the stage Parameter
        # list entirely (the total input_ids redirect llama/qwen36 get for free).
        import numpy as _np
        ids_param = None
        for p in model.get_parameters():
            names = set(p.output(0).get_names()) | {p.get_friendly_name()}
            if any("input_ids" in nm for nm in names):
                ids_param = p
                break
        if ids_param is not None:
            consumers = list(ids_param.output(0).get_target_inputs())
            if consumers:
                hs_shape = ops.shape_of(param.output(0), output_type="i64")
                bt_seq = ops.gather(
                    hs_shape,
                    ops.constant(_np.array([0, 1], dtype=_np.int64)),
                    ops.constant(_np.array(0, dtype=_np.int64)))
                zeros_ids = ops.broadcast(
                    ops.constant(_np.array(0, dtype=_np.int64)),
                    bt_seq.output(0))
                zeros_out = maybe_convert(
                    zeros_ids.output(0), ids_param.get_element_type(),
                    "hidden_states->input_ids_pad")
                for tgt in consumers:
                    tgt.replace_source_output(zeros_out)
                log(f"  severed input_ids padding branch: rewired "
                    f"{len(consumers)} consumer(s) onto zeros[batch,seq] from "
                    f"hidden_states; input_ids Parameter drops out of stage")

    if last:
        # Natural logits output; optionally sliced to the last position so
        # batched prefill stops materializing [1, T, vocab].
        results = []
        for r in list(model.get_results()):
            p = r.input_value(0)
            if last_logits_only:
                import numpy as np
                sl = ops.slice(
                    p,
                    ops.constant(np.array([-1], dtype=np.int64)),
                    ops.constant(np.array([np.iinfo(np.int64).max], dtype=np.int64)),
                    ops.constant(np.array([1], dtype=np.int64)),
                    ops.constant(np.array([1], dtype=np.int64)),
                )
                results.append(ops.result(sl.output(0)))
            else:
                results.append(ops.result(p))
    else:
        out = layer_entry_output(model, b + 1, suffix)
        if out is None:
            raise RuntimeError(
                f"stage output boundary not found: layers.{b + 1}.{suffix}")
        results = [ops.result(out)]
        results[0].output(0).set_names({"stage_hidden_out"})

    # per-layer KV Assign sinks owned by this range. Attribute each Assign to
    # the GLOBAL decoder layer of the ops that FEED it (``layers.{idx}`` scope)
    # — the same index space the boundary cut uses — instead of the number
    # inside the variable_id, which optimum may sequence per attention-type for
    # gemma-4's heterogeneous sliding/global KV. This keeps each layer's key AND
    # value ReadValue<->Assign pair on the same side of the boundary, so a
    # num_kv_shared_layers==0 model is expected to have NO genuine-KV orphans
    # (an empirical result — the ``_feeds_attention_concat`` refusal below
    # backstops it; only shape-only bookkeeping reads get rewired).
    sinks, all_vids, vid_parsed = [], [], []
    for op in model.get_ops():
        if op.get_type_name() != "Assign":
            continue
        vid = op.get_variable_id()
        all_vids.append(vid)
        idx = _nearest_scope_layer(op)
        if idx is None:
            idx = sink_layer_index(vid)  # fallback: un-scoped variable_id
            if idx is not None:
                vid_parsed.append(vid)
            else:
                # Doubly unattributable: this Assign lands in NO stage. A
                # genuine-KV sink dropped here still trips the orphan refusal
                # in its owning stage, but say so instead of dropping silently.
                log(f"  WARNING: Assign {vid!r} unattributable (no layers.N "
                    f"scope, no digit in variable_id) — excluded from every "
                    f"stage")
        if idx is not None and a <= idx <= b:
            sinks.append(op)
    if vid_parsed:
        log(f"  WARNING: {len(vid_parsed)} Assign(s) had no layers.N op scope; "
            f"attributed by variable_id parse (verify on-node): "
            f"{vid_parsed[:4]}")

    # original Parameters still reachable from results + sinks (input_ids,
    # attention_mask, position_ids, beam_idx — but never hidden_states)
    seen, reach = set(), set()
    stack = list(results) + sinks
    while stack:
        node = stack.pop()
        if node.get_instance_id() in seen:
            continue
        seen.add(node.get_instance_id())
        for iv in node.input_values():
            src = iv.get_node()
            if (src.get_type_name() == "Parameter"
                    and src.get_friendly_name() != "hidden_states"):
                reach.add(src)
            stack.append(src)
    orig = [p for p in model.get_parameters() if p in reach]

    stage = ov.Model(results, sinks, params_new + orig,
                     f"gemma4_stage_{a}_{b}")

    # Orphan-state rewire. With sink ownership attributed by global-layer scope
    # (above), a num_kv_shared_layers==0 model has NO genuine-KV orphans — every
    # layer's key AND value ReadValue is owned by the stage that owns the layer.
    # The only orphans left are shape-only bookkeeping reads (mask / position
    # ShapeOf chains that reach an out-of-stage layer's cache); those are safe to
    # redirect onto a same-kind cache this stage owns — but ONLY a
    # SHAPE-COMPATIBLE one. gemma-4's heterogeneous attention gives 'key'/'value'
    # caches TWO geometries (sliding [?,8,?,256] vs global [?,2,?,512]), so a
    # kind-ONLY substitution can wire the wrong geometry onto a consumer and
    # OpenVINO rejects the past(+)current Concat at serialize (the 26B crash).
    # Match on (kind AND partial_shape); refuse with a clear error rather than
    # force a wrong-shape or genuine-KV substitution.
    owned = {s.get_variable_id() for s in stage.get_sinks()}
    by_kind, orphans = {}, []
    for op in stage.get_ops():
        if op.get_type_name() != "ReadValue":
            continue
        vid = op.get_variable_id()
        kind = sink_kind(vid)
        ps = op.output(0).get_partial_shape()
        if vid in owned:
            by_kind.setdefault(kind, []).append((op, ps))
        else:
            orphans.append((op, kind, ps))
    for op, kind, ps in orphans:
        # A genuine KV read (feeds the layer's self-attention KV Concat) must
        # never be substituted — that would silently corrupt attention. Its
        # Assign should have been owned by this stage; reaching here means sink
        # ownership could not attribute the layer for this IR's variable-id
        # layout, so refuse loudly instead of guessing.
        if _feeds_attention_concat(op):
            layer = _readvalue_scope_layer(op)
            raise SystemExit(
                f"gemma4 slice stage {a}..{b}: KV ReadValue "
                f"{op.get_variable_id()} (layer {layer}, shape {ps}) is a "
                f"genuine self-attention KV read but its Assign was not owned "
                f"by this stage — a stage boundary split this layer's KV "
                f"ReadValue<->Assign pair. Substituting another cache would "
                f"silently corrupt attention, so the stage is refused. Re-run "
                f"on-node and check the logged Assign variable_ids / layers.N "
                f"scoping for this IR.")
        candidates = by_kind.get(kind, [])
        sub = next((c for c, cps in candidates if _shapes_compatible(cps, ps)),
                   None)
        if sub is None:
            owned_shapes = sorted({str(cps) for _, cps in candidates})
            layer = _readvalue_scope_layer(op)
            raise SystemExit(
                f"gemma4 slice stage {a}..{b}: no shape-compatible owned "
                f"'{kind}' cache for orphan state {op.get_variable_id()} "
                f"(layer {layer}, shape {ps}); this stage owns '{kind}' caches "
                f"with shapes {owned_shapes or '[]'}. A stage boundary split a "
                f"KV group with incompatible geometry (gemma-4 heterogeneous "
                f"sliding/global attention: e.g. sliding [?,8,?,256] vs global "
                f"[?,2,?,512]) and this stage owns no cache of matching "
                f"geometry — the boundary cannot be placed here without "
                f"cross-stage KV passing (not emitted by this tool).")
        for tgt in list(op.output(0).get_target_inputs()):
            tgt.replace_source_output(sub.output(0))
    if orphans:
        log(f"  rewired {len(orphans)} orphan state read(s) onto "
            f"shape-compatible owned caches")

    return stage, sorted({s.get_variable_id() for s in sinks}), all_vids


def slice_stages(grafted: ov.Model, src_dir: str, out_dir: str,
                 num_stages: int, num_layers: int, hidden_size,
                 num_kv_shared_layers: int, suffix: str,
                 last_logits_only: bool, only_stage=None,
                 keep_grafted: bool = False,
                 skip_tokenizer_regen: bool = False,
                 allow_kv_share: bool = False,
                 validate: bool = False,
                 num_kv_heads=None, head_dim=None) -> None:
    """N>1: slice the grafted IR into per-stage stateful shards, emitting the
    v3 on-disk layout that the ov-runtime engine loads.

    Layout (mirrors ``tools/export_shards.py`` so ov-runtime's
    ``read_pipeline_config`` / ``read_stage_config`` find every field):

        <out>/pipeline_config.json          (model_id, num_stages, num_layers,
                                              hidden_size, export_version)
        <out>/stage_{i}/openvino_model.xml  + openvino_model.bin
        <out>/stage_{i}/stage_config.json   (layer_start, layer_end, has_embed,
                                              has_head, stateful, num_kv_heads,
                                              head_dim, export_version)

    Saves the grafted IR to a temp dir first, then re-reads it per stage
    (mmap, no RAM wall) — mirroring qwen36's ``extract_stage(xml_path, ...)``.
    ``num_kv_heads`` / ``head_dim`` are best-effort per-model defaults carried
    into each ``stage_config.json`` as the ``Option`` hints StageConfig reads.
    """
    # HARD guard: KV-sharing severed across a stage boundary silently rewires
    # an orphan ReadValue to the WRONG cache -> silent garbage. This tool emits
    # no cross-stage KV passing, so refuse unless explicitly overridden.
    if num_kv_shared_layers and num_kv_shared_layers > 0 and not allow_kv_share:
        raise SystemExit(
            f"gemma-4 with num_kv_shared_layers={num_kv_shared_layers} cannot "
            f"be sliced: cross-stage KV passing is not emitted, so a stage that "
            f"splits a KV-sharing group would silently produce garbage. Use "
            f"--num-stages 1, or a model with num_kv_shared_layers=0 (e.g. "
            f"26B/31B), or pass --allow-kv-share ONLY if your boundary keeps "
            f"each sharing group inside one stage.")

    os.makedirs(out_dir, exist_ok=True)
    # pipeline_config.json is the "tree is complete" marker (harnesses and
    # operators key on it), so a stale one from a previous export must never
    # survive a failed/mixed re-export. Remove it up front; it is re-written
    # LAST, after every stage + aux step + the optional parity gate succeeded.
    # (--stage i incremental exports keep the existing marker untouched.)
    marker_path = os.path.join(out_dir, "pipeline_config.json")
    if only_stage is None and os.path.exists(marker_path):
        os.remove(marker_path)
        log(f"removed stale pipeline_config.json (re-export into {out_dir})")
    grafted_dir = os.path.join(out_dir, "_grafted")
    os.makedirs(grafted_dir, exist_ok=True)
    grafted_xml = os.path.join(grafted_dir, "openvino_model.xml")
    log(f"saving grafted whole IR (temp) -> {grafted_xml}")
    ov.save_model(grafted, grafted_xml, compress_to_fp16=False)

    ranges = stage_ranges(num_stages, num_layers)
    if allow_kv_share and num_kv_shared_layers and num_kv_shared_layers > 0:
        log(f"  WARNING: --allow-kv-share with num_kv_shared_layers="
            f"{num_kv_shared_layers}; correctness depends on boundaries keeping "
            f"each sharing group inside one stage. VALIDATE on-node.")

    # Top-level metadata in the v3 layout ov-runtime loads. The first
    # five keys are what PipelineConfig reads; the rest are gemma-4-specific
    # diagnostics ov-runtime ignores (no deny_unknown_fields) but keep for the
    # on-node operator.
    model_id = os.path.basename(os.path.abspath(src_dir))
    pipeline_config = {
        "model_id": model_id,
        "num_stages": num_stages,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "export_version": EXPORT_VERSION,
        # gemma-4 diagnostics (ignored by ov-runtime's PipelineConfig):
        "arch": "gemma4_text",
        "num_kv_shared_layers": num_kv_shared_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "source": model_id,
        "last_logits_only": last_logits_only,
        "kv_share_overridden": bool(allow_kv_share and num_kv_shared_layers),
        "stages": [],
    }

    try:
        for i, (a, b) in enumerate(ranges):
            if only_stage is not None and i != only_stage:
                continue
            first, last = i == 0, i == len(ranges) - 1
            t0 = time.time()
            stage, state_vars, all_vids = extract_stage(
                grafted_xml, a, b, first, last, suffix, last_logits_only)
            sdir = os.path.join(out_dir, f"stage_{i}")
            os.makedirs(sdir, exist_ok=True)
            ov.save_model(stage, os.path.join(sdir, "openvino_model.xml"),
                          compress_to_fp16=False)
            inputs = [p.get_friendly_name() for p in stage.get_parameters()]
            # Keys ov-runtime's StageConfig reads (layer_start/end, has_embed,
            # has_head, stateful, num_kv_heads, head_dim, export_version) plus
            # gemma-4 diagnostics it ignores (stage, inputs, state_vars).
            stage_cfg = {
                # layer_end is HALF-OPEN (v3 contract: cascadia-types
                # num_layers = layer_end - layer_start). stage_ranges returns an
                # INCLUSIVE b, so write b + 1. The internal slice math keeps
                # using the inclusive b (and b+1 for the residual cut)
                # untouched.
                "stage": i, "layer_start": a, "layer_end": b + 1,
                "has_embed": first, "has_head": last, "stateful": True,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "export_version": EXPORT_VERSION,
                "inputs": inputs,
                "state_vars": state_vars,
            }
            with open(os.path.join(sdir, "stage_config.json"), "w") as f:
                json.dump(stage_cfg, f, indent=2)
            pipeline_config["stages"].append(stage_cfg)
            log(f"stage_{i}: layers {a}..{b} saved in {time.time() - t0:.0f}s "
                f"inputs={inputs} states={len(state_vars)}")
            if i == 0:
                log(f"  (discovered {len(all_vids)} Assign variable_ids; "
                    f"sample: {all_vids[:4]})")

        # tokenizer/detokenizer/config alongside the stages (single-dir UX),
        # same as qwen36 — plus the proven BOS regen + config flatten.
        copy_aux(src_dir, out_dir)
        flatten_config(os.path.join(out_dir, "config.json"))
        if skip_tokenizer_regen:
            log("skipping BOS tokenizer regen (--skip-tokenizer-regen)")
        else:
            try:
                regenerate_tokenizer_bos(out_dir)
            except ImportError as e:
                # Off-node tolerance only (deps absent); other failures must
                # fail the export — see the N==1 twin in save_whole.
                log(f"  WARNING: tokenizer regen skipped ({e}); rerun on-node")

        # ov-runtime reads the HF tokenizer + config from a ``tokenizer/``
        # subdir (not the model root) — mirror export_shards.py. Runs after
        # flatten_config + regen so the shipped tokenizer/ has the flat
        # text-gen config.json and the coerced tokenizer_config.json. N==1
        # (ov-genai) reads root, untouched.
        copy_tokenizer_subdir(out_dir)

        # Parity gate (on-node): chained stages vs the grafted whole IR. Run
        # before the temp grafted IR is removed. Can't chain a partial export.
        if validate:
            if only_stage is not None:
                log("  (--validate skipped: --stage exports a single shard, "
                    "cannot chain — this also bypasses the cross-stage "
                    "KV-ownership refusal, which fires in the SIBLING stage "
                    "of a misattributed sink; verify against a full export)")
            else:
                _validate(grafted_xml, out_dir, ranges, last_logits_only)

        # Completion marker LAST: a tree carrying pipeline_config.json has by
        # construction finished every stage, aux step, and (if requested) the
        # parity gate. --stage i is an incremental single-stage export: don't
        # clobber a complete pipeline_config.json.
        if only_stage is None:
            with open(marker_path, "w") as f:
                json.dump(pipeline_config, f, indent=2)
        else:
            log(f"  (--stage {only_stage}: skipping pipeline_config.json "
                f"write)")
    except BaseException:
        log(f"  ERROR: export failed — PARTIAL tree left at {out_dir} "
            f"(no pipeline_config.json written; _grafted/ retained for "
            f"debugging). Delete the tree before use.")
        raise

    if not keep_grafted:
        # The last extracted stage still memory-maps ``_grafted/*.bin``; on
        # Windows a mapped file cannot be deleted, and ignore_errors turned
        # that into a silent ~17 GB leak per export (observed on-fleet: every
        # successful 31B slice left _grafted behind). Drop the reference,
        # collect, and VERIFY the dir is gone — never fail the export over
        # temp-dir hygiene, but never be silent about it either.
        stage = None  # release the last stage model's mapping
        gc.collect()
        shutil.rmtree(grafted_dir, ignore_errors=True)
        if os.path.isdir(grafted_dir):
            log(f"  WARNING: could not remove temp {grafted_dir} (files "
                f"likely still mapped) — delete it manually to reclaim disk")
    log("SLICED (N>1) DONE")


def _validate(grafted_xml: str, out_dir: str, ranges, last_logits_only: bool,
              device: str = "CPU") -> None:
    """On-node parity gate for the N>1 slice.

    Ports the intent of qwen36's ``_validate`` + ``probe_chain_vs_full_prompt``:
    run the grafted WHOLE IR and the CHAINED stages on the same fixed short
    prompt, feeding each stage's output as the next stage's ``hidden_states``
    input (the tiling contract). Assert last-position top-1 logit agreement,
    top-5 overlap, and bounded relative drift. Also asserts mid/last stages
    carry NO ``input_ids`` (a leaked ``input_ids`` means the embedding-drop
    rewire failed). Prints EXPORT_VALIDATE_OK/FAIL and
    exits non-zero on failure so it can gate a script.
    """
    import numpy as np

    core = ov.Core()
    ids = np.array([VALIDATE_PROMPT_IDS], dtype=np.int64)
    seq = ids.shape[1]

    def feeds(compiled, hidden=None):
        f = {}
        for inp in compiled.inputs:
            nm = inp.get_any_name()
            et = inp.get_element_type().to_dtype()
            ps = inp.get_partial_shape()
            rank = ps.rank.get_length() if ps.rank.is_static else 1
            if nm == "hidden_states":
                f[nm] = hidden.astype(et)
            elif "input_ids" in nm:
                f[nm] = ids.astype(et)
            elif "attention_mask" in nm:
                f[nm] = np.ones((1, seq), dtype=et)
            elif "position" in nm:
                pos = np.arange(seq, dtype=et)
                f[nm] = pos.reshape((1,) * max(rank - 1, 0) + (seq,))
            elif "beam_idx" in nm:
                f[nm] = np.zeros((1,), dtype=et)
            else:
                dims = [(d.get_length() if d.is_static else 1) for d in ps]
                f[nm] = np.zeros(dims or [1], dtype=et)
        return f

    def last_row(arr):
        a = np.asarray(arr)
        if a.ndim == 3:
            return a[0, -1]
        if a.ndim == 2:
            return a[-1]
        return a.reshape(-1)

    # reference: grafted whole IR
    whole = core.compile_model(core.read_model(grafted_xml), device)
    ref = whole.create_infer_request().infer(feeds(whole))
    ref_logits = last_row(ref[whole.outputs[0]]).astype(np.float32)

    # chained stages
    hidden, chain_logits = None, None
    input_leak = []
    for i in range(len(ranges)):
        sm = core.read_model(
            os.path.join(out_dir, f"stage_{i}", "openvino_model.xml"))
        sc = core.compile_model(sm, device)
        names = [inp.get_any_name() for inp in sc.inputs]
        if i != 0 and any("input_ids" in n for n in names):
            input_leak.append((i, names))
        out = sc.create_infer_request().infer(feeds(sc, hidden=hidden))
        hidden = np.asarray(out[sc.outputs[0]]).astype(np.float32)
        log(f"  stage_{i} ran, out shape {hidden.shape} inputs={names}")
    chain_logits = last_row(hidden).astype(np.float32)

    d = float(np.abs(chain_logits - ref_logits).max())
    n = float(np.abs(ref_logits).max()) + 1e-9
    top1 = int(chain_logits.argmax()) == int(ref_logits.argmax())
    k = 5
    top5_c = set(np.argsort(-chain_logits)[:k].tolist())
    top5_r = set(np.argsort(-ref_logits)[:k].tolist())
    overlap = len(top5_c & top5_r)
    log(f"CHAIN vs WHOLE: max_abs={d:.3e} rel={d / n:.3e} top1_match={top1} "
        f"top5_overlap={overlap}/{k}")
    if input_leak:
        log(f"  WARNING: input_ids leaked into non-first stage(s): {input_leak} "
            f"(embedding-drop rewire failed — check GRAFTED_EMBEDS_NAME)")

    ok = top1 and overlap >= 4 and d / n < 0.5 and not input_leak
    log("EXPORT_VALIDATE_OK" if ok else "EXPORT_VALIDATE_FAIL")
    if not ok:
        raise SystemExit(1)


# ===========================================================================
# config discovery + CLI
# ===========================================================================

def read_arch(src_dir: str):
    """Pull (num_layers, hidden_size, num_kv_shared_layers, num_kv_heads,
    head_dim) from the VLM config's ``text_config`` (or the top level if
    already flat).

    ``num_kv_heads`` / ``head_dim`` are the model's DEFAULT
    ``num_key_value_heads`` / ``head_dim`` (best-effort, may be None). gemma-4
    interleaves sliding/global attention with different KV geometry, so these
    are the per-model defaults ov-runtime's StageConfig carries as ``Option``
    hints — not authoritative per-layer values."""
    with open(os.path.join(src_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    return (tc.get("num_hidden_layers"), tc.get("hidden_size"),
            tc.get("num_kv_shared_layers", 0) or 0,
            tc.get("num_key_value_heads"), tc.get("head_dim"))


def run_export(model, output_dir, num_stages=1, quantization="int4",
               allow_kv_share=False, validate=False,
               boundary_suffix=DEFAULT_LAYERNORM_SUFFIX,
               stage=None, num_layers=None, hidden_size=None,
               no_last_logits_only=False, keep_grafted=False,
               skip_tokenizer_regen=False, **_ignored) -> None:
    """Programmatic entry point for the gemma-4 VLM-IR -> text surgery.

    Mirrors ``qwen36_surgery/export_qwen36_moe.run_export``: ``main()`` parses
    argv then calls this. The generic ``cascadia shard`` dispatcher
    (``tools/export_shards.py``) calls it directly with ``model``,
    ``output_dir``, ``num_stages``, ``quantization``. ``quantization`` is
    accepted for CLI symmetry but IGNORED — the surgery inherits the source
    int4 IR's quantized weights byte-for-byte (never re-quantizes). ``**_ignored``
    is forward-compat for kwargs that are genuinely no-ops here
    (``default_dtype``, ``static_seq``, ``static_context``, …); the dispatcher
    REJECTS ``--target npu`` and ``--layer-split`` before calling, because
    silently ignoring those would not be harmless.
    """
    if num_stages < 1:
        raise SystemExit("--num-stages must be >= 1")

    # Guard: never write into the source model dir (flatten_config /
    # regenerate_tokenizer_bos overwrite files in-place).
    if os.path.abspath(model) == os.path.abspath(output_dir):
        raise SystemExit(
            "--output-dir must differ from --model (this tool rewrites "
            "config.json / tokenizer in the output dir in place)")

    t0 = time.time()
    grafted, info = graft_text_frontend(model,
                                        tag_inputs_embeds=(num_stages > 1))
    log(f"graft done at +{time.time() - t0:.1f}s (per_layer_dropped="
        f"{info['per_layer_dropped']})")

    if num_stages == 1:
        save_whole(grafted, model, output_dir,
                   skip_tokenizer_regen=skip_tokenizer_regen)
    else:
        nl, hs, kv_shared, n_kv_heads, hd = read_arch(model)
        num_layers = num_layers or nl
        hidden_size = hidden_size or hs
        if num_layers is None:
            raise SystemExit(
                "could not determine num_hidden_layers from config; pass "
                "--num-layers")
        if hidden_size is None:
            raise SystemExit(
                "could not determine hidden_size from config; pass "
                "--hidden-size (PipelineConfig.hidden_size is a required "
                "non-optional field, so a null would make the sliced tree "
                "fail to load)")
        if num_stages > num_layers:
            raise SystemExit(
                f"--num-stages ({num_stages}) exceeds decoder layer count "
                f"({num_layers}); would produce empty/invalid stages")
        log(f"slicing {num_layers} layers into {num_stages} stages "
            f"(hidden={hidden_size}, num_kv_shared_layers={kv_shared})")
        slice_stages(
            grafted, model, output_dir, num_stages,
            num_layers, hidden_size, kv_shared, boundary_suffix,
            last_logits_only=not no_last_logits_only,
            only_stage=stage, keep_grafted=keep_grafted,
            skip_tokenizer_regen=skip_tokenizer_regen,
            allow_kv_share=allow_kv_share, validate=validate,
            num_kv_heads=n_kv_heads, head_dim=hd)

    log(f"ALL DONE in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="gemma-4 VLM OpenVINO int4 IR dir")
    ap.add_argument("--output-dir", required=True, help="output dir")
    # --num-stages is primary; --total is a qwen36-CLI-style alias.
    ap.add_argument("--num-stages", type=int, default=1)
    ap.add_argument("--total", type=int, default=None,
                    help="alias for --num-stages (qwen36 CLI style)")
    ap.add_argument("--stage", type=int, default=None,
                    help="export only this stage index (default: all)")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="override decoder layer count (else from config)")
    ap.add_argument("--hidden-size", type=int, default=None,
                    help="override hidden size (else from config)")
    ap.add_argument("--boundary-suffix", default=DEFAULT_LAYERNORM_SUFFIX,
                    help="decoder-layer input-norm op suffix for the slice cut")
    ap.add_argument("--no-last-logits-only", action="store_true",
                    help="keep full [1,T,vocab] logits in the last stage")
    ap.add_argument("--keep-grafted", action="store_true",
                    help="keep the temp grafted whole IR (N>1)")
    ap.add_argument("--skip-tokenizer-regen", action="store_true",
                    help="do not regenerate the BOS tokenizer")
    ap.add_argument("--allow-kv-share", action="store_true",
                    help="permit slicing a num_kv_shared_layers>0 model "
                         "(UNSAFE unless boundaries keep sharing groups intact)")
    ap.add_argument("--validate", action="store_true",
                    help="N>1 on-node parity gate: chained stages vs whole IR")
    args = ap.parse_args()

    num_stages = args.total if args.total is not None else args.num_stages

    run_export(
        model=args.model, output_dir=args.output_dir, num_stages=num_stages,
        allow_kv_share=args.allow_kv_share, validate=args.validate,
        boundary_suffix=args.boundary_suffix, stage=args.stage,
        num_layers=args.num_layers, hidden_size=args.hidden_size,
        no_last_logits_only=args.no_last_logits_only,
        keep_grafted=args.keep_grafted,
        skip_tokenizer_regen=args.skip_tokenizer_regen)


if __name__ == "__main__":
    main()
