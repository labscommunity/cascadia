#!/usr/bin/env python3
"""Gemma-4 multimodal VLM (OpenVINO int4 IR) -> cascadia text-only artifacts.

Productionizes the proven ``_gemma4_spike`` scripts (pawan-01,
``C:\\cascadia\\_gemma4_spike\\``) that took a gemma-4-E2B VLM int4 IR ->
grafted text-only LM -> cascadia ov-genai -> coherent "Paris" @ ~14 tok/s.

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
    N per-stage stateful shards + a ``manifest.json`` (+ per-stage
    ``stage_config.json``), mirroring
    ``tools/qwen36_surgery/export_qwen36_moe.py``'s boundary + sink-ownership
    contract, so a gemma-4 too big for one node (31B) runs pipeline-parallel.
    Stage 0 keeps the grafted ``input_ids -> embeddings`` front-end; the last
    stage keeps the ``logits`` output.

    Slicing is REFUSED for models with ``num_kv_shared_layers > 0`` (E2B=20,
    E4B=18): those layers reuse an earlier layer's KV cache, and this tool does
    not emit cross-stage KV passing (that lives in ``export_gemma4.py``).
    Override with ``--allow-kv-share`` only if you know the boundary keeps each
    sharing group inside one stage.

The graft (``graft_text_frontend``) and the N==1 aux steps
(``regenerate_tokenizer_bos``, ``flatten_config``) are faithful ports of the
proven spike scripts (``step1_surgery.py`` / ``step1_surgery_26b.py`` /
``regen_tok.py`` / ``flatten_config.py``). The N>1 slice
(``slice_stages`` / ``extract_stage`` / ``_validate``) is NEW code, adapted
from the qwen36 surgery; run ``--validate`` on-node to gate it.

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
import json
import os
import re
import shutil
import sys
import time

import openvino as ov
from openvino import opset13 as ops

# Files copied verbatim from the source VLM IR dir into the text output dir.
# (Same list the proven spike scripts copy, minus the sub-IRs we graft away.)
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
    """Insert a Convert if source element-type != target (spike behavior)."""
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
    stays byte-identical to the proven spike output.

    Returns ``(grafted_model, info)`` where ``info`` records
    ``per_layer_dropped`` and the surviving input names.
    """
    core = ov.Core()
    log("=== read sub-IRs ===")
    lm = core.read_model(os.path.join(src_dir, "openvino_language_model.xml"))
    emb = core.read_model(os.path.join(src_dir, "openvino_text_embeddings_model.xml"))
    pl = core.read_model(os.path.join(src_dir, "openvino_text_embeddings_per_layer_model.xml"))

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
    cfg = json.load(open(config_path, encoding="utf-8"))
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
    json.dump(flat, open(config_path, "w", encoding="utf-8"), indent=2)
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
        except Exception as e:  # noqa: BLE001
            log(f"  WARNING: tokenizer regen failed ({e}); the copied VLM "
                f"tokenizer may omit BOS — rerun regen on-node")
    log("WHOLE-IR (N=1) DONE")


# ===========================================================================
# N > 1  slice  (NEW — adapted from qwen36_surgery/export_qwen36_moe.py)
# ===========================================================================

# variable_id patterns for per-layer KV state (optimum VLM IR + gemma4
# present.N/past_key_values.N naming). Permissive on purpose; the discovered
# (variable_id -> layer) map is logged so the on-node operator can verify.
_LAYER_RE = re.compile(
    r"(?:past_key_values|present|past|layers?|blocks?|decoder)[._](\d+)")
# Cache kind (key/value/conv/ssm) taken as the delimited token immediately
# after the layer index — avoids matching the "key"/"value" inside the literal
# "key_values" substring (which is present in EVERY KV variable_id).
_KIND_RE = re.compile(r"[._]\d+[._](value|key|conv|ssm)")


def sink_layer_index(variable_id: str):
    """Global decoder-layer index owning a KV Assign/ReadValue, or None."""
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
        param = ops.parameter(entry.get_partial_shape(),
                              entry.get_element_type(), name="stage_hidden")
        param.output(0).set_names({"stage_hidden"})
        for tgt in list(entry.get_target_inputs()):
            tgt.replace_source_output(param.output(0))
        params_new.append(param)

        # Drop the token-embedding matrix from mid/last stages. The grafted
        # inputs_embeds Output (tagged GRAFTED_EMBEDS_NAME) still feeds two
        # things: (a) layer 0's scaled residual path — pruned in a non-first
        # stage — and (b) mask/position ShapeOf chains that read it for SHAPE
        # only. stage_hidden has the identical [?,?,hidden] shape, so redirect
        # ALL its consumers onto stage_hidden; the embedding subgraph (and its
        # input_ids Parameter, unless per_layer keeps it) then drops out.
        emb_src_out = find_output_by_name(model, GRAFTED_EMBEDS_NAME)
        if emb_src_out is None:
            # Fallback (untagged graft): layer-0 entry is the closest proxy.
            emb_src_out = layer_entry_output(model, 0, suffix)
        if emb_src_out is not None and emb_src_out is not entry:
            n = 0
            for tgt in list(emb_src_out.get_target_inputs()):
                tgt.replace_source_output(param.output(0))
                n += 1
            if n:
                log(f"  rewired {n} grafted-inputs_embeds shape-consumer(s) "
                    f"onto stage_hidden")

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

    # per-layer KV Assign sinks owned by this range
    sinks, all_vids = [], []
    for op in model.get_ops():
        if op.get_type_name() == "Assign":
            vid = op.get_variable_id()
            all_vids.append(vid)
            idx = sink_layer_index(vid)
            if idx is not None and a <= idx <= b:
                sinks.append(op)

    # original Parameters still reachable from results + sinks (input_ids,
    # attention_mask, position_ids, beam_idx — but never stage_hidden)
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
                    and src.get_friendly_name() != "stage_hidden"):
                reach.add(src)
            stack.append(src)
    orig = [p for p in model.get_parameters() if p in reach]

    stage = ov.Model(results, sinks, params_new + orig,
                     f"gemma4_stage_{a}_{b}")

    # Orphan-state rewire (qwen36): global bookkeeping (past-length / mask
    # ShapeOf chains) can reach a ReadValue whose Assign lives in another
    # stage. Redirect each orphan onto a same-kind cache this stage owns.
    # (For num_kv_shared_layers==0 models — the only ones slice_stages permits
    # without --allow-kv-share — every orphan is genuine bookkeeping, not a
    # cross-stage KV dependency.)
    owned = {s.get_variable_id() for s in stage.get_sinks()}
    by_kind, orphans = {}, []
    for op in stage.get_ops():
        if op.get_type_name() != "ReadValue":
            continue
        vid = op.get_variable_id()
        kind = sink_kind(vid)
        if vid in owned:
            by_kind.setdefault(kind, op)
        else:
            orphans.append((op, kind))
    for op, kind in orphans:
        sub = by_kind.get(kind)
        if sub is None:
            raise RuntimeError(
                f"no owned same-kind substitute for orphan state "
                f"{op.get_variable_id()} (kind={kind}) in stage {a}..{b}")
        for tgt in list(op.output(0).get_target_inputs()):
            tgt.replace_source_output(sub.output(0))
    if orphans:
        log(f"  rewired {len(orphans)} orphan state read(s) onto owned caches")

    return stage, sorted({s.get_variable_id() for s in sinks}), all_vids


def slice_stages(grafted: ov.Model, src_dir: str, out_dir: str,
                 num_stages: int, num_layers: int, hidden_size,
                 num_kv_shared_layers: int, suffix: str,
                 last_logits_only: bool, only_stage=None,
                 keep_grafted: bool = False,
                 skip_tokenizer_regen: bool = False,
                 allow_kv_share: bool = False,
                 validate: bool = False) -> None:
    """N>1: slice the grafted IR into per-stage stateful shards + manifest.

    Saves the grafted IR to a temp dir first, then re-reads it per stage
    (mmap, no RAM wall) — mirroring qwen36's ``extract_stage(xml_path, ...)``.
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

    manifest = {
        "arch": "gemma4_text",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_kv_shared_layers": num_kv_shared_layers,
        "source": os.path.basename(os.path.abspath(src_dir)),
        "last_logits_only": last_logits_only,
        "kv_share_overridden": bool(allow_kv_share and num_kv_shared_layers),
        "stages": [],
    }

    for i, (a, b) in enumerate(ranges):
        if only_stage is not None and i != only_stage:
            continue
        first, last = i == 0, i == len(ranges) - 1
        t0 = time.time()
        stage, state_vars, all_vids = extract_stage(
            grafted_xml, a, b, first, last, suffix, last_logits_only)
        sdir = os.path.join(out_dir, f"stage{i}")
        os.makedirs(sdir, exist_ok=True)
        ov.save_model(stage, os.path.join(sdir, "stage.xml"),
                      compress_to_fp16=False)
        inputs = [p.get_friendly_name() for p in stage.get_parameters()]
        stage_cfg = {
            "stage": i, "layer_start": a, "layer_end": b,
            "has_embed": first, "has_head": last, "stateful": True,
            "inputs": inputs,
            "state_vars": state_vars,
        }
        with open(os.path.join(sdir, "stage_config.json"), "w") as f:
            json.dump(stage_cfg, f, indent=2)
        manifest["stages"].append(stage_cfg)
        log(f"stage{i}: layers {a}..{b} saved in {time.time() - t0:.0f}s "
            f"inputs={inputs} states={len(state_vars)}")
        if i == 0:
            log(f"  (discovered {len(all_vids)} Assign variable_ids; "
                f"sample: {all_vids[:4]})")

    # --stage i is an incremental single-stage export: don't clobber a
    # complete manifest.json (finding #5).
    if only_stage is None:
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    else:
        log(f"  (--stage {only_stage}: skipping manifest.json write)")

    # tokenizer/detokenizer/config alongside the stages (single-dir UX), same
    # as qwen36 — plus the proven BOS regen + config flatten.
    copy_aux(src_dir, out_dir)
    flatten_config(os.path.join(out_dir, "config.json"))
    if skip_tokenizer_regen:
        log("skipping BOS tokenizer regen (--skip-tokenizer-regen)")
    else:
        try:
            regenerate_tokenizer_bos(out_dir)
        except Exception as e:  # noqa: BLE001
            log(f"  WARNING: tokenizer regen failed ({e}); rerun on-node")

    # Parity gate (on-node): chained stages vs the grafted whole IR. Run before
    # the temp grafted IR is removed. Can't chain a partial export.
    if validate:
        if only_stage is not None:
            log("  (--validate skipped: --stage exports a single shard, "
                "cannot chain)")
        else:
            _validate(grafted_xml, out_dir, ranges, last_logits_only)

    if not keep_grafted:
        shutil.rmtree(grafted_dir, ignore_errors=True)
    log("SLICED (N>1) DONE")


def _validate(grafted_xml: str, out_dir: str, ranges, last_logits_only: bool,
              device: str = "CPU") -> None:
    """On-node parity gate for the N>1 slice.

    Ports the intent of qwen36's ``_validate`` + ``probe_chain_vs_full_prompt``:
    run the grafted WHOLE IR and the CHAINED stages on the same fixed short
    prompt, feeding each stage's output as the next stage's ``stage_hidden``
    input (the tiling contract). Assert last-position top-1 logit agreement,
    top-5 overlap, and bounded relative drift. Also asserts mid/last stages
    carry NO ``input_ids`` (finding #4). Prints EXPORT_VALIDATE_OK/FAIL and
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
            if nm == "stage_hidden":
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
        sm = core.read_model(os.path.join(out_dir, f"stage{i}", "stage.xml"))
        sc = core.compile_model(sm, device)
        names = [inp.get_any_name() for inp in sc.inputs]
        if i != 0 and any("input_ids" in n for n in names):
            input_leak.append((i, names))
        out = sc.create_infer_request().infer(feeds(sc, hidden=hidden))
        hidden = np.asarray(out[sc.outputs[0]]).astype(np.float32)
        log(f"  stage{i} ran, out shape {hidden.shape} inputs={names}")
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
    """Pull (num_layers, hidden_size, num_kv_shared_layers) from the VLM
    config's ``text_config`` (or the top level if already flat)."""
    cfg = json.load(open(os.path.join(src_dir, "config.json"), encoding="utf-8"))
    tc = cfg.get("text_config", cfg)
    return (tc.get("num_hidden_layers"), tc.get("hidden_size"),
            tc.get("num_kv_shared_layers", 0) or 0)


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
    if num_stages < 1:
        raise SystemExit("--num-stages must be >= 1")

    # Guard: never write into the source model dir (flatten_config /
    # regenerate_tokenizer_bos overwrite files in-place).
    if os.path.abspath(args.model) == os.path.abspath(args.output_dir):
        raise SystemExit(
            "--output-dir must differ from --model (this tool rewrites "
            "config.json / tokenizer in the output dir in place)")

    t0 = time.time()
    grafted, info = graft_text_frontend(args.model,
                                        tag_inputs_embeds=(num_stages > 1))
    log(f"graft done at +{time.time() - t0:.1f}s (per_layer_dropped="
        f"{info['per_layer_dropped']})")

    if num_stages == 1:
        save_whole(grafted, args.model, args.output_dir,
                   skip_tokenizer_regen=args.skip_tokenizer_regen)
    else:
        nl, hs, kv_shared = read_arch(args.model)
        num_layers = args.num_layers or nl
        hidden_size = args.hidden_size or hs
        if num_layers is None:
            raise SystemExit(
                "could not determine num_hidden_layers from config; pass "
                "--num-layers")
        if num_stages > num_layers:
            raise SystemExit(
                f"--num-stages ({num_stages}) exceeds decoder layer count "
                f"({num_layers}); would produce empty/invalid stages")
        log(f"slicing {num_layers} layers into {num_stages} stages "
            f"(hidden={hidden_size}, num_kv_shared_layers={kv_shared})")
        slice_stages(
            grafted, args.model, args.output_dir, num_stages,
            num_layers, hidden_size, kv_shared, args.boundary_suffix,
            last_logits_only=not args.no_last_logits_only,
            only_stage=args.stage, keep_grafted=args.keep_grafted,
            skip_tokenizer_regen=args.skip_tokenizer_regen,
            allow_kv_share=args.allow_kv_share, validate=args.validate)

    log(f"ALL DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
