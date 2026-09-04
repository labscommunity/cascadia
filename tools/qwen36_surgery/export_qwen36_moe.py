#!/usr/bin/env python3
"""Qwen3.5-family shard exporter — IR surgery on the official int4 IR.

Covers every `qwen3_5*` model_type that optimum-intel exports with the
VLM layout (`openvino_language_model.xml` + text-embeddings IR):

  * `qwen3_5_moe` — Qwen3.5/3.6-35B-A3B (hybrid GatedDeltaNet + 256-expert
    MoE; 40 layers, hidden 2048)
  * `qwen3_5`     — Qwen3.8-27B (hybrid GatedDeltaNet, dense MLP; 64 layers,
    hidden 5120)

Cuts the whole-model `openvino_language_model.xml` into per-stage stateful
IRs at decoder-layer boundaries. No re-export, no re-quantization: stages
inherit Intel's artifacts byte-for-byte (probes proved bit-exact parity —
see probe_shell_extract*.py and docs/architectures/qwen36-moe-support.md).
The MoE block is opaque to the surgery (no expert slicing), which is why
the dense sibling needs nothing but the model shape.

Model shape (hidden size, layer count, per-layer attention type) is read
from the model dir's `config.json` (`text_config` when nested), never
hard-coded: state-variable ids are numbered by LAYER-TYPE SEQUENCE, so
the exporter walks `layer_types` to map a global layer index onto its
`past.{conv,ssm}.<m>cache` / `past.{key,value}.<k>cache` pair.

Boundary contract (proven on the 35B, re-validated on the 27B):
  * stage input  = input_value(0) of `layers.A.input_layernorm/aten::pow/Power`
                   (replaced with a `stage_hidden` Parameter; first stage
                   keeps the natural `inputs_embeds` path instead)
  * stage output = output of `layers.B/aten::add/Add_1`
                   (last stage keeps the natural `logits` output instead)
  * per-layer state rides along as ReadValue/Assign sinks, selected by
    semantic variable_id (see layer_state_vids).

Usage (on a node with the model dir):
  python export_qwen36_moe.py --model <int4-ov dir> --out <dir> \
      --total 2 [--validate]
"""
import argparse
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

# `openvino` is imported lazily inside the functions that touch IRs so the
# pure shape helpers (read_model_spec / layer_state_vids / stage_ranges)
# stay unit-testable on hosts without the runtime.

LINEAR = "linear_attention"
FULL = "full_attention"
DEFAULT_FULL_ATTN_INTERVAL = 4  # 3× DeltaNet : 1× full attention


@dataclass
class ModelSpec:
    """Shape facts the surgery needs, read from config.json."""

    model_type: str  # outer model_type: qwen3_5 | qwen3_5_moe
    hidden: int
    num_layers: int
    layer_types: list = field(default_factory=list)
    vocab: int = 0

    @property
    def family(self) -> str:
        return "qwen3_5"


def synth_layer_types(num_layers: int, interval: int = DEFAULT_FULL_ATTN_INTERVAL) -> list:
    """`full_attention_interval` fallback: full attention at every
    `interval`-th layer (4k+3 for interval 4), DeltaNet elsewhere."""
    return [FULL if (i + 1) % interval == 0 else LINEAR for i in range(num_layers)]


def spec_from_config(raw: dict) -> ModelSpec:
    """Build a ModelSpec from a parsed config.json dict (outer VLM wrapper
    or bare text config)."""
    outer_mt = (raw.get("model_type") or "").lower()
    cfg = raw.get("text_config") or raw
    inner_mt = (cfg.get("model_type") or "").lower()
    mt = outer_mt or inner_mt
    if not (mt.startswith("qwen3_5") or inner_mt.startswith("qwen3_5")):
        raise ValueError(
            f"model_type {outer_mt or inner_mt!r} is not a qwen3_5-family "
            f"model (expected qwen3_5 / qwen3_5_moe); this exporter does IR "
            f"surgery on the official Qwen3.5/3.6/3.8 OpenVINO IRs only"
        )
    hidden = cfg.get("hidden_size")
    num_layers = cfg.get("num_hidden_layers")
    if not isinstance(hidden, int) or not isinstance(num_layers, int):
        raise ValueError(
            "config.json lacks integer hidden_size / num_hidden_layers "
            "(looked under text_config and at the top level)"
        )
    layer_types = cfg.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != num_layers:
        interval = cfg.get("full_attention_interval") or DEFAULT_FULL_ATTN_INTERVAL
        layer_types = synth_layer_types(num_layers, int(interval))
    unknown = sorted(set(layer_types) - {LINEAR, FULL})
    if unknown:
        raise ValueError(f"unsupported layer_types {unknown}; expected only {LINEAR}/{FULL}")
    # Normalise to the HF family name (qwen3_5 / qwen3_5_moe) even when
    # handed a bare text config (qwen3_5_text / qwen3_5_moe_text).
    arch = outer_mt or inner_mt
    if arch.endswith("_text"):
        arch = arch[: -len("_text")]
    return ModelSpec(
        model_type=arch,
        hidden=hidden,
        num_layers=num_layers,
        layer_types=list(layer_types),
        vocab=int(cfg.get("vocab_size") or 0),
    )


def read_model_spec(model_dir: str) -> ModelSpec:
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — the surgery exporter reads hidden_size / "
            f"num_hidden_layers / layer_types from the model dir's config.json"
        )
    with open(path) as f:
        return spec_from_config(json.load(f))


def layer_state_vids(layer_types: list, global_idx: int) -> list:
    """variable_id match strings owned by a global layer index.

    State vars are numbered by LAYER-TYPE SEQUENCE, not globally:
    conv/ssm over the DeltaNet layers in order, key/value over the
    full-attention layers in order (0..29 / 0..9 on the 40-layer 35B,
    0..47 / 0..15 on the 64-layer 27B). The `past.X.<n>cache` form is the
    junction inside the concatenated variable_id and is collision-proof
    (`conv.3cache` never matches `conv.30cache`).
    """
    kind = layer_types[global_idx]
    n = sum(1 for t in layer_types[:global_idx] if t == kind)
    if kind == FULL:
        return [f"past.key.{n}cache", f"past.value.{n}cache"]
    return [f"past.conv.{n}cache", f"past.ssm.{n}cache"]


def stage_ranges(num_layers: int, total: int) -> list:
    if total < 1 or total > num_layers:
        raise ValueError(f"--total must be in 1..{num_layers}, got {total}")
    per = num_layers // total
    ranges = []
    start = 0
    for i in range(total):
        end = start + per - 1 if i < total - 1 else num_layers - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def check_stage_ranges(spec: ModelSpec, ranges: list) -> None:
    """Every stage must OWN at least one full-attention layer: the orphan
    rewire (extract_stage) redirects the global mask/past-length chains'
    reads of early-layer caches onto a same-kind cache the stage owns, and
    a stage with no attention layer has no key/value substitute."""
    for i, (a, b) in enumerate(ranges):
        kinds = set(spec.layer_types[a : b + 1])
        if FULL not in kinds:
            raise ValueError(
                f"stage{i} (layers {a}..{b}) owns no {FULL} layer; with "
                f"{spec.num_layers} layers in a 3:1 pattern the stage count "
                f"is bounded by the number of full-attention layers "
                f"({spec.layer_types.count(FULL)})"
            )


def find_boundary_ports(model, a: int, b: int):
    b_in = b_out = None
    in_name = f"layers.{a}.input_layernorm/aten::pow/Power"
    out_name = f"layers.{b}/aten::add/Add_1"
    for op in model.get_ops():
        n = op.get_friendly_name()
        if n.endswith(in_name):
            b_in = op.input_value(0)
        elif n.endswith(out_name):
            b_out = op.output(0)
    return b_in, b_out


def extract_stage(xml_path: str, spec: ModelSpec, a: int, b: int, first: bool, last: bool):
    import openvino as ov
    from openvino import opset13 as ops

    core = ov.Core()
    model = core.read_model(xml_path)
    b_in, b_out = find_boundary_ports(model, a, b)
    if (not first and b_in is None) or (not last and b_out is None):
        raise RuntimeError(f"boundary not found for stage layers {a}..{b}")

    params_new = []
    if not first:
        param = ops.parameter(b_in.get_partial_shape(), b_in.get_element_type(),
                              name="stage_hidden")
        param.output(0).set_names({"stage_hidden"})
        for tgt in list(b_in.get_target_inputs()):
            tgt.replace_source_output(param.output(0))
        params_new.append(param)
        # Retire the v1 dummy-input wart: upstream mask/position ShapeOf
        # chains read inputs_embeds for its SHAPE only, and stage_hidden
        # has the identical [?,?,hidden] shape — redirect every consumer
        # so the dummy Parameter drops out of the stage entirely.
        emb_param = next(
            (p for p in model.get_parameters()
             if any("embed" in n for n in p.output(0).get_names())),
            None,
        )
        if emb_param is not None:
            n_rewired = 0
            for tgt in list(emb_param.output(0).get_target_inputs()):
                tgt.replace_source_output(param.output(0))
                n_rewired += 1
            print(f"  rewired {n_rewired} inputs_embeds consumers onto stage_hidden",
                  flush=True)

    if last:
        # Natural logits output, sliced to the LAST position: decode (T=1)
        # is unchanged, batched prefill stops materializing [1, T, vocab]
        # (~1 MB/token at the 248320 vocab). manifest sets
        # last_logits_only so the engine skips its own row slicing.
        results = list(model.get_results())
        out_ports = [r.input_value(0) for r in results]
        results = []
        for p in out_ports:
            sl = ops.slice(
                p,
                ops.constant(np.array([-1], dtype=np.int64)),
                ops.constant(np.array([np.iinfo(np.int64).max], dtype=np.int64)),
                ops.constant(np.array([1], dtype=np.int64)),
                ops.constant(np.array([1], dtype=np.int64)),
            )
            results.append(ops.result(sl.output(0)))
    else:
        results = [ops.result(b_out)]
        results[0].output(0).set_names({"stage_hidden_out"})

    # per-layer state sinks for this range
    want = []
    for g in range(a, b + 1):
        want.extend(layer_state_vids(spec.layer_types, g))
    sinks = []
    for op in model.get_ops():
        if op.get_type_name() == "Assign":
            vid = op.get_variable_id()
            if any(w in vid for w in want):
                sinks.append(op)
    expect = 2 * (b - a + 1)
    if len(sinks) != expect:
        raise RuntimeError(
            f"stage layers {a}..{b}: found {len(sinks)} state sinks, expected "
            f"{expect} (2 per layer) — variable_id numbering does not match "
            f"layer_types {spec.layer_types[a:b + 1]}"
        )

    # original Parameters still reachable from results+sinks
    seen, reach = set(), set()
    stack = list(results) + sinks
    while stack:
        node = stack.pop()
        if node.get_instance_id() in seen:
            continue
        seen.add(node.get_instance_id())
        for iv in node.input_values():
            src = iv.get_node()
            if src.get_type_name() == "Parameter" and src.get_friendly_name() != "stage_hidden":
                reach.add(src)
            stack.append(src)
    orig = [p for p in model.get_parameters() if p in reach]

    stage = ov.Model(results, sinks, params_new + orig, f"qwen35_stage_{a}_{b}")

    # Orphan-state rewire: global bookkeeping (past-length / mask chains)
    # derives shapes from EARLY layers' caches (e.g. ShapeOf(key.0)), so
    # non-owning stages reach ReadValues they have no Assign for — the CPU
    # plugin rejects sibling-less ReadValues. All caches of a kind grow in
    # lockstep, so redirect each orphan's consumers to a same-kind cache
    # this stage owns.
    owned = {s.get_variable_id() for s in stage.get_sinks()}
    by_kind = {}
    orphans = []
    for op in stage.get_ops():
        if op.get_type_name() != "ReadValue":
            continue
        vid = op.get_variable_id()
        kind = next(k for k in ("key", "value", "conv", "ssm") if f"past.{k}." in vid)
        if vid in owned:
            by_kind.setdefault(kind, op)
        else:
            orphans.append((op, kind))
    for op, kind in orphans:
        sub = by_kind.get(kind)
        if sub is None:
            raise RuntimeError(f"no owned substitute for orphan state {op.get_variable_id()}")
        for tgt in list(op.output(0).get_target_inputs()):
            tgt.replace_source_output(sub.output(0))
    if orphans:
        print(f"  rewired {len(orphans)} orphan state reads onto owned caches", flush=True)
    return stage


def build_feeds(model, hidden_size: int, hidden=None, embeds=None):
    feeds = {}
    for inp in model.inputs:
        nm = inp.get_any_name()
        dims = [(d.get_length() if d.is_static else 1) for d in inp.get_partial_shape()]
        et = inp.get_element_type().to_dtype()
        if nm == "stage_hidden":
            feeds[nm] = hidden.astype(et)
        elif "embed" in nm:
            dims[-1] = hidden_size
            feeds[nm] = (embeds if embeds is not None else np.zeros(dims)).astype(et)
        elif "attention_mask" in nm:
            feeds[nm] = np.ones(dims, dtype=et)
        else:
            feeds[nm] = np.zeros(dims, dtype=et)
    return feeds


def run_export(model_dir, output_dir, num_stages=2, validate=False):
    """Cut the official int4 OV IR in `model_dir` into `num_stages` stage
    dirs under `output_dir` + manifest + aux files. Entry point for
    `cascadia shard` dispatch (export_shards.py) and for main() below."""
    import openvino as ov

    xml = os.path.join(model_dir, "openvino_language_model.xml")
    if not os.path.exists(xml):
        raise FileNotFoundError(
            f"{xml} not found — the qwen3_5-family exporter does IR surgery "
            f"on the official int4 OpenVINO IR (e.g. an *-int4-ov model dir), "
            f"not on safetensors"
        )
    spec = read_model_spec(model_dir)
    ranges = stage_ranges(spec.num_layers, num_stages)
    check_stage_ranges(spec, ranges)
    print(
        f"model: {spec.model_type} hidden={spec.hidden} layers={spec.num_layers} "
        f"({spec.layer_types.count(LINEAR)} {LINEAR} + "
        f"{spec.layer_types.count(FULL)} {FULL}) vocab={spec.vocab}",
        flush=True,
    )
    manifest = {
        "arch": spec.model_type,
        "family": spec.family,
        "hidden_size": spec.hidden,
        "num_layers": spec.num_layers,
        "layer_types": spec.layer_types,
        "source": os.path.basename(os.path.abspath(model_dir)),
        "last_logits_only": True,
        "stages": [],
    }

    for i, (a, b) in enumerate(ranges):
        first, last = i == 0, i == len(ranges) - 1
        t0 = time.time()
        stage = extract_stage(xml, spec, a, b, first, last)
        sdir = os.path.join(output_dir, f"stage{i}")
        os.makedirs(sdir, exist_ok=True)
        ov.save_model(stage, os.path.join(sdir, "stage.xml"), compress_to_fp16=False)
        info = {
            "stage": i, "layer_start": a, "layer_end": b,
            "has_embed": first, "has_head": last,
            "inputs": [p.get_friendly_name() for p in stage.get_parameters()],
            "state_vars": sorted({s.get_variable_id() for s in stage.get_sinks()}),
        }
        manifest["stages"].append(info)
        print(f"stage{i}: layers {a}..{b} saved in {time.time()-t0:.0f}s "
              f"inputs={info['inputs']} states={len(info['state_vars'])}", flush=True)

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    # aux files the engine needs alongside the stages (single-dir UX)
    import shutil
    for aux in ("openvino_text_embeddings_model.xml", "openvino_text_embeddings_model.bin",
                "tokenizer.json", "generation_config.json", "config.json",
                # API renders /v1/chat/completions through this when present
                # (legacy "role: content" formatting otherwise)
                "chat_template.jinja"):
        src = os.path.join(model_dir, aux)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, aux))
    print("manifest + aux files written", flush=True)

    if not validate:
        return

    _validate(model_dir, output_dir, xml, ranges, spec.hidden)


def _validate(model_dir, output_dir, xml, ranges, hidden_size):
    import openvino as ov

    core = ov.Core()
    # Real token embedding (degenerate random embeds make logits near-flat
    # and top-1 noise-sensitive): embed a fixed token via the model dir's
    # own text-embeddings IR — exactly what the runtime feeds.
    emb_model = core.read_model(os.path.join(model_dir, "openvino_text_embeddings_model.xml"))
    emb_comp = core.compile_model(emb_model, "CPU")
    token = np.array([[1000]], dtype=np.int64)
    embeds = emb_comp.create_infer_request().infer({emb_comp.inputs[0].get_any_name(): token})
    embeds = embeds[emb_comp.outputs[0]].astype(np.float32).reshape(1, 1, hidden_size)
    del emb_comp, emb_model

    # reference: full model logits
    full = core.read_model(xml)
    comp = core.compile_model(full, "CPU")
    ref = comp.create_infer_request().infer(build_feeds(full, hidden_size, embeds=embeds))
    logits_ref = ref[comp.outputs[0]].astype(np.float32)
    del comp, full

    # chained stages
    hidden = None
    out = None
    for i in range(len(ranges)):
        sm = core.read_model(os.path.join(output_dir, f"stage{i}", "stage.xml"))
        sc = core.compile_model(sm, "CPU")
        feeds = build_feeds(sm, hidden_size, hidden=hidden, embeds=embeds if i == 0 else None)
        out = sc.create_infer_request().infer(feeds)
        hidden = out[sc.outputs[0]].astype(np.float32)
        print(f"stage{i} ran, out shape {hidden.shape}", flush=True)
        del sc, sm
    logits_chain = hidden  # last stage's first output = logits

    # Acceptance is TOKEN-level (spec benchmark protocol): the standalone
    # stages compile SDPA/mask fusions in a different order than the full
    # graph, so f16 accumulation injects ~1e-4 rel at each full-attention
    # layer (measured: bit-exact through DeltaNet layers, first divergence
    # at the first full-attn layer, compounding to ~2e-2 hidden / ~1e-1
    # logits over 20 layers). Greedy decode consumes argmax, so validate
    # top-k agreement + a bounded raw drift, not bitwise logits.
    d = float(np.abs(logits_chain - logits_ref).max())
    n = float(np.abs(logits_ref).max()) + 1e-9
    flat_c, flat_r = logits_chain.reshape(-1), logits_ref.reshape(-1)
    top1 = int(flat_c.argmax()) == int(flat_r.argmax())
    k = 5
    top5_c = set(np.argsort(-flat_c)[:k].tolist())
    top5_r = set(np.argsort(-flat_r)[:k].tolist())
    overlap = len(top5_c & top5_r)
    print(f"CHAIN logits max_abs={d:.3e} rel={d/n:.3e} top1_match={top1} "
          f"top5_overlap={overlap}/{k}", flush=True)
    ok = top1 and overlap >= 4 and d / n < 0.5

    # Multi-token greedy: 8 decode tokens from a synthetic prompt, chain
    # vs full at T=1 steps (proto_m3_decode.py measured 64/64 in this
    # regime). f16 fusion-order noise can flip a late near-tie, so accept
    # >= 6/8; the full >=64-token criterion stays engine-level (the
    # qwen36_parity golden test).
    PROMPT_IDS = [9707, 11, 1246, 525, 498, 30]
    N_DEC = 8
    emb_req2 = core.compile_model(
        core.read_model(os.path.join(model_dir, "openvino_text_embeddings_model.xml")),
        "CPU",
    )

    def embed_tok(t):
        a = np.array([[t]], dtype=np.int64)
        r = emb_req2.create_infer_request().infer(
            {emb_req2.inputs[0].get_any_name(): a})
        return r[emb_req2.outputs[0]].astype(np.float32).reshape(1, 1, hidden_size)

    def step_feeds(comp, step, hidden=None, embeds=None):
        f = {}
        for inp in comp.inputs:
            nm = inp.get_any_name()
            et = inp.get_element_type().to_dtype()
            if nm == "stage_hidden":
                f[nm] = hidden.astype(et)
            elif "embed" in nm:
                f[nm] = (embeds if embeds is not None
                         else np.zeros((1, 1, hidden_size))).astype(et)
            elif "attention_mask" in nm:
                f[nm] = np.ones((1, step + 1), dtype=et)
            elif "position" in nm:
                ps = inp.get_partial_shape()
                d0 = ps[0].get_length() if ps[0].is_static else 1
                f[nm] = np.full((d0, 1, 1), step, dtype=et)
            else:
                f[nm] = np.zeros([1], dtype=et)
        return f

    def greedy(paths, label):
        comps = [core.compile_model(core.read_model(p), "CPU") for p in paths]
        reqs = [(c, c.create_infer_request()) for c in comps]
        step, logits, gen = 0, None, []
        for phase_toks in (PROMPT_IDS, None):
            seq = phase_toks if phase_toks is not None else range(N_DEC)
            for item in seq:
                tok = item if phase_toks is not None else int(np.argmax(logits))
                if phase_toks is None:
                    gen.append(tok)
                cur = embed_tok(tok)
                for j, (c, r) in enumerate(reqs):
                    f = step_feeds(c, step,
                                   hidden=None if j == 0 else cur,
                                   embeds=cur if j == 0 else None)
                    cur = r.infer(f)[c.outputs[0]].astype(np.float32)
                logits = cur.reshape(-1)
                step += 1
        del reqs, comps
        print(f"{label} greedy: {gen}", flush=True)
        return gen

    chain_gen = greedy(
        [os.path.join(output_dir, f"stage{i}", "stage.xml") for i in range(len(ranges))],
        "chain")
    full_gen = greedy([xml], "full")
    m = sum(1 for x, y in zip(chain_gen, full_gen) if x == y)
    print(f"MULTI_TOKEN_PARITY {m}/{N_DEC}", flush=True)
    ok = ok and m >= 6

    print("EXPORT_VALIDATE_OK" if ok else "EXPORT_VALIDATE_FAIL", flush=True)
    if ok:
        print("note: >=64-token greedy parity is the engine-level "
              "criterion (qwen36_parity golden test); this validates one "
              f"step token-level + {N_DEC}-token greedy.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="official int4-ov model dir")
    ap.add_argument("--out", required=True, help="output dir for stage dirs")
    ap.add_argument("--total", type=int, default=2)
    ap.add_argument("--validate", action="store_true",
                    help="chain stages vs full model on one synthetic token")
    args = ap.parse_args()
    run_export(args.model, args.out, args.total, args.validate)


if __name__ == "__main__":
    main()
