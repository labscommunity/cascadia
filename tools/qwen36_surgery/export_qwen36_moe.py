#!/usr/bin/env python3
"""Qwen3.6-35B-A3B shard exporter — IR surgery on the official int4 IR.

Cuts the whole-model `openvino_language_model.xml` into per-stage stateful
IRs at decoder-layer boundaries. No re-export, no re-quantization: stages
inherit Intel's artifacts byte-for-byte (probes proved bit-exact parity —
see probe_shell_extract*.py and docs/architectures/qwen36-moe-support.md).

Boundary contract (proven):
  * stage input  = input_value(0) of `layers.A.input_layernorm/aten::pow/Power`
                   (replaced with a `stage_hidden` Parameter; first stage
                   keeps the natural `inputs_embeds` path instead)
  * stage output = output of `layers.B/aten::add/Add_1`
                   (last stage keeps the natural `logits` output instead)
  * per-layer state rides along as ReadValue/Assign sinks, selected by
    semantic variable_id: cache_params.past.{conv,ssm}.<global> for
    DeltaNet layers, cache_params.past.{key,value}.<(global-3)//4> for
    full-attention layers (3:1 pattern, full-attn at 4k+3).

Known v1 wart (documented in spec): mid stages keep `inputs_embeds` as a
dummy input — upstream mask/position ShapeOf chains reference it. Feed
zeros of shape [1,1,hidden]; values are never read, only shapes.

Usage (on a node with the model dir):
  python export_qwen36_moe.py --model <int4-ov dir> --out <dir> \
      --total 2 [--validate]
"""
import argparse
import json
import os
import time

import numpy as np
import openvino as ov
from openvino import opset13 as ops

HIDDEN = 2048
NUM_LAYERS = 40
FULL_ATTN_INTERVAL = 4  # layers 3, 7, ..., 39 are full attention


def layer_state_vids(global_idx: int) -> list[str]:
    """variable_id match strings owned by a global layer index.

    State vars are numbered by LAYER-TYPE SEQUENCE, not globally:
    conv/ssm 0..29 over the 30 DeltaNet layers, key/value 0..9 over the
    10 full-attention layers. The `past.X.<n>cache` form is the junction
    inside the concatenated variable_id and is collision-proof
    (`conv.3cache` never matches `conv.30cache`).
    """
    if (global_idx + 1) % FULL_ATTN_INTERVAL == 0:
        k = global_idx // FULL_ATTN_INTERVAL
        return [f"past.key.{k}cache", f"past.value.{k}cache"]
    m = global_idx - (global_idx + 1) // FULL_ATTN_INTERVAL
    return [f"past.conv.{m}cache", f"past.ssm.{m}cache"]


def stage_ranges(total: int) -> list[tuple[int, int]]:
    per = NUM_LAYERS // total
    ranges = []
    start = 0
    for i in range(total):
        end = start + per - 1 if i < total - 1 else NUM_LAYERS - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def find_boundary_ports(model: ov.Model, a: int, b: int):
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


def extract_stage(xml_path: str, a: int, b: int, first: bool, last: bool) -> ov.Model:
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

    if last:
        # natural logits Result stays
        results = list(model.get_results())
        out_ports = [r.input_value(0) for r in results]
        results = [ops.result(p) for p in out_ports]
    else:
        results = [ops.result(b_out)]
        results[0].output(0).set_names({"stage_hidden_out"})

    # per-layer state sinks for this range
    want = []
    for g in range(a, b + 1):
        want.extend(layer_state_vids(g))
    sinks = []
    for op in model.get_ops():
        if op.get_type_name() == "Assign":
            vid = op.get_variable_id()
            if any(w in vid for w in want):
                sinks.append(op)

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

    stage = ov.Model(results, sinks, params_new + orig, f"qwen36_stage_{a}_{b}")

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


def build_feeds(model, hidden=None, embeds=None):
    feeds = {}
    for inp in model.inputs:
        nm = inp.get_any_name()
        dims = [(d.get_length() if d.is_static else 1) for d in inp.get_partial_shape()]
        et = inp.get_element_type().to_dtype()
        if nm == "stage_hidden":
            feeds[nm] = hidden.astype(et)
        elif "embed" in nm:
            dims[-1] = HIDDEN
            feeds[nm] = (embeds if embeds is not None else np.zeros(dims)).astype(et)
        elif "attention_mask" in nm:
            feeds[nm] = np.ones(dims, dtype=et)
        else:
            feeds[nm] = np.zeros(dims, dtype=et)
    return feeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="official int4-ov model dir")
    ap.add_argument("--out", required=True, help="output dir for stage dirs")
    ap.add_argument("--total", type=int, default=2)
    ap.add_argument("--validate", action="store_true",
                    help="chain stages vs full model on one synthetic token")
    args = ap.parse_args()

    xml = os.path.join(args.model, "openvino_language_model.xml")
    ranges = stage_ranges(args.total)
    manifest = {
        "arch": "qwen3_5_moe", "hidden_size": HIDDEN, "num_layers": NUM_LAYERS,
        "source": os.path.basename(os.path.abspath(args.model)),
        "stages": [],
    }

    for i, (a, b) in enumerate(ranges):
        first, last = i == 0, i == len(ranges) - 1
        t0 = time.time()
        stage = extract_stage(xml, a, b, first, last)
        sdir = os.path.join(args.out, f"stage{i}")
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

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    # aux files the engine needs alongside the stages (single-dir UX)
    import shutil
    for aux in ("openvino_text_embeddings_model.xml", "openvino_text_embeddings_model.bin",
                "tokenizer.json", "generation_config.json", "config.json"):
        src = os.path.join(args.model, aux)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, aux))
    print("manifest + aux files written", flush=True)

    if not args.validate:
        return

    core = ov.Core()
    # Real token embedding (degenerate random embeds make logits near-flat
    # and top-1 noise-sensitive): embed a fixed token via the model dir's
    # own text-embeddings IR — exactly what the runtime feeds.
    emb_model = core.read_model(os.path.join(args.model, "openvino_text_embeddings_model.xml"))
    emb_comp = core.compile_model(emb_model, "CPU")
    token = np.array([[1000]], dtype=np.int64)
    embeds = emb_comp.create_infer_request().infer({emb_comp.inputs[0].get_any_name(): token})
    embeds = embeds[emb_comp.outputs[0]].astype(np.float32).reshape(1, 1, HIDDEN)
    del emb_comp, emb_model

    # reference: full model logits
    full = core.read_model(xml)
    comp = core.compile_model(full, "CPU")
    ref = comp.create_infer_request().infer(build_feeds(full, embeds=embeds))
    logits_ref = ref[comp.outputs[0]].astype(np.float32)
    del comp, full

    # chained stages
    hidden = None
    out = None
    for i in range(len(ranges)):
        sm = core.read_model(os.path.join(args.out, f"stage{i}", "stage.xml"))
        sc = core.compile_model(sm, "CPU")
        feeds = build_feeds(sm, hidden=hidden, embeds=embeds if i == 0 else None)
        out = sc.create_infer_request().infer(feeds)
        hidden = out[sc.outputs[0]].astype(np.float32)
        print(f"stage{i} ran, out shape {hidden.shape}", flush=True)
        last_outputs = sc.outputs
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
    print("EXPORT_VALIDATE_OK" if ok else "EXPORT_VALIDATE_FAIL", flush=True)
    if ok:
        print("note: multi-token greedy parity (>=64 tokens) is the M3' "
              "engine-level criterion; this validates one decode step.", flush=True)


if __name__ == "__main__":
    main()
