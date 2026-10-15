#!/usr/bin/env python3
"""Turn a stateful VLM stage tree into the stateless static-shape tree that
ov-runtime's static (NPU) path drives.

`cut_vlm_lm.py` writes stateful stages: each layer's KV lives in an OpenVINO
state variable (ReadValue -> Gather(beam_idx) -> Concat -> Assign). The
static path instead keeps a host-side KV ring per stage (runtime.rs
`StaticKv`) and needs the export_shards NPU contract:

  inputs   inputs_embeds | hidden_states [1, S(, hidden)], attention_mask
           [1, context], position_ids [1, S], past_key_values.{i}.key|value
           [1, kv_heads, context - S, head_dim] (f16)
  outputs  index 0 logits | hidden_states, then present.{i}.key|value
           [1, kv_heads, context, head_dim] (f16)

with every dimension static. This tool rewrites each stage that way:
`beam_idx` folds to the constant [0] (batch 1), each ReadValue becomes a
`past_key_values` Parameter and each Assign's value a `present` Result,
numbered by the layer that owns it within the stage. The head's
last-position logits slice is dropped so a prefill chunk returns every row
(the runtime reads the last REAL row, not the last padded one).

`openvino._offline_transformations.stateful_to_stateless_transformation`
does the same job for whole optimum exports but rejects cut stages (their
beam_idx Gathers do not match its pattern), hence this tool.

Layers may differ in KV geometry (Gemma 4: 8x256 sliding, 2x512 global);
stage_config.json then lists `kv_layers` ([kv_heads, head_dim] per layer) for
the runtime's ring. Gemma 4 trees keep `token_type_ids` (bidirectional
attention inside an image), so they need `--prefill-seq` at least one image
wide, and context + prefill-seq within the sliding window: the IR masks the
window by ring index, which runs ahead of the real distance across the
ring's padding.

Usage:
  python static_variant.py --tree <stateful tree> --output-dir <static tree> \
      --context 2048 [--prefill-seq 256] [--stage I]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import openvino as ov
from openvino import opset13 as ops

KV_RE = re.compile(r"past_key_values\.(\d+)\.(key|value)")


def log(msg: str) -> None:
    print(msg, flush=True)


def fold_beam_idx(model: ov.Model) -> None:
    """Replace the `beam_idx` Parameter with the constant [0]: the static
    path serves one sequence, so every beam Gather is the identity."""
    for p in model.get_parameters():
        if p.get_friendly_name() == "beam_idx":
            zero = ops.constant(np.array([0], dtype=np.int32))
            if p.get_element_type() != ov.Type.i32:
                zero = ops.convert(zero, p.get_element_type())
            for t in list(p.output(0).get_target_inputs()):
                t.replace_source_output(zero.output(0))
            model.remove_parameter(p)
            return


def kv_order(variable_ids: list[str]) -> list[str]:
    """Variable ids in (layer, key-before-value) order."""
    def key(v: str):
        m = KV_RE.match(v)
        if not m:
            raise SystemExit(f"unexpected KV variable id {v!r}")
        return int(m.group(1)), m.group(2) != "key"
    return sorted(variable_ids, key=key)


def unslice_logits(model: ov.Model) -> bool:
    """Drop a last-position Slice in front of the head's Result, if any."""
    changed = False
    for r in model.get_results():
        src = r.input_value(0).get_node()
        if src.get_type_name() != "Slice":
            continue
        start = src.input_value(1).get_node()
        if start.get_type_name() == "Constant" and list(start.get_data().ravel()) == [-1]:
            r.input(0).replace_source_output(src.input_value(0))
            changed = True
    return changed


def make_stateless(model: ov.Model, has_head: bool) -> list[tuple[int, int]]:
    """Rewrite `model` in place; returns each layer's (kv_heads, head_dim)."""
    fold_beam_idx(model)
    reads = {n.get_variable_id(): n for n in model.get_ordered_ops()
             if n.get_type_name() == "ReadValue"}
    assigns = {s.get_variable_id(): s for s in model.get_sinks()}
    if set(reads) != set(assigns):
        raise SystemExit("ReadValue/Assign variables do not pair up; not a cut stage")
    order = kv_order(list(reads))
    if len(order) % 2:
        raise SystemExit(f"odd number of KV variables ({len(order)})")
    geom = []
    params, results = [], []
    for j, vid in enumerate(order):
        local, kind = j // 2, KV_RE.match(vid).group(2)
        if (kind == "key") != (j % 2 == 0):
            raise SystemExit(f"layer {local}: key/value variables are not paired ({vid})")
        rv = reads[vid]
        shape = rv.get_output_partial_shape(0)
        if shape.rank.get_length() != 4 or not shape[1].is_static or not shape[3].is_static:
            raise SystemExit(f"{vid}: KV shape {shape} is not [batch, kv_heads, seq, head_dim]")
        g = (shape[1].get_length(), shape[3].get_length())
        if kind == "key":
            geom.append(g)
        elif geom[-1] != g:
            raise SystemExit(f"layer {local}: key {geom[-1]} and value {g} geometries differ")
        name = f"past_key_values.{local}.{kind}"
        p = ops.parameter([1, shape[1], -1, shape[3]], ov.Type.f16, name=name)
        p.output(0).set_names({name})
        past = p.output(0)
        if rv.get_output_element_type(0) != ov.Type.f16:
            past = ops.convert(past, rv.get_output_element_type(0)).output(0)
        for t in list(rv.output(0).get_target_inputs()):
            t.replace_source_output(past)
        a = assigns[vid]
        present = a.input_value(0)
        if present.get_element_type() != ov.Type.f16:
            present = ops.convert(present, ov.Type.f16).output(0)
        r = ops.result(present)
        r.output(0).set_names({f"present.{local}.{kind}"})
        model.remove_sink(a)
        params.append(p)
        results.append(r)
    for v in list(model.get_variables()):
        model.remove_variable(v)
    model.add_parameters(params)
    model.add_results(results)
    if has_head:
        unslice_logits(model)
    model.outputs[0].set_names({"logits" if has_head else "hidden_states"})
    model.validate_nodes_and_infer_types()
    return geom


def pin_shapes(model: ov.Model, seq: int, context: int, past: int | None = None) -> None:
    """Make every input static for a `seq`-row query over `context` slots.
    `seq == -1` leaves the query width (and so the mask length) dynamic over a
    fixed `past`: one IR then serves seq=1 decode and wide prefill chunks."""
    if seq != -1:
        past = context - seq
    shapes = {}
    canon = ("inputs_embeds", "hidden_states", "input_ids", "attention_mask",
             "position_ids", "token_type_ids")
    for inp in model.inputs:
        names = inp.get_names()
        name = inp.get_any_name()
        # A port's canonical name may be an alias (the Gemma surgery's stage 0
        # is `grafted_inputs_embeds`, aliased `inputs_embeds`); reshape keys
        # take any of a port's names.
        kind = next((c for c in canon if c in names), name)
        ps = inp.get_partial_shape()
        if kind in ("inputs_embeds", "hidden_states"):
            shapes[name] = [1, seq, ps[2].get_length()]
        elif kind == "input_ids":
            shapes[name] = [1, seq]
        elif kind == "attention_mask":
            shapes[name] = [1, context if seq != -1 else -1]
        elif kind == "position_ids":
            if ps.rank.get_length() != 2:
                raise SystemExit(f"position_ids rank {ps.rank} (multi-axis rope) is not supported")
            shapes[name] = [1, seq]
        elif name.startswith("past_key_values."):
            shapes[name] = [1, ps[1].get_length(), past, ps[3].get_length()]
        elif kind == "token_type_ids":
            shapes[name] = [1, seq]
        else:
            raise SystemExit(f"unexpected stage input {name!r}")
    model.reshape(shapes)
    bad = [f"{p.get_any_name()}{p.partial_shape}"
           for p in list(model.inputs) + list(model.outputs) if not p.partial_shape.is_static]
    if bad and seq != -1:
        raise SystemExit(f"reshape left dynamic dims: {bad}")


def copy_file(src: str, dst: str) -> None:
    """A real copy, not a hardlink: cut_vlm_lm.py rewrites these paths in
    place, which would change a linked static tree under it."""
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy2(src, dst)


def convert_stage(tree: str, out: str, i: int, context: int, prefill_seq: int,
                  shared_prefill: bool = False) -> None:
    sdir, odir = os.path.join(tree, f"stage_{i}"), os.path.join(out, f"stage_{i}")
    with open(os.path.join(sdir, "stage_config.json")) as f:
        meta = json.load(f)
    if not meta.get("stateful", True):
        raise SystemExit(f"stage_{i} is already stateless")
    os.makedirs(odir, exist_ok=True)
    xml = os.path.join(sdir, "openvino_model.xml")
    t0 = time.time()
    core = ov.Core()
    model = core.read_model(xml)
    geom = make_stateless(model, meta["has_head"])
    if shared_prefill:
        # CPU/GPU: one IR with a dynamic query width serves decode and
        # prefill chunks, so the weights are compiled once, not twice.
        pin_shapes(model, -1, context, past=context - 1)
    else:
        pin_shapes(model, 1, context)
    ov.save_model(model, os.path.join(odir, "openvino_model.xml"), compress_to_fp16=False)
    kinds = sorted(set(geom))
    log(f"stage_{i}: {len(geom)} layers, kv {kinds}, decode IR saved ({time.time() - t0:.0f}s)")
    meta.update({
        "stateful": False, "static_seq": 1, "static_context": context,
        "static_prefill_seq": None, "static_prefill_context": None,
        # Uniform trees keep the scalar pair; mixed ones list every layer.
        "num_kv_heads": geom[0][0], "head_dim": geom[0][1],
        "kv_layers": [list(g) for g in geom] if len(kinds) > 1 else None,
        "target": "npu",
        "static_from": "tools/vlm_surgery/static_variant.py",
    })
    prefill = os.path.join(odir, "openvino_prefill_model.xml")
    for stale in (prefill, prefill[:-4] + ".bin"):
        if os.path.exists(stale):
            os.remove(stale)
    if prefill_seq > 1 and shared_prefill:
        meta.update({"static_prefill_seq": prefill_seq,
                     "static_prefill_context": context - 1 + prefill_seq,
                     "prefill_shares_decode": True})
        log(f"stage_{i}: the decode IR also serves prefill chunks of {prefill_seq}")
    elif prefill_seq > 1:
        # Same contract as export_shards: the prefill window's past length
        # equals the decode model's, so both share one host-side ring.
        pctx = context - 1 + prefill_seq
        pin_shapes(model, prefill_seq, pctx)
        ov.save_model(model, prefill, compress_to_fp16=False)
        meta.update({"static_prefill_seq": prefill_seq, "static_prefill_context": pctx})
        log(f"stage_{i}: prefill IR seq {prefill_seq} context {pctx} saved "
            f"({time.time() - t0:.0f}s)")
    meta["inputs"] = [p.get_any_name() for p in model.inputs if
                      not p.get_any_name().startswith("past_key_values.")]
    with open(os.path.join(odir, "stage_config.json"), "w") as f:
        json.dump(meta, f, indent=2)


def copy_root(tree: str, out: str) -> None:
    """Everything outside stage_* (embedding IRs, tokenizer, configs)."""
    for name in os.listdir(tree):
        src = os.path.join(tree, name)
        if name.startswith("stage_") or name == "pipeline_config.json":
            continue
        if os.path.isdir(src):
            os.makedirs(os.path.join(out, name), exist_ok=True)
            for fn in os.listdir(src):
                if os.path.isfile(os.path.join(src, fn)):
                    copy_file(os.path.join(src, fn), os.path.join(out, name, fn))
        else:
            copy_file(src, os.path.join(out, name))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", required=True, help="stateful tree (cut_vlm_lm.py output)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--context", type=int, default=2048,
                    help="static context (KV ring is context-1 tokens)")
    ap.add_argument("--prefill-seq", type=int, default=0,
                    help="also write a chunked-prefill IR with this many rows")
    ap.add_argument("--shared-prefill", action="store_true",
                    help="CPU/GPU: one IR with a dynamic query width serves both decode and "
                         "prefill (weights compiled once); NPU needs the separate static IRs")
    ap.add_argument("--stage", type=int, default=None,
                    help="convert only this stage (a worker holding one stage)")
    args = ap.parse_args()
    tree, out = os.path.abspath(args.tree), os.path.abspath(args.output_dir)
    if out == tree or out.startswith(tree + os.sep) or tree.startswith(out + os.sep):
        raise SystemExit("--output-dir must be outside --tree (and not contain it)")
    if args.context < 2:
        raise SystemExit("--context must be >= 2")
    if args.prefill_seq > args.context - 1:
        raise SystemExit(f"--prefill-seq must be <= --context - 1 ({args.context - 1})")
    with open(os.path.join(args.tree, "pipeline_config.json")) as f:
        pipeline = json.load(f)
    v = pipeline.get("vision") or {}
    # Gemma 4 trees default to token_type_ids when the block does not say.
    if v.get("token_type_ids", v.get("family") == "gemma4") and args.prefill_seq < 2:
        raise SystemExit(
            f"vision family {v.get('family')!r} attends bidirectionally inside images "
            "(token_type_ids): an image must sit whole in one inference, so pass "
            "--prefill-seq at least one image wide (Gemma 4: 280)")
    with open(os.path.join(args.tree, "config.json")) as f:
        window = json.load(f).get("sliding_window")
    if window and args.context + max(args.prefill_seq, 1) > window + 1:
        raise SystemExit(
            f"--context {args.context} + --prefill-seq {args.prefill_seq} exceeds the "
            f"sliding window ({window}) + 1: the IR masks the window by ring index, so "
            "a longer ring would cut sliding layers off from tokens inside their window")
    os.makedirs(args.output_dir, exist_ok=True)
    marker = os.path.join(args.output_dir, "pipeline_config.json")
    if os.path.exists(marker):
        os.remove(marker)
    stages = range(pipeline["num_stages"]) if args.stage is None else [args.stage]
    for i in stages:
        convert_stage(args.tree, args.output_dir, i, args.context, args.prefill_seq,
                      args.shared_prefill)
    copy_root(args.tree, args.output_dir)
    for s in pipeline["stages"]:
        s.update({"stateful": False, "static_seq": 1, "static_context": args.context})
    pipeline["static_from"] = os.path.basename(os.path.abspath(args.tree))
    # Written last, so its presence means every requested stage converted.
    # A worker that holds one stage converts just that one (--stage I).
    with open(marker, "w") as f:
        json.dump(pipeline, f, indent=2)
    log(f"STATIC DONE -> {args.output_dir}")


if __name__ == "__main__":
    sys.exit(main())
