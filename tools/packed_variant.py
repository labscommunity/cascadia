"""Build the packed multi-slot ("seq-as-batch") variant of a static NPU export.

The NPU compiler rejects a batch dimension (`ConvertBatchedLayerTo1N` fails to
legalize `IE.Convolution`), but it compiles seq > 1 — the chunked-prefill
variant already relies on that. So continuous batching on the NPU packs N
requests into the SEQUENCE dimension and isolates them with a block-diagonal
attention mask. That needs a per-query-row mask, which the stock export cannot
express: it takes a 2D `attention_mask [1, T]` that every query row shares.

This module performs the one graph edit that unlocks it. The export builds its
4D additive mask in a SINGLE node whose output feeds every SDPA's mask input, so
replacing that node with a `[1, 1, S, T]` Parameter — and dropping the now-unused
2D `attention_mask` — hands mask construction to the host, which is where the
per-slot policy belongs anyway.

Reusable on an already-exported stage (no HF model, no torch, no re-trace):

    python tools/packed_variant.py <stage_dir> --slots 8
"""

import json
import os

import openvino as ov

try:  # OV moved op helpers between namespaces across releases
    from openvino.runtime import op as _ovop
except ImportError:  # pragma: no cover - depends on installed OV version
    from openvino import op as _ovop


def build_packed_model(ov_model, packed_seq: int, has_embed: bool, hidden_size: int):
    """Return `ov_model` rewired to take a `[1,1,S,T]` mask and seq=S queries.

    Mutates and returns a NEW ov.Model sharing the original's ops; the caller's
    handle should be considered consumed.
    """
    sdpa = [
        o
        for o in ov_model.get_ordered_ops()
        if o.get_type_name() == "ScaledDotProductAttention"
    ]
    if not sdpa:
        raise RuntimeError(
            "no ScaledDotProductAttention nodes: this export does not use the "
            "fused-attention layout the packed variant edits"
        )
    mask_src = sdpa[0].input_value(3).get_node()
    feeders = {s.input_value(3).get_node().get_friendly_name() for s in sdpa}
    if len(feeders) != 1:
        raise RuntimeError(
            f"expected ONE shared mask producer across {len(sdpa)} SDPA nodes, "
            f"found {len(feeders)}: {sorted(feeders)}. Packing needs a single "
            "cut point; per-layer masks would need per-layer edits."
        )

    past_len = None
    for p in ov_model.get_parameters():
        if "past_key_values" in p.output(0).get_any_name():
            past_len = int(p.get_partial_shape().get_shape()[2])
            break
    if past_len is None:
        raise RuntimeError("no past_key_values.* inputs: not a stateless static export")
    packed_context = past_len + packed_seq

    # Create the Parameter at the graph's CURRENT mask shape, then move
    # everything to seq=S in one reshape(). Building it pre-shaped fails SDPA
    # shape inference at construction time, because the rest of the graph is
    # still seq=1 at that moment.
    et = mask_src.output(0).get_element_type()
    cur = [int(d) for d in mask_src.output(0).get_partial_shape().get_shape()]
    new_mask = _ovop.Parameter(et, ov.PartialShape(cur))
    new_mask.set_friendly_name("attn_mask_4d")
    new_mask.output(0).set_names({"attn_mask_4d"})
    for target in list(mask_src.output(0).get_target_inputs()):
        target.replace_source_output(new_mask.output(0))

    old_mask = next(
        p
        for p in ov_model.get_parameters()
        if "attention_mask" in p.output(0).get_any_name()
    )
    params = [p for p in ov_model.get_parameters() if p is not old_mask] + [new_mask]
    packed = ov.Model(ov_model.get_results(), params, "packed_slots")

    main_name = "input_ids" if has_embed else "hidden_states"
    shape_map = {}
    for p in packed.inputs:
        name = p.get_any_name()
        shape = [int(d) for d in p.get_partial_shape().get_shape()]
        if "past_key_values" in name:
            pass
        elif name == "attn_mask_4d":
            shape = [1, 1, packed_seq, packed_context]
        elif name == main_name:
            shape = [1, packed_seq] if has_embed else [1, packed_seq, hidden_size]
        elif name == "position_ids":
            shape = [1, packed_seq]
        shape_map[name] = ov.PartialShape(shape)
    packed.reshape(shape_map)
    packed.validate_nodes_and_infer_types()

    bad = [
        f"{p.get_any_name()}{p.get_partial_shape()}"
        for p in list(packed.inputs) + list(packed.outputs)
        if not p.get_partial_shape().is_static
    ]
    if bad:
        raise RuntimeError(
            f"packed variant reshape to seq={packed_seq} left dynamic dims — "
            f"the NPU compiler will reject this IR: {bad}"
        )
    return packed, packed_context


def write_packed_variant(stage_dir: str, slots: int, packed_seq: int = None) -> dict:
    """Build + save `openvino_packed_model.xml` beside a stage's decode IR and
    patch `stage_config.json`. `packed_seq` defaults to one query row per slot
    (the decode shape); pass a larger value to leave room for prefill chunks."""
    cfg_path = os.path.join(stage_dir, "stage_config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    if cfg.get("stateful", True):
        raise RuntimeError(
            f"{stage_dir} is a stateful (CPU/GPU) export; packed slots are a "
            "stateless static-path feature — re-export with --target npu"
        )
    packed_seq = packed_seq or slots
    src = os.path.join(stage_dir, "openvino_model.xml")
    model = ov.Core().read_model(src)
    packed, packed_context = build_packed_model(
        model,
        packed_seq,
        bool(cfg.get("has_embed")),
        int(cfg["hidden_size"]),
    )
    out = os.path.join(stage_dir, "openvino_packed_model.xml")
    ov.save_model(packed, out, compress_to_fp16=True)

    past_len = int(cfg["static_context"]) - int(cfg["static_seq"])
    cfg["packed_slots"] = slots
    cfg["packed_seq"] = packed_seq
    cfg["packed_context"] = packed_context
    cfg["packed_region"] = past_len // slots
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    size_mb = os.path.getsize(out.replace(".xml", ".bin")) / 1e6
    print(
        f"  packed variant: {out} ({size_mb:.0f} MB) "
        f"slots={slots} seq={packed_seq} context={packed_context} "
        f"region={past_len // slots}"
    )
    return cfg


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage_dir", help="directory holding openvino_model.xml")
    ap.add_argument("--slots", type=int, required=True, help="packed slots (>=2)")
    ap.add_argument(
        "--packed-seq",
        type=int,
        default=None,
        help="query rows per inference (default: one per slot)",
    )
    args = ap.parse_args()
    if args.slots < 2:
        ap.error("--slots must be >= 2")
    write_packed_variant(args.stage_dir, args.slots, args.packed_seq)


if __name__ == "__main__":
    main()
