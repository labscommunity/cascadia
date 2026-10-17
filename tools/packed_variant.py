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

Some families build more than one mask. Gemma 4's stages feed their SDPAs from
TWO producers: one for the sliding-window layers and one for the full-attention
layers (both also open each image bidirectionally, from `token_type_ids`). Each
producer is classified from the tree's `config.json` `layer_types` (by the
layer index in the producer's name, cross-checked against the SDPA order) and
replaced by its own Parameter: `attn_mask_4d` for full attention,
`attn_mask_sliding_4d` for the sliding window. Inputs left with no consumer
(`attention_mask`, `token_type_ids`) are dropped, and `stage_config.json`
records the ports under `packed_masks` (plus `packed_sliding_window`) so the
runtime knows what to feed. One full-attention producer (every family served
before Gemma 4) takes the original edit unchanged and records nothing new.

Reusable on an already-exported stage (no HF model, no torch, no re-trace):

    python tools/packed_variant.py <stage_dir> --slots 8
"""

import json
import os
import re

import openvino as ov

try:  # OV moved op helpers between namespaces across releases
    from openvino.runtime import op as _ovop
except ImportError:  # pragma: no cover - depends on installed OV version
    from openvino import op as _ovop


FULL, SLIDING = "full_attention", "sliding_attention"
# Parameter name per mask kind. The full-attention name is the one every
# packed IR has had; the runtime feeds whichever of the two the IR declares.
MASK_PORTS = {FULL: "attn_mask_4d", SLIDING: "attn_mask_sliding_4d"}
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _kind(layer_type) -> str:
    if layer_type in (FULL, SLIDING):
        return layer_type
    raise RuntimeError(f"unknown layer type {layer_type!r} in config.json layer_types")


def classify_mask_producers(sdpa, layer_types=None, layer_start: int = 0, num_layers=None):
    """Map each SDPA mask producer's friendly name to FULL or SLIDING.

    Without sliding layers in `layer_types` every producer is full attention
    (the families served before Gemma 4).
    With it, a producer's kind comes from the layer index in its name (the
    surgery keeps the full model's node names, and the shared mask is named
    after the first layer that built it, e.g. `layers.5.self_attn/.../Slice_11`).
    When the stage has one SDPA per layer, the SDPA order gives each consumer's
    own layer; that must agree, and it decides when the name carries no index.
    """
    producers = {}
    for k, node in enumerate(sdpa):
        src = node.input_value(3).get_node()
        producers.setdefault(src.get_friendly_name(), []).append(k)
    if not layer_types or SLIDING not in layer_types:
        # No sliding layers anywhere: whatever else the config lists, every
        # mask is the plain causal one.
        return {name: FULL for name in producers}
    by_order = num_layers is not None and len(sdpa) == num_layers
    out = {}
    for name, consumers in producers.items():
        named = None
        m = _LAYER_RE.search(name)
        if m and int(m.group(1)) < len(layer_types):
            named = _kind(layer_types[int(m.group(1))])
        ordered = None
        if by_order:
            kinds = {_kind(layer_types[layer_start + k]) for k in consumers}
            if len(kinds) != 1:
                raise RuntimeError(
                    f"mask producer {name} feeds layers of several types {sorted(kinds)}; "
                    "one host mask cannot serve both")
            ordered = kinds.pop()
        if named and ordered and named != ordered:
            raise RuntimeError(
                f"mask producer {name}: its name says {named} but the layers it feeds "
                f"(stage SDPAs {consumers}, layer_start {layer_start}) are {ordered}")
        kind = named or ordered
        if kind is None:
            raise RuntimeError(
                f"cannot tell whether mask producer {name} is sliding or full attention: "
                "no layer index in its name and the stage's SDPA count "
                f"({len(sdpa)}) does not match its layer count ({num_layers})")
        out[name] = kind
    return out


def _reachable_parameters(model):
    """Parameters some Result still depends on."""
    seen, stack, found = set(), list(model.get_results()), set()
    while stack:
        node = stack.pop()
        if node.get_instance_id() in seen:
            continue
        seen.add(node.get_instance_id())
        if node.get_type_name() == "Parameter":
            found.add(node.get_friendly_name())
        for inp in node.inputs():
            stack.append(inp.get_source_output().get_node())
    return found


def _rewire_masks(ov_model, sdpa, kinds, info):
    """Several producers (or a sliding one): one Parameter per mask kind.
    Returns the new parameter list; fills `info` with what changed."""
    by_name = {}
    for node in sdpa:
        src = node.input_value(3).get_node()
        by_name.setdefault(src.get_friendly_name(), src)
    new_params = {}
    for name, src in by_name.items():
        kind = kinds[name]
        port = MASK_PORTS[kind]
        if port not in new_params:
            # Current (possibly dynamic) shape; the caller reshapes to [1,1,S,T].
            p = _ovop.Parameter(
                src.output(0).get_element_type(), src.output(0).get_partial_shape()
            )
            p.set_friendly_name(port)
            p.output(0).set_names({port})
            new_params[port] = p
        p = new_params[port]
        if p.output(0).get_element_type() != src.output(0).get_element_type():
            raise RuntimeError(f"{port}: mask producers disagree on element type")
        for target in list(src.output(0).get_target_inputs()):
            target.replace_source_output(p.output(0))
    old = list(ov_model.get_parameters())
    probe = ov.Model(ov_model.get_results(), old + list(new_params.values()), "probe")
    live = _reachable_parameters(probe)
    dropped = [p.get_friendly_name() for p in old if p.get_friendly_name() not in live]
    for port in new_params:
        if port not in live:
            raise RuntimeError(f"{port} feeds nothing after the rewrite")
    info["packed_masks"] = {port: kind for kind, port in MASK_PORTS.items() if port in new_params}
    info["packed_dropped_inputs"] = dropped
    info["mask_producers"] = {n: kinds[n] for n in by_name}
    return [p for p in old if p.get_friendly_name() in live] + list(new_params.values())


def build_packed_model(
    ov_model,
    packed_seq: int,
    has_embed: bool,
    hidden_size: int,
    layer_types=None,
    layer_start: int = 0,
    num_layers=None,
    info=None,
    deepstack_layers: int = 0,
):
    """Return `ov_model` rewired to take `[1,1,S,T]` mask(s) and seq=S queries.

    Mutates and returns a NEW ov.Model sharing the original's ops; the caller's
    handle should be considered consumed. `layer_types` (config.json, whole
    model) and `layer_start` / `num_layers` (this stage) classify several mask
    producers; `info`, when given, receives what the edit did.
    """
    info = {} if info is None else info
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
    kinds = classify_mask_producers(sdpa, layer_types, layer_start, num_layers)
    if len(feeders) > 1 and set(kinds.values()) == {FULL}:
        raise RuntimeError(
            f"expected ONE shared mask producer across {len(sdpa)} SDPA nodes, "
            f"found {len(feeders)}: {sorted(feeders)}. Several producers are "
            "supported only as sliding + full attention, told apart by config.json "
            "`layer_types` (with sliding layers) beside the stage."
        )
    single = len(feeders) == 1 and set(kinds.values()) == {FULL}

    past_len = None
    for p in ov_model.get_parameters():
        if "past_key_values" in p.output(0).get_any_name():
            past_len = int(p.get_partial_shape().get_shape()[2])
            break
    if past_len is None:
        raise RuntimeError("no past_key_values.* inputs: not a stateless static export")
    packed_context = past_len + packed_seq

    if single:
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
    else:
        params = _rewire_masks(ov_model, sdpa, kinds, info)
    packed = ov.Model(ov_model.get_results(), params, "packed_slots")

    # A port's canonical name may be an alias: the Gemma surgery's stage 0
    # input is `grafted_inputs_embeds`, aliased `inputs_embeds`.
    names = {n for p in packed.inputs for n in p.get_names()}
    # A VLM tree's stage 0 takes rank 0's embedding rows (`inputs_embeds`,
    # tools/vlm_surgery/static_variant.py) instead of ids.
    main_name = next(
        n for n in ("input_ids", "inputs_embeds", "hidden_states") if n in names
    )
    if main_name != "input_ids" and not hidden_size:
        raise RuntimeError(f"{main_name} input needs a hidden size and none was found")
    if has_embed and main_name == "hidden_states":
        raise RuntimeError("embed stage has no input_ids / inputs_embeds input")
    shape_map = {}
    canon = ("input_ids", "inputs_embeds", "hidden_states", "position_ids",
             "token_type_ids", "visual_pos_masks", "deepstack_visual_embeds")
    for p in packed.inputs:
        key = p.get_any_name()
        name = next((c for c in canon if c in p.get_names()), key)
        ps = p.get_partial_shape()
        # A `--shared-prefill` IR leaves the query width dynamic; every port
        # below that has one is given its packed shape.
        shape = [int(d) for d in ps.get_shape()] if ps.is_static else None
        if "past_key_values" in name:
            pass
        elif name in MASK_PORTS.values():
            shape = [1, 1, packed_seq, packed_context]
        elif name == "token_type_ids":
            # Kept only when something besides the masks still reads it.
            shape = [1, packed_seq]
        elif name == main_name:
            shape = (
                [1, packed_seq] if name == "input_ids" else [1, packed_seq, hidden_size]
            )
        elif name == "position_ids":
            # Qwen2.5-VL / Qwen3-VL M-RoPE: [rows, 1, S]; 1-D otherwise.
            rank = ps.rank.get_length()
            if rank == 3 and ps[0].is_static:
                shape = [ps[0].get_length(), 1, packed_seq]
            else:
                shape = [1, packed_seq]
        elif name == "visual_pos_masks":
            # Qwen3-VL: 1 on the image rows of this packed inference.
            shape = [1, packed_seq]
        elif name == "deepstack_visual_embeds":
            # Qwen3-VL: [layers, S, hidden]; the LM gathers at most one row per
            # image token, so S rows (padded) is always enough.
            layers = ps[0].get_length() if ps[0].is_static else deepstack_layers
            if layers <= 0:
                raise RuntimeError(
                    "deepstack_visual_embeds layer count is unknown; the tree's "
                    "pipeline_config.json needs vision.deepstack_layers"
                )
            shape = [layers, packed_seq, hidden_size]
        if shape is None:
            raise RuntimeError(f"stage input {name}{ps} is dynamic and has no packed shape")
        shape_map[key] = ov.PartialShape(shape)
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


def stage_hidden_size(model, cfg: dict, stage_dir: str):
    """Row width of the stage's embedding / hidden input.

    `stage_config.hidden_size` when present; else the IR's `inputs_embeds` /
    `hidden_states` port (found by any of its names: the Gemma surgery's
    stage 0 port is `grafted_inputs_embeds`, aliased `inputs_embeds`); else
    the tree's config.json (`hidden_size`, or `text_config.hidden_size`).
    None on an id-fed stage that has none of these (it needs no width).
    """
    if cfg.get("hidden_size") is not None:
        return int(cfg["hidden_size"])
    for p in model.inputs:
        if p.get_names() & {"inputs_embeds", "hidden_states"}:
            dim = p.get_partial_shape()[2]
            if dim.is_static:
                return dim.get_length()
    path = os.path.join(os.path.dirname(os.path.abspath(stage_dir)), "config.json")
    if os.path.exists(path):
        with open(path) as fh:
            c = json.load(fh)
        width = c.get("hidden_size") or (c.get("text_config") or {}).get("hidden_size")
        if width:
            return int(width)
    return None


def tree_deepstack_layers(stage_dir: str) -> int:
    """Qwen3-VL DeepStack layer count from the tree's pipeline_config.json
    (one level above the stage), 0 when the tree has no DeepStack."""
    path = os.path.join(os.path.dirname(os.path.abspath(stage_dir)), "pipeline_config.json")
    if not os.path.exists(path):
        return 0
    with open(path) as fh:
        p = json.load(fh)
    return int((p.get("vision") or {}).get("deepstack_layers") or 0)


def tree_layer_types(stage_dir: str):
    """`(layer_types, sliding_window)` from the tree's config.json (one level
    above the stage), or `(None, None)` when the tree has neither."""
    path = os.path.join(os.path.dirname(os.path.abspath(stage_dir)), "config.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as fh:
        c = json.load(fh)
    text = c.get("text_config") or {}
    layer_types = c.get("layer_types") or text.get("layer_types")
    window = c.get("sliding_window") or text.get("sliding_window")
    return layer_types, window


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
    hidden_size = stage_hidden_size(model, cfg, stage_dir)
    layer_types, window = tree_layer_types(stage_dir)
    deepstack_layers = tree_deepstack_layers(stage_dir)
    start, end = cfg.get("layer_start"), cfg.get("layer_end")
    info = {}
    packed, packed_context = build_packed_model(
        model,
        packed_seq,
        bool(cfg.get("has_embed")),
        int(hidden_size) if hidden_size is not None else 0,
        layer_types=layer_types,
        layer_start=int(start or 0),
        num_layers=(int(end) - int(start)) if start is not None and end else None,
        info=info,
        deepstack_layers=deepstack_layers,
    )
    if "packed_masks" in info and SLIDING in info["packed_masks"].values() and not window:
        raise RuntimeError(
            f"{stage_dir}: the stage has sliding-window layers but config.json gives "
            "no sliding_window"
        )
    out = os.path.join(stage_dir, "openvino_packed_model.xml")
    ov.save_model(packed, out, compress_to_fp16=True)

    past_len = int(cfg["static_context"]) - int(cfg["static_seq"])
    cfg["packed_slots"] = slots
    cfg["packed_seq"] = packed_seq
    cfg["packed_context"] = packed_context
    cfg["packed_region"] = past_len // slots
    # Only a multi-mask (or sliding) rewrite records its ports; a stage
    # without `packed_masks` has the one full-attention `attn_mask_4d`.
    for key in ("packed_masks", "packed_sliding_window", "packed_dropped_inputs"):
        cfg.pop(key, None)
    if "packed_masks" in info:
        cfg["packed_masks"] = info["packed_masks"]
        cfg["packed_dropped_inputs"] = info["packed_dropped_inputs"]
        if SLIDING in info["packed_masks"].values():
            cfg["packed_sliding_window"] = int(window)
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    size_mb = os.path.getsize(out.replace(".xml", ".bin")) / 1e6
    print(
        f"  packed variant: {out} ({size_mb:.0f} MB) "
        f"slots={slots} seq={packed_seq} context={packed_context} "
        f"region={past_len // slots}"
    )
    if "packed_masks" in info:
        print(
            f"  mask producers {info['mask_producers']} -> ports {info['packed_masks']}"
            f"{f' window={window}' if 'packed_sliding_window' in cfg else ''}; "
            f"dropped inputs {info['packed_dropped_inputs']}"
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
