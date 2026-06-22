#!/usr/bin/env python3
"""Split MiniMax-M2 OV-IR shells into a GPU-friendly core + a CPU router.

The MoE router subgraph (gate MatMul -> sigmoid -> +e_score_correction_bias
-> TopK -> Gather -> renorm) makes the OpenVINO Intel GPU plugin emit a
kernel that fails at runtime with CL_OUT_OF_RESOURCES (and wedges the
OpenCL context). The attention + KV part of the same shell runs on the
iGPU fine. So for iGPU execution we carve each shell into two graphs:

  shell_core.xml : x, past_k, past_v, past_seq_len
                   -> attn_out_post_norm, attn_residual,
                      shared_expert_out, present_k, present_v
                   (runs on the iGPU)
  router.xml     : apn_in (= attn_out_post_norm)
                   -> routing_ids, routing_weights
                   (runs on the CPU; trivial compute)

`OvMoeRunner` loads these instead of the monolithic shell when the target
device is not CPU and the split files exist. The router input is named
`apn_in` with the same shape/type as `attn_out_post_norm`.

Usage:
  python tools/split_m2_shells.py --model-dir /path/to/m2_export
  (idempotent; skips layers already split unless --force)
"""
import argparse
import os

import openvino as ov
from openvino import Model
from openvino import opset15 as ops

CORE_OUTPUTS = [
    "attn_out_post_norm",
    "attn_residual",
    "shared_expert_out",
    "present_k",
    "present_v",
]
ROUTER_OUTPUTS = ["routing_ids", "routing_weights"]


def _out(model, name):
    for o in model.outputs:
        if name in o.get_names():
            return o
    raise SystemExit(f"output {name!r} not found (have {[list(o.get_names()) for o in model.outputs]})")


def split_one(core, shell_xml, out_dir, force):
    core_xml = os.path.join(out_dir, "shell_core.xml")
    router_xml = os.path.join(out_dir, "router.xml")
    if not force and os.path.exists(core_xml) and os.path.exists(router_xml):
        return "skip"

    # shell_core: keep the 5 non-router outputs; OV prunes the router subgraph.
    m = core.read_model(shell_xml)
    keep = [_out(m, n).get_node() for n in CORE_OUTPUTS]
    shell_core = Model(keep, m.get_parameters())
    shell_core.set_friendly_name("m2_shell_core")
    ov.save_model(shell_core, core_xml, compress_to_fp16=False)

    # router: replace the attn_out_post_norm tensor with an `apn_in` input;
    # OV prunes the attention subgraph, leaving gate->sigmoid->topk->gather.
    r = core.read_model(shell_xml)
    apn_src = _out(r, "attn_out_post_norm").get_node().input_value(0)
    apn_in = ops.parameter(apn_src.get_partial_shape(), apn_src.get_element_type(), name="apn_in")
    apn_in.get_output_tensor(0).set_names({"apn_in"})
    rewired = 0
    for ti in list(apn_src.get_target_inputs()):
        if ti.get_node().get_type_name() != "Result":
            ti.replace_source_output(apn_in.output(0))
            rewired += 1
    if rewired == 0:
        raise SystemExit(f"{shell_xml}: found no router consumer of attn_out_post_norm")
    router = Model([_out(r, n).get_node() for n in ROUTER_OUTPUTS], [apn_in])
    router.set_friendly_name("m2_router")
    ov.save_model(router, router_xml, compress_to_fp16=False)
    return "split"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="M2 export dir (has shells/layer_NN/openvino_model.xml)")
    ap.add_argument("--force", action="store_true", help="re-split layers even if split files exist")
    args = ap.parse_args()

    shells_dir = os.path.join(args.model_dir, "shells")
    if not os.path.isdir(shells_dir):
        raise SystemExit(f"no shells/ under {args.model_dir}")
    core = ov.Core()
    n_split = n_skip = 0
    for layer in sorted(os.listdir(shells_dir)):
        d = os.path.join(shells_dir, layer)
        xml = os.path.join(d, "openvino_model.xml")
        if not os.path.exists(xml):
            continue
        res = split_one(core, xml, d, args.force)
        if res == "split":
            n_split += 1
        else:
            n_skip += 1
    print(f"done: {n_split} split, {n_skip} skipped (already present). "
          f"OvMoeRunner uses shell_core.xml/router.xml on non-CPU devices.")


if __name__ == "__main__":
    main()
