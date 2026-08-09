#!/usr/bin/env python3
"""T2 spike: glm5 MLA attention projection GEMMs on the OV device (iGPU/CPU).

Builds the five MLA projections (wq_a, wq_b, wkv_a, wkv_b, wo) as compile-once
OV IRs with bf16 weight constants, at exact shapes read from the target model's
manifest.json, and measures:

  * per-projection GEMM time at each batch size,
  * a fused chain (x -> all five, dataflow-chained without rope/softmax/norms),
  * transfer overhead via an identity model at the same activation shapes,
  * the compiled weight precision actually used (bf16 may be upconverted).

Companion of the Rust baseline bench
(`cargo run -p cascadia-engine-sparse-moe --example glm5_attn_bench --release`).
Spec: enterprise repo `docs/superpowers/specs/2026-08-09-glm5-attn-igpu-spike-spec.md`.
Contended runs: start the fleet decode driver on the node first; this probe
does not generate its own contention load.

Usage:
  python tools/glm5_attn_ov_probe.py --src <model dir> [--device GPU] [--iters 50]
"""
import argparse
import json
import os
import statistics
import time

import numpy as np
import openvino as ov
from openvino import Model, PartialShape, Shape, Type
from openvino import opset15 as ops

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="model dir containing manifest.json")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--batches", default="1,512,1024,2048")
    cli = ap.parse_args()

    man = json.load(open(os.path.join(cli.src, "manifest.json")))
    H = man["hidden_size"]
    QL = man["q_lora_rank"]
    KVL = man["kv_lora_rank"]
    NOPE = man["qk_nope_head_dim"]
    ROPE = man["qk_rope_head_dim"]
    VH = man["v_head_dim"]
    NH = man["num_attention_heads"]
    QK = NOPE + ROPE

    # (name, out_dim, in_dim) — shapes per glm/attn.rs
    PROJ = [
        ("wq_a", QL, H),
        ("wq_b", NH * QK, QL),
        ("wkv_a", KVL + ROPE, H),
        ("wkv_b", NH * (NOPE + VH), KVL),
        ("wo", H, NH * VH),
    ]
    print(f"openvino {ov.__version__}  device={cli.device}")
    print(f"shapes: H={H} QL={QL} KVL={KVL} QK={QK} VH={VH} NH={NH}")

    rng = np.random.default_rng(9)


    def bf16_const(out_dim, in_dim):
        bits = (rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.02)
        u16 = (bits.view(np.uint32) >> 16).astype(np.uint16)
        t = ov.Tensor(Type.bf16, Shape([out_dim, in_dim]))
        v = t.data if isinstance(t.data, np.ndarray) else np.frombuffer(t.data, np.uint8)
        v.reshape(-1).view(np.uint8)[:] = np.frombuffer(u16.tobytes(), np.uint8)
        return ov.op.Constant(t, shared_memory=False)


    def lin(x, w_const):
        return ops.matmul(x, ops.convert(w_const, Type.f32), False, True)


    consts = {name: bf16_const(o, i) for name, o, i in PROJ}


    def single_model(name, out_dim, in_dim, batch):
        x = ops.parameter(PartialShape([batch, in_dim]), Type.f32, name="x")
        x.get_output_tensor(0).set_names({"x"})
        y = lin(x, consts[name])
        m = Model([y], [x], name)
        m.outputs[0].tensor.set_names({"y"})
        return m


    def fused_model(batch):
        """x -> all five projections, dataflow-chained, no rope/softmax/norms.
        wo is fed from the v_head slice of wkv_b's output so every GEMM runs at
        its real shape; the missing glue is priced by the spec's roofline."""
        x = ops.parameter(PartialShape([batch, H]), Type.f32, name="x")
        x.get_output_tensor(0).set_names({"x"})
        qa = lin(x, consts["wq_a"])                      # [B, QL]
        q = lin(qa, consts["wq_b"])                      # [B, NH*QK]
        kva = lin(x, consts["wkv_a"])                    # [B, KVL+ROPE]
        kv_lat = ops.slice(
            kva,
            ops.constant(np.array([0], np.int64)),
            ops.constant(np.array([KVL], np.int64)),
            ops.constant(np.array([1], np.int64)),
            ops.constant(np.array([1], np.int64)),
        )                                                # [B, KVL]
        kvb = lin(kv_lat, consts["wkv_b"])               # [B, NH*(NOPE+VH)]
        ctx = ops.slice(
            kvb,
            ops.constant(np.array([0], np.int64)),
            ops.constant(np.array([NH * VH], np.int64)),
            ops.constant(np.array([1], np.int64)),
            ops.constant(np.array([1], np.int64)),
        )                                                # [B, NH*VH]
        out = lin(ctx, consts["wo"])                     # [B, H]
        m = Model([out, q], [x], "fused")
        return m


    def identity_model(batch, dim):
        x = ops.parameter(PartialShape([batch, dim]), Type.f32, name="x")
        x.get_output_tensor(0).set_names({"x"})
        y = ops.add(x, ops.constant(np.float32(0.0)))
        m = Model([y], [x], "ident")
        m.outputs[0].tensor.set_names({"y"})
        return m


    core = ov.Core()


    def compiled_precisions(compiled):
        """Best-effort: the element types the exec graph actually runs."""
        try:
            kinds = set()
            others = set()
            for node in compiled.get_runtime_model().get_ops():
                # Exec graphs wrap ops as "ExecutionNode"; the real kind is in
                # rt_info["layerType"] on plugins that do so. Binding surface
                # varies by version — every accessor is best-effort.
                t = node.get_type_name()
                try:
                    any_v = node.get_rt_info()["layerType"]
                    t = any_v.astype(str) if hasattr(any_v, "astype") else str(any_v)
                except Exception:  # noqa: BLE001
                    pass
                t = t.lower()
                if "matmul" in t or "fullyconnected" in t or "gemm" in t:
                    kinds.add(f"{t}:{node.get_output_element_type(0)}")
                else:
                    others.add(t)
            if kinds:
                return sorted(kinds)
            # Plugin renamed the op — dump what IS there so on-node diagnosis works.
            return [f"<no matmul-like op; exec ops: {sorted(others)[:12]}>"]
        except Exception as e:  # noqa: BLE001 — diagnostic only
            return [f"<unavailable: {type(e).__name__}>"]


    def bench(model, feed):
        rq = core.compile_model(model, cli.device).create_infer_request()
        for _ in range(5):
            rq.infer(feed)
        ts = []
        for _ in range(cli.iters):
            t0 = time.perf_counter()
            rq.infer(feed)
            ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts), sorted(ts)[int(len(ts) * 0.95) - 1], rq


    batches = [int(b) for b in cli.batches.split(",")]
    for batch in batches:
        print(f"\n== batch {batch} ==")
        total_single = 0.0
        for name, o, i in PROJ:
            xv = rng.standard_normal((batch, i)).astype(np.float32)
            med, p95, _rq = bench(single_model(name, o, i, batch), {"x": xv})
            total_single += med
            print(f"  {name:6s} [{o:6d}x{i:6d}]: median={med:8.3f}ms p95={p95:8.3f}ms")
        xv = rng.standard_normal((batch, H)).astype(np.float32)
        med, p95, rq = bench(fused_model(batch), {"x": xv})
        prec = compiled_precisions(rq.get_compiled_model()) if hasattr(rq, "get_compiled_model") else ["<n/a>"]
        imed, ip95, _ = bench(identity_model(batch, H), {"x": xv})
        print(f"  fused           : median={med:8.3f}ms p95={p95:8.3f}ms (sum singles {total_single:.3f}ms)")
        print(f"  transfer (ident): median={imed:8.3f}ms p95={ip95:8.3f}ms")
        print(f"  matmul exec precisions: {prec}")


if __name__ == "__main__":
    main()
