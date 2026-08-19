#!/usr/bin/env python3
"""Emit a synthetic Kimi-K2.6 tree that the NATIVE sparse-MoE engine
(`shell_backend = "rust_k26"`, i.e. `runner.rs` + `cascadia-int4-gemm`)
will actually load — the native analogue of the `m2_tiny` OV fixture.

WHY "--tiny" IS NOT TINY
------------------------
The native shell/layer0 kernels are compile-time-constant K2.6:
`cascadia-int4-gemm/src/shell.rs` pins HIDDEN=7168, Q_LORA_RANK=1536,
KV_LORA_RANK=512, NUM_HEADS=64, QK_HEAD_DIM=192, V_HEAD_DIM=128,
INTERMEDIATE_SHARED=2048, INTERMEDIATE_DENSE=18432, N_ROUTED_EXPERTS=384,
TOPK=8, and `cascadia-int4-gemm/src/lib.rs` pins the expert INTERMEDIATE=2048.
`shell_int4.rs::quantize_int4_group` opens with

    assert_eq!(weight_bf16.len(), n_rows * k_cols * 2);

so every weight MUST be exactly the K2.6 shape. The only knobs that shrink
anything are vocab_size (embed rows), the NUMBER of MoE layers, and the
number of experts materialised per layer. A one-dense-plus-one-MoE tree is
~1.2 GiB before experts. Do not commit the output; generate it on demand.

  layer 0 (dense) : 949.0 MiB
  1 MoE shell     : 282.2 MiB
  1 expert        :  23.6 MiB
  embed (vocab 256):  3.5 MiB

FORCED ROUTING
--------------
`shell_int4.rs` picks top-8 of 384 by `sigmoid(router @ x) + e_score_correction_bias`.
We write bias = +1e3 for experts [0, top_k) and -1e3 elsewhere, so routing is
pinned to those ids regardless of the hidden state and only `--experts` experts
need to exist on disk. Routing stays deterministic and shape-valid; it is not
numerically meaningful (nothing here is — see the module docstring).

Usage:
    python tools/export_kimi_k26.py --tiny --out /tmp/k26_tiny
    python tools/export_kimi_k26.py --tiny --out /tmp/k26_mid --no-layer0
"""

import argparse
import json
import os
import sys

import numpy as np

# --- Compile-time constants mirrored from cascadia-int4-gemm ----------------
# shell.rs
HIDDEN = 7168
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
NUM_HEADS = 64
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
V_HEAD_DIM = 128
INTERMEDIATE_SHARED = 2048
INTERMEDIATE_DENSE = 18432
N_ROUTED_EXPERTS = 384
TOPK = 8
# lib.rs (routed experts)
EXPERT_INTERMEDIATE = 2048
GROUP_SIZE = 32

PREFIX = "language_model.model"


def bf16_bytes(rng, shape, scale=0.02):
    """Random bf16 tensor as raw little-endian bytes.

    Truncate f32 -> bf16 by dropping the low 16 bits. Non-zero-mean is
    irrelevant; what matters is that no 32-wide group is all-zero (that would
    push quantize_int4_group onto its 1e-10 degenerate-scale branch).
    """
    n = int(np.prod(shape))
    x = (rng.standard_normal(n, dtype=np.float32) * scale).astype(np.float32)
    return (x.view(np.uint32) >> 16).astype("<u2").tobytes()


def f32_bytes(arr):
    return np.asarray(arr, dtype="<f4").tobytes()


class Writer:
    """Streaming safetensors writer.

    Offsets come from the declared shapes, so the header can be written before
    any payload exists and each tensor's bytes are generated one at a time.
    Nothing bigger than one tensor is ever resident.
    """

    DTYPE_BYTES = {"BF16": 2, "F32": 4, "I32": 4}

    def __init__(self):
        self.entries = []  # (name, dtype, shape, producer)
        self.offset = 0
        self.meta = {}

    def add(self, name, dtype, shape, producer):
        nbytes = int(np.prod(shape)) * self.DTYPE_BYTES[dtype]
        self.meta[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [self.offset, self.offset + nbytes],
        }
        self.entries.append((name, nbytes, producer))
        self.offset += nbytes

    def write(self, path):
        header = json.dumps(self.meta, separators=(",", ":")).encode("utf-8")
        pad = (-len(header)) % 8
        header += b" " * pad
        with open(path, "wb") as f:
            f.write(len(header).to_bytes(8, "little"))
            f.write(header)
            for name, nbytes, producer in self.entries:
                blob = producer()
                assert len(blob) == nbytes, f"{name}: {len(blob)} != {nbytes}"
                f.write(blob)
        return self.offset


def add_attention(w, rng, layer):
    """The 8 tensors shared by layer 0 and every MoE shell (MLA block)."""
    base = f"{PREFIX}.layers.{layer}"
    add = lambda n, shape: w.add(  # noqa: E731
        f"{base}.{n}", "BF16", shape, lambda s=shape: bf16_bytes(rng, s)
    )
    add("input_layernorm.weight", (HIDDEN,))
    add("self_attn.q_a_proj.weight", (Q_LORA_RANK, HIDDEN))
    add("self_attn.q_a_layernorm.weight", (Q_LORA_RANK,))
    add("self_attn.q_b_proj.weight", (NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK))
    add("self_attn.kv_a_proj_with_mqa.weight", (KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN))
    add("self_attn.kv_a_layernorm.weight", (KV_LORA_RANK,))
    add(
        "self_attn.kv_b_proj.weight",
        (NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK),
    )
    add("self_attn.o_proj.weight", (HIDDEN, NUM_HEADS * V_HEAD_DIM))
    add("post_attention_layernorm.weight", (HIDDEN,))


def add_dense_layer0(w, rng):
    add_attention(w, rng, 0)
    base = f"{PREFIX}.layers.0.mlp"
    for n, shape in [
        ("gate_proj.weight", (INTERMEDIATE_DENSE, HIDDEN)),
        ("up_proj.weight", (INTERMEDIATE_DENSE, HIDDEN)),
        ("down_proj.weight", (HIDDEN, INTERMEDIATE_DENSE)),
    ]:
        w.add(f"{base}.{n}", "BF16", shape, lambda s=shape: bf16_bytes(rng, s))


def add_shell(w, rng, layer, top_k):
    add_attention(w, rng, layer)
    base = f"{PREFIX}.layers.{layer}.mlp"
    w.add(
        f"{base}.gate.weight",
        "BF16",
        (N_ROUTED_EXPERTS, HIDDEN),
        lambda: bf16_bytes(rng, (N_ROUTED_EXPERTS, HIDDEN)),
    )
    # Forced routing: only experts [0, top_k) ever get dispatched.
    bias = np.full(N_ROUTED_EXPERTS, -1.0e3, dtype=np.float32)
    bias[:top_k] = 1.0e3
    w.add(
        f"{base}.gate.e_score_correction_bias",
        "F32",
        (N_ROUTED_EXPERTS,),
        lambda b=bias: f32_bytes(b),
    )
    for n, shape in [
        ("shared_experts.gate_proj.weight", (INTERMEDIATE_SHARED, HIDDEN)),
        ("shared_experts.up_proj.weight", (INTERMEDIATE_SHARED, HIDDEN)),
        ("shared_experts.down_proj.weight", (HIDDEN, INTERMEDIATE_SHARED)),
    ]:
        w.add(f"{base}.{n}", "BF16", shape, lambda s=shape: bf16_bytes(rng, s))


def packed_int4_bytes(rng, n_rows, k_cols):
    """compressed-tensors `weight_packed`: one byte = two columns.

    Low nibble = even column, high nibble = odd column, each stored as the
    unsigned (q + 8) the AVX-512 dequant kernel expects — matching what
    `shell_int4.rs::quantize_int4_group` emits. Declared as I32 [rows, k/8]
    because that is what the real checkpoint uses; the loader only ever looks
    at the byte range, never the declared dtype.
    """
    return rng.integers(0, 256, size=n_rows * k_cols // 2, dtype=np.uint8).tobytes()


def add_expert(w, rng, layer, eid):
    base = f"{PREFIX}.layers.{layer}.mlp.experts.{eid}"
    for proj, n_rows, k_cols in [
        ("gate_proj", EXPERT_INTERMEDIATE, HIDDEN),
        ("up_proj", EXPERT_INTERMEDIATE, HIDDEN),
        ("down_proj", HIDDEN, EXPERT_INTERMEDIATE),
    ]:
        w.add(
            f"{base}.{proj}.weight_packed",
            "I32",
            (n_rows, k_cols // 8),
            lambda r=n_rows, k=k_cols: packed_int4_bytes(rng, r, k),
        )
        w.add(
            f"{base}.{proj}.weight_scale",
            "BF16",
            (n_rows, k_cols // GROUP_SIZE),
            lambda r=n_rows, k=k_cols: bf16_bytes(rng, (r, k // GROUP_SIZE), scale=0.01),
        )


def write_tokenizer(path, vocab_size):
    """Minimal WordLevel tokenizer.json — the engine needs rank 0 to have one."""
    vocab = {"[UNK]": 0}
    for i in range(1, vocab_size):
        vocab[f"t{i}"] = i
    doc = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    }
    with open(path, "w") as f:
        json.dump(doc, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiny", action="store_true", help="synthetic random-init tree (the only mode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--moe-layers", type=int, default=1, help="MoE shells, ids 1..N")
    ap.add_argument("--experts", type=int, default=TOPK,
                    help=f"experts materialised per MoE layer (>= {TOPK} to run a forward)")
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--no-layer0", action="store_true",
                    help="omit layer 0 + embed_tokens (middle-rank tree: is_first=false)")
    ap.add_argument("--no-experts", action="store_true",
                    help="omit expert weights (load-only tree; experts are lazy)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.tiny:
        ap.error("only --tiny is implemented; a real K2.6 export is export_shards.py's job")
    if args.experts > N_ROUTED_EXPERTS:
        ap.error(f"--experts must be <= {N_ROUTED_EXPERTS}")

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    w = Writer()
    if not args.no_layer0:
        w.add(f"{PREFIX}.embed_tokens.weight", "BF16", (args.vocab, HIDDEN),
              lambda: bf16_bytes(rng, (args.vocab, HIDDEN)))
        add_dense_layer0(w, rng)
    moe_ids = list(range(1, args.moe_layers + 1))
    for lid in moe_ids:
        add_shell(w, rng, lid, TOPK)
        if not args.no_experts:
            for eid in range(args.experts):
                add_expert(w, rng, lid, eid)

    shard = "model-00001-of-00001.safetensors"
    print(f"writing {shard} ({w.offset / 2**20:.1f} MiB, {len(w.entries)} tensors)...",
          file=sys.stderr)
    w.write(os.path.join(args.out, shard))

    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": w.offset},
                   "weight_map": {name: shard for name, _, _ in w.entries}}, f)

    manifest = {
        # No "shell_backend" key: Manifest defaults it to "rust_k26", the
        # native branch. Emitting "ov_ir" here would divert to OvMoeEngine.
        "arch": "kimi_k2.6",
        "num_layers": 1 + args.moe_layers,
        "dense_layers": [0],
        "num_experts": N_ROUTED_EXPERTS,
        "top_k": TOPK,
        "hidden_size": HIDDEN,
        "num_kv_heads": NUM_HEADS,
        "qk_head_dim": QK_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        "vocab_size": args.vocab,
        "eos_token_ids": [args.vocab - 1],
        "experts_format": "safetensors_bin",
        "expert_intermediate": EXPERT_INTERMEDIATE,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    write_tokenizer(os.path.join(args.out, "tokenizer.json"), args.vocab)

    print(f"wrote {args.out}: {w.offset / 2**20:.1f} MiB, "
          f"{len(moe_ids)} MoE shell(s), "
          f"{0 if args.no_experts else args.experts} expert(s)/layer, "
          f"layer0={'no' if args.no_layer0 else 'yes'}", file=sys.stderr)
    print("NOTE: no head/openvino_model.xml — the native head is OV-IR only, so this "
          "tree can only be loaded by a non-last stage (see the test).", file=sys.stderr)


if __name__ == "__main__":
    main()
