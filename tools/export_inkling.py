#!/usr/bin/env python3
"""Inkling (`inkling`, Thinking Machines MoE) exporter -> the sparse-moe engine's on-disk layout.

Modes:
  --validate CONFIG.json          hard-fail config contract (PORT_SPEC §1); prints the derived manifest
  --tiny OUT                      synthetic tiny model (transformers InklingForCausalLM, seed 7) exported
                                  through the SAME code path as a real checkpoint + OUT/reference.json
  --model DIR --out OUT           convert a local thinkingmachines/Inkling{,-Small} checkpoint
      [--layers a-b]              only layers a..b inclusive (embed/head still exported unless --shards-only)
      [--shards-only]             only the int4 expert/dense bins: skip embed, head, shells, manifest
      [--skip-missing-shards]     streaming pass: export everything the PRESENT shards allow, skip the
                                  rest, exit 0 with an `INKLING_EXPORT ...` summary line; re-run as
                                  more `model-000NN-of-00109.safetensors` files arrive
      [--delete-consumed-shards]  delete a source shard once EVERY tensor it holds is exported + fsynced
                                  (tracked per shard through model.safetensors.index.json)
      [--workers N]               expert-quantization threads (default min(8, cpus))
  --layers-done-check --out OUT [--model DIR]
                                  assert embed, head and every layer are complete; print the manifest;
                                  exit 1 (listing what is missing) otherwise

On-disk layout (PORT_SPEC §1-§2; read by crates/cascadia-engine-sparse-moe/src/inkling/loader.rs):
  <out>/manifest.json                          arch "inkling"
  <out>/embed.safetensors                      embed.weight bf16 [V,H], embed_norm.weight f32 [H]
  <out>/head.safetensors                       unembed.weight bf16 [V,H], norm.weight f32 [H]
  <out>/shells/layer_NN.safetensors            checkpoint suffix names (attn.wq_du.weight, ...):
                                               projections bf16; norms, convs (stored [C,4]),
                                               rel_logits_proj.proj, router (weight/bias/global_scale) f32
  <out>/experts/layer_NN/expert_EEE.bin        routed expert: int4 group-32 gate, up, down
                                               (gate/up DE-INTERLEAVED from w13 rows 0::2 / 1::2)
  <out>/experts/layer_NN/expert_sharedS.bin    one bin per shared expert (each has its own gamma)
  <out>/experts/layer_NN/dense.bin             dense-layer MLP (w13_dn / w2_md)
  <out>/source_config.json                     copy of the source config.json (for --layers-done-check)

Every output is written to `<name>.tmp`, fsynced, then renamed, so a kill mid-tensor never leaves a
truncated file that looks done. Re-runs skip finished work by file presence + size (and per-layer
`.layer_NN.done` markers) in seconds. Per-tensor timing is printed for every unit written.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_INT4_GROUP = 32
LAYER_PREFIX = "model.llm.layers."
DROP_PREFIXES = ("model.mtp.", "model.audio.", "model.visual.")
# <out>/embed.safetensors, <out>/head.safetensors: key -> (checkpoint name, stored dtype)
EMBED_TENSORS = {
    "embed.weight": ("model.llm.embed.weight", "bf16"),
    "embed_norm.weight": ("model.llm.embed_norm.weight", "f32"),
}
HEAD_TENSORS = {
    "unembed.weight": ("model.llm.unembed.weight", "bf16"),
    "norm.weight": ("model.llm.norm.weight", "f32"),
}
SIDECARS = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
            "generation_config.json", "special_tokens_map.json", "tokenizer.model")
# Contract flags (PORT_SPEC §1): the Rust shell implements exactly this variant. A present key with
# another value fails loudly; an absent key is assumed (transformers' modeling code hardcodes these).
CONTRACT_FLAGS = {
    "gate_activation": "sigmoid",
    "norm_after_topk": True,
    "use_global_scale": True,
    "use_gate_bias": True,
    "use_sconv": True,
    "use_embed_norm": True,
    "shared_expert_sink": True,
    "q_bias": False,
    "o_bias": False,
    "final_logit_softcapping": None,
}

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# int4_bin packing — cascadia-int4-gemm group-32 layout. torch port of
# export_glm5._pack_int4_grouped (byte-identical; tools/tests/test_inkling_export.py asserts it).
# --------------------------------------------------------------------------
def pack_int4(w):
    """[out, in] -> (packed u8 bytes [out, in/2], scales bf16-LE bytes [out, in/32]).
    Symmetric: s = max|w|/7 per group of 32, q = clip(round(w/s), -8, 7), nibble = q+8, low
    nibble first. Vectorized (torch intra-op threads), no Python loop over rows."""
    import torch

    w = w.to(torch.float32).contiguous()
    out, inn = w.shape
    g = _INT4_GROUP
    assert inn % g == 0, f"in_dim {inn} not divisible by int4 group {g}"
    wg = w.view(out, inn // g, g)
    max_abs = wg.abs().amax(dim=2)
    s = torch.where(max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs))
    q = torch.clamp(torch.round(wg / s[:, :, None]), -8, 7).to(torch.int16)
    nib = (q + 8).to(torch.uint8).view(out, inn)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    bf = s.to(torch.bfloat16).view(torch.int16).contiguous()  # RNE f32 -> bf16 (LE host)
    return packed.numpy().tobytes(), bf.numpy().tobytes()


def expert_bin_chunks(gate, up, down):
    """SwiGLU FFN bin = gate, up, down sections (packed nibbles then bf16 scales each).
    Matches MmapExpert::open's `2*section(inter,dim) + section(dim,inter)`."""
    chunks = []
    for w in (gate, up, down):
        p, s = pack_int4(w)
        chunks += [p, s]
    return chunks


def int4_bin_bytes(hidden: int, inter: int) -> int:
    def sec(o, i):
        return o * i // 2 + o * (i // _INT4_GROUP) * 2
    return 2 * sec(inter, hidden) + sec(hidden, inter)


# --------------------------------------------------------------------------
# atomic, fsynced writes
# --------------------------------------------------------------------------
def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # some filesystems refuse dir fsync; the file itself is already synced


def atomic_write_bytes(path: Path, chunks) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "wb") as f:
        for c in chunks:
            f.write(c)
            n += len(c)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return n


def atomic_save_safetensors(tensors: dict, path: Path) -> int:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file(tensors, str(tmp))
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return path.stat().st_size


# --------------------------------------------------------------------------
# Config contract — hard-fail on anything the Rust shell does not implement.
# --------------------------------------------------------------------------
class ConfigError(SystemExit):
    pass


def _require(cond, msg):
    if not cond:
        raise ConfigError(f"[export_inkling] config contract violated: {msg}")


def load_and_validate_config(src, strict: bool = False) -> dict:
    """config.json (path or dict; top-level `inkling_mm_model` with `text_config`, or a bare text
    config) -> the manifest dict (PORT_SPEC §1). Fails loudly on any contract surprise."""
    c = json.loads(Path(src).read_text()) if isinstance(src, (str, Path)) else dict(src)
    tc = c["text_config"] if isinstance(c.get("text_config"), dict) else c
    mt = tc.get("model_type", c.get("model_type"))
    _require(mt in (None, "inkling_mm_model", "inkling_text"),
             f"model_type must be 'inkling_mm_model' / 'inkling_text', got {mt!r}")

    assumed = []
    for k, want in CONTRACT_FLAGS.items():
        if k in tc:
            _require(tc[k] == want, f"text_config.{k} must be {want!r}, got {tc[k]!r}")
        else:
            assumed.append(k)
    if assumed:
        _require(not strict, f"--strict: contract keys absent from text_config: {assumed}")
        log("[validate] WARNING: contract keys absent, assuming transformers' semantics: "
            + ", ".join(f"{k}={CONTRACT_FLAGS[k]!r}" for k in assumed))

    def g(k):
        _require(k in tc, f"text_config missing key '{k}'")
        return tc[k]

    hidden, num_layers, vocab = int(g("hidden_size")), int(g("num_hidden_layers")), int(g("vocab_size"))
    unpadded = int(tc.get("unpadded_vocab_size") or vocab)
    _require(0 < unpadded <= vocab, f"unpadded_vocab_size {unpadded} not in (0, vocab_size={vocab}]")
    heads, kv = int(g("num_attention_heads")), int(g("num_key_value_heads"))
    head_dim = int(tc.get("head_dim") or hidden // heads)
    swa_heads = int(tc.get("swa_num_attention_heads") or heads)
    swa_kv = int(tc.get("swa_num_key_value_heads") or kv)
    swa_hd = int(tc.get("swa_head_dim") or head_dim)
    _require(heads % kv == 0 and swa_heads % swa_kv == 0, "attention heads must be a multiple of kv heads")
    d_rel, rel_extent = int(g("d_rel")), int(g("rel_extent"))
    window = tc.get("sliding_window_size", tc.get("sliding_window"))
    _require(window, "sliding_window_size missing")
    window = int(window)

    if tc.get("layer_types"):
        lt = list(tc["layer_types"])
        _require(len(lt) == num_layers and all(t in ("hybrid", "hybrid_sliding") for t in lt),
                 f"layer_types must be {num_layers} x 'hybrid'|'hybrid_sliding', got {lt}")
        layer_types = ["sliding" if t == "hybrid_sliding" else "global" for t in lt]
    else:
        if tc.get("local_layer_ids") is not None:
            local = {int(i) for i in tc["local_layer_ids"]}
        else:  # transformers default: every layer whose (i+1) is not a multiple of 6 slides
            local = {i for i in range(num_layers) if (i + 1) % 6}
        _require(all(0 <= i < num_layers for i in local), f"local_layer_ids out of range: {sorted(local)}")
        layer_types = ["sliding" if i in local else "global" for i in range(num_layers)]

    if tc.get("mlp_layer_types"):
        ml = list(tc["mlp_layer_types"])
        _require(len(ml) == num_layers and all(t in ("dense", "sparse") for t in ml),
                 f"mlp_layer_types must be {num_layers} x 'dense'|'sparse', got {ml}")
        dense_layers = [i for i, t in enumerate(ml) if t == "dense"]
    else:
        dense_layers = list(range(int(tc.get("dense_mlp_idx") or 0)))
    _require(len(dense_layers) <= num_layers, "more dense layers than layers")

    moe_inter = tc.get("moe_intermediate_size", tc.get("intermediate_size"))
    _require(moe_inter, "moe_intermediate_size / intermediate_size missing")
    moe_inter = int(moe_inter)
    dense_inter = int(tc.get("dense_intermediate_size", tc.get("intermediate_size")) or 0) if dense_layers else 0
    _require(not dense_layers or dense_inter > 0, "dense layers present but no dense_intermediate_size")
    n_experts, top_k = int(g("n_routed_experts")), int(g("num_experts_per_tok"))
    n_shared = int(tc.get("n_shared_experts", 2))
    _require(1 <= top_k <= n_experts, f"num_experts_per_tok {top_k} not in [1, {n_experts}]")
    _require(n_shared >= 0, "n_shared_experts < 0")
    n_floor = tc.get("log_scaling_n_floor")
    act = tc.get("hidden_act", "silu")
    _require(act == "silu", f"hidden_act must be 'silu', got {act!r}")
    kernel = int(tc.get("sconv_kernel_size", tc.get("conv_kernel_size", 4)))
    eos = tc.get("eos_token_id", c.get("eos_token_id"))
    eos = [] if eos is None else ([int(eos)] if isinstance(eos, int) else [int(e) for e in eos])

    _require(hidden % _INT4_GROUP == 0, f"hidden_size {hidden} not divisible by {_INT4_GROUP}")
    _require(moe_inter % _INT4_GROUP == 0, f"moe intermediate {moe_inter} not divisible by {_INT4_GROUP}")
    _require(not dense_layers or dense_inter % _INT4_GROUP == 0,
             f"dense intermediate {dense_inter} not divisible by {_INT4_GROUP}")
    _require(num_layers >= 1 and d_rel >= 1 and rel_extent >= 1 and window >= 1 and kernel >= 1, "bad dims")

    return {
        "arch": "inkling",
        "num_layers": num_layers,
        "hidden_size": hidden,
        "vocab_size": vocab,
        "unpadded_vocab_size": unpadded,
        "num_attention_heads": heads,
        "num_kv_heads": kv,
        "head_dim": head_dim,
        "swa_num_attention_heads": swa_heads,
        "swa_num_kv_heads": swa_kv,
        "swa_head_dim": swa_hd,
        "d_rel": d_rel,
        "rel_extent": rel_extent,
        "sliding_window": window,
        "layer_types": layer_types,
        "dense_layers": dense_layers,
        "dense_intermediate": dense_inter,
        "moe_intermediate": moe_inter,
        "num_experts": n_experts,
        "top_k": top_k,
        "n_shared_experts": n_shared,
        "route_scale": float(tc.get("route_scale", 8.0)),
        "rms_norm_eps": float(tc.get("rms_norm_eps", 1e-6)),
        "log_scaling_n_floor": int(n_floor) if n_floor is not None else None,
        "log_scaling_alpha": float(tc.get("log_scaling_alpha", 0.1)),
        "logits_mup_width_multiplier": float(tc.get("logits_mup_width_multiplier", 24.0)),
        "conv_kernel_size": kernel,
        "eos_token_ids": eos,
        "hidden_act": act,
        "experts_format": "int4_bin",
        "shell_backend": "rust_inkling",
        "has_mtp": False,
    }


def write_manifest(man: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(man, indent=2))
    os.replace(tmp, out / "manifest.json")
    log(f"[manifest] {out / 'manifest.json'}")


# --------------------------------------------------------------------------
# Per-layer tensor plan (checkpoint names + expected shapes; PORT_SPEC §2)
# --------------------------------------------------------------------------
def layer_dims(man: dict, li: int):
    """(heads, kv_heads, head_dim, rel_extent) for layer li (sliding layers use the swa dims and
    rel_extent == sliding_window)."""
    if man["layer_types"][li] == "sliding":
        return man["swa_num_attention_heads"], man["swa_num_kv_heads"], man["swa_head_dim"], man["sliding_window"]
    return man["num_attention_heads"], man["num_kv_heads"], man["head_dim"], man["rel_extent"]


def shell_spec(man: dict, li: int):
    """[(checkpoint suffix, stored role bf16|f32|conv, expected source shape)] for one layer."""
    H, K, d_rel = man["hidden_size"], man["conv_kernel_size"], man["d_rel"]
    heads, kv, hd, extent = layer_dims(man, li)
    spec = [
        ("attn_norm.weight", "f32", (H,)),
        ("mlp_norm.weight", "f32", (H,)),
        ("attn.wq_du.weight", "bf16", (heads * hd, H)),
        ("attn.wk_dv.weight", "bf16", (kv * hd, H)),
        ("attn.wv_dv.weight", "bf16", (kv * hd, H)),
        ("attn.wr_du.weight", "bf16", (heads * d_rel, H)),
        ("attn.wo_ud.weight", "bf16", (H, heads * hd)),
        ("attn.q_norm.weight", "f32", (hd,)),
        ("attn.k_norm.weight", "f32", (hd,)),
        ("attn.k_sconv.weight", "conv", (kv * hd, 1, K)),
        ("attn.v_sconv.weight", "conv", (kv * hd, 1, K)),
        ("attn.rel_logits_proj.proj", "f32", (d_rel, extent)),
        ("attn_sconv.weight", "conv", (H, 1, K)),
        ("mlp_sconv.weight", "conv", (H, 1, K)),
    ]
    if li in man["dense_layers"]:
        spec.append(("mlp.global_scale", "f32", (1,)))
    else:
        E, S = man["num_experts"], man["n_shared_experts"]
        spec += [
            ("mlp.gate.weight", "f32", (E + S, H)),
            ("mlp.gate.bias", "f32", (E,)),
            ("mlp.gate.global_scale", "f32", (1,)),
        ]
    return spec


class Unit:
    """One output file: embed | head | shell(layer) | expert(layer, e) | shared(layer, s) | dense(layer)."""
    __slots__ = ("kind", "layer", "idx", "needs", "output", "size")

    def __init__(self, kind, layer, idx, needs, output, size=None):
        self.kind, self.layer, self.idx, self.needs, self.output, self.size = kind, layer, idx, needs, output, size

    @property
    def label(self) -> str:
        if self.kind in ("embed", "head"):
            return self.kind
        if self.kind == "expert":
            tag = f"expert {self.idx:03d}"
        elif self.kind == "shared":
            tag = f"shared {self.idx}"
        else:
            tag = self.kind
        return f"layer {self.layer:02d}][{tag}"


def build_plan(man: dict, out: Path):
    """Every unit of a complete export (independent of --layers/--shards-only, which only choose
    what to RUN; shard consumption is judged against the full plan)."""
    H, I, Id = man["hidden_size"], man["moe_intermediate"], man["dense_intermediate"]
    units = [
        Unit("embed", None, None, [n for n, _ in EMBED_TENSORS.values()], out / "embed.safetensors"),
        Unit("head", None, None, [n for n, _ in HEAD_TENSORS.values()], out / "head.safetensors"),
    ]
    per_layer = defaultdict(list)
    for li in range(man["num_layers"]):
        p = f"{LAYER_PREFIX}{li}."
        edir = out / "experts" / f"layer_{li:02d}"
        u = Unit("shell", li, None, [p + suf for suf, _, _ in shell_spec(man, li)],
                 out / "shells" / f"layer_{li:02d}.safetensors")
        per_layer[li].append(u)
        if li in man["dense_layers"]:
            per_layer[li].append(Unit("dense", li, None, [p + "mlp.w13_dn.weight", p + "mlp.w2_md.weight"],
                                      edir / "dense.bin", int4_bin_bytes(H, Id)))
        else:
            for e in range(man["num_experts"]):
                per_layer[li].append(Unit("expert", li, e,
                                          [p + "mlp.experts.w13_weight", p + "mlp.experts.w2_weight"],
                                          edir / f"expert_{e:03d}.bin", int4_bin_bytes(H, I)))
            for s in range(man["n_shared_experts"]):
                per_layer[li].append(Unit("shared", li, s,
                                          [p + "mlp.shared_experts.shared_w13_weight",
                                           p + "mlp.shared_experts.shared_w2_weight"],
                                          edir / f"expert_shared{s}.bin", int4_bin_bytes(H, I)))
        units += per_layer[li]
    return units, per_layer


def layer_marker(out: Path, li: int) -> Path:
    return out / f".layer_{li:02d}.done"


def unit_output_done(u: Unit) -> bool:
    try:
        st = u.output.stat()
    except FileNotFoundError:
        return False
    return u.size is None or st.st_size == u.size


def completeness(man: dict, out: Path):
    """(complete, missing descriptions, layers_done, embed_done, head_done) from the files on disk.
    Verifies every output's presence + size; `.layer_NN.done` markers are deliberately NOT trusted
    here (they are the exporter's fast path; this is the audit)."""
    units, per_layer = build_plan(man, out)
    missing, layers_done = [], 0
    embed_done = unit_output_done(units[0])
    head_done = unit_output_done(units[1])
    if not embed_done:
        missing.append("embed.safetensors")
    if not head_done:
        missing.append("head.safetensors")
    for li in range(man["num_layers"]):
        bad = [u for u in per_layer[li] if not unit_output_done(u)]
        if not bad:
            layers_done += 1
        else:
            kinds = defaultdict(int)
            for u in bad:
                kinds[u.kind] += 1
            missing.append(f"layer {li:02d}: missing " + ", ".join(f"{n} {k}" for k, n in kinds.items()))
    return not missing, missing, layers_done, embed_done, head_done


# --------------------------------------------------------------------------
# Checkpoint sources
# --------------------------------------------------------------------------
class DictSource:
    """In-memory checkpoint (the --tiny path). Same interface as ShardSource."""

    def __init__(self, tensors: dict):
        self.t = dict(tensors)
        self.shards = {}
        self.dir = None

    def names(self):
        return list(self.t)

    def has(self, n):
        return n in self.t

    def shard_of(self, n):
        return None

    def present(self, shard):
        return True

    def available(self, n):
        return n in self.t

    def shape(self, n):
        return tuple(self.t[n].shape)

    def get(self, n):
        return self.t[n]

    def get_row(self, n, i):
        return self.t[n][i]

    def close_handles(self):
        pass


class ShardSource:
    """Streamed reads over a local HF checkpoint via `model.safetensors.index.json`. Never loads a
    shard: `get` reads one tensor, `get_row` one expert's slice (`safe_open(...).get_slice(name)[e]`,
    a [2I, H] read out of a 9.7 GB w13). One safe_open handle per (thread, shard)."""

    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.dir = Path(model_dir)
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
        else:
            single = self.dir / "model.safetensors"
            if not single.exists():
                raise SystemExit(f"[export_inkling] {self.dir}: no model.safetensors.index.json / model.safetensors")
            with safe_open(str(single), framework="pt") as f:
                wm = {k: "model.safetensors" for k in f.keys()}
        self.wm = wm
        self.shards = defaultdict(list)
        for n, s in wm.items():
            self.shards[s].append(n)
        self._tls = threading.local()
        self._caches = []
        self._lock = threading.Lock()

    def names(self):
        return list(self.wm)

    def has(self, n):
        return n in self.wm

    def shard_of(self, n):
        return self.wm.get(n)

    def present(self, shard) -> bool:
        return (self.dir / shard).is_file()

    def available(self, n) -> bool:
        s = self.wm.get(n)
        return s is not None and self.present(s)

    def _handle(self, n):
        if n not in self.wm:
            raise SystemExit(f"[export_inkling] tensor '{n}' not in the checkpoint index — "
                             "checkpoint layout differs from PORT_SPEC §2")
        shard = self.wm[n]
        d = getattr(self._tls, "h", None)
        if d is None:
            d = {}
            self._tls.h = d
            with self._lock:
                self._caches.append(d)
        h = d.get(shard)
        if h is None:
            h = self._safe_open(str(self.dir / shard), framework="pt")
            d[shard] = h
        return h

    def shape(self, n):
        return tuple(self._handle(n).get_slice(n).get_shape())

    def get(self, n):
        return self._handle(n).get_tensor(n)

    def get_row(self, n, i):
        return self._handle(n).get_slice(n)[i]

    def close_handles(self):
        with self._lock:
            for d in self._caches:
                d.clear()


# --------------------------------------------------------------------------
# Disk pre-flight
# --------------------------------------------------------------------------
def estimate_export_bytes(man: dict) -> int:
    H, K, d_rel, V = man["hidden_size"], man["conv_kernel_size"], man["d_rel"], man["vocab_size"]
    E, S, I, Id = man["num_experts"], man["n_shared_experts"], man["moe_intermediate"], man["dense_intermediate"]
    total = 2 * 2 * V * H + 4 * 2 * H
    for li in range(man["num_layers"]):
        heads, kv, hd, extent = layer_dims(man, li)
        total += 2 * (heads * hd * H + 2 * kv * hd * H + heads * d_rel * H + H * heads * hd)
        total += 4 * (2 * H + 2 * hd + 2 * kv * hd * K + d_rel * extent + 2 * H * K)
        if li in man["dense_layers"]:
            total += int4_bin_bytes(H, Id) + 4
        else:
            total += (E + S) * int4_bin_bytes(H, I) + 4 * ((E + S) * H + E + 1)
    return total


def check_space(out: Path, man: dict, first_pass: bool) -> None:
    """Refuse to START an export the disk cannot hold (INKLING_SKIP_SPACE_CHECK=1 overrides). On a
    resumed / streaming pass the free space is a moving target (shards arrive and get deleted), so
    only the numbers are printed."""
    out.mkdir(parents=True, exist_ok=True)
    est = estimate_export_bytes(man)
    free = shutil.disk_usage(out).free
    log(f"[check_space] estimated export ~{est / 1e9:.1f} GB, free ~{free / 1e9:.1f} GB"
        + ("" if first_pass else " (resumed pass: not enforced)"))
    if first_pass and free < est * 1.05 and os.environ.get("INKLING_SKIP_SPACE_CHECK") != "1":
        raise SystemExit(f"[export_inkling] insufficient disk: need ~{est / 1e9:.1f} GB (+5% margin), "
                         f"have ~{free / 1e9:.1f} GB. Set INKLING_SKIP_SPACE_CHECK=1 to override.")


# --------------------------------------------------------------------------
# The exporter
# --------------------------------------------------------------------------
def _shape_check(name, got, want):
    if tuple(got) != tuple(want):
        raise SystemExit(f"[export_inkling] {name}: shape {tuple(got)} != expected {tuple(want)} "
                         "— config.json and checkpoint disagree (or PORT_SPEC §2 is wrong for this checkpoint)")


class Exporter:
    def __init__(self, src, man: dict, out: Path, *, layers=None, shards_only=False, skip_missing=False,
                 delete_consumed=False, workers=4):
        self.src, self.man, self.out = src, man, Path(out)
        self.layers = layers
        self.shards_only, self.skip_missing, self.delete_consumed = shards_only, skip_missing, delete_consumed
        self.workers = max(1, workers)
        self.units, self.per_layer = build_plan(man, self.out)
        self._done: dict[int, bool] = {}
        self._lock = threading.Lock()
        self.pending_shards: set[str] = set()
        self.deleted_shards: list[str] = []
        self.counts = defaultdict(int)
        self.layer_stats = None

    # ---- done / runnable bookkeeping
    def unit_done(self, u: Unit) -> bool:
        k = id(u)
        v = self._done.get(k)
        if v is None:
            if u.layer is not None and layer_marker(self.out, u.layer).exists():
                v = True
            else:
                v = unit_output_done(u)
            self._done[k] = v
        return v

    def mark_done(self, u: Unit) -> None:
        with self._lock:
            self._done[id(u)] = True

    def runnable(self, u: Unit) -> bool:
        missing = sorted({self.src.shard_of(n) for n in u.needs if not self.src.available(n)} - {None})
        absent = [n for n in u.needs if not self.src.has(n)]
        if absent:
            raise SystemExit(f"[export_inkling] {u.label}: tensors missing from the checkpoint index: {absent} "
                             "— checkpoint layout differs from PORT_SPEC §2")
        if missing:
            if not self.skip_missing:
                raise SystemExit(f"[export_inkling] {u.label}: source shard(s) not on disk: {missing} "
                                 "(use --skip-missing-shards for a streaming pass)")
            with self._lock:
                self.pending_shards.update(missing)
            return False
        return True

    # ---- unit runners
    def _read_global(self, u: Unit):
        import torch

        spec = EMBED_TENSORS if u.kind == "embed" else HEAD_TENSORS
        V, H = self.man["vocab_size"], self.man["hidden_size"]
        tensors, nw = {}, 0
        for key, (name, role) in spec.items():
            t = self.src.get(name)
            _shape_check(name, t.shape, (V, H) if role == "bf16" else (H,))
            tensors[key] = (t.to(torch.bfloat16) if role == "bf16" else t.to(torch.float32)).contiguous()
            nw += t.numel()
        return tensors, nw

    def _read_shell(self, u: Unit):
        import torch

        tensors, nw = {}, 0
        for suffix, role, shape in shell_spec(self.man, u.layer):
            name = f"{LAYER_PREFIX}{u.layer}.{suffix}"
            t = self.src.get(name)
            _shape_check(name, t.shape, shape)
            if role == "bf16":
                t = t.to(torch.bfloat16)
            elif role == "f32":
                t = t.to(torch.float32)
            else:  # conv [C, 1, K] -> [C, K] f32
                t = t.to(torch.float32).reshape(shape[0], shape[2])
            tensors[suffix] = t.contiguous()
            nw += t.numel()
        return tensors, nw

    def _read_ffn(self, u: Unit):
        import torch

        H = self.man["hidden_size"]
        w13n, w2n = u.needs
        if u.kind == "dense":
            Id = self.man["dense_intermediate"]
            w13, w2 = self.src.get(w13n), self.src.get(w2n)
            _shape_check(w13n, w13.shape, (2 * Id, H))
            _shape_check(w2n, w2.shape, (H, Id))
        else:
            I = self.man["moe_intermediate"]
            n = self.man["num_experts"] if u.kind == "expert" else self.man["n_shared_experts"]
            _shape_check(w13n, self.src.shape(w13n), (n, 2 * I, H))
            _shape_check(w2n, self.src.shape(w2n), (n, H, I))
            w13, w2 = self.src.get_row(w13n, u.idx), self.src.get_row(w2n, u.idx)
        w13 = w13.to(torch.float32)
        # transformers Interleave(dim=1): gate = rows 0::2, up = rows 1::2
        return w13[0::2], w13[1::2], w2.to(torch.float32)

    def run_unit(self, u: Unit) -> str:
        if self.unit_done(u):
            return "skipped"
        if not self.runnable(u):
            return "pending"
        t0 = time.perf_counter()
        if u.kind in ("embed", "head"):
            tensors, nw = self._read_global(u)
            t1 = t2 = time.perf_counter()
            nbytes = atomic_save_safetensors(tensors, u.output)
        elif u.kind == "shell":
            tensors, nw = self._read_shell(u)
            t1 = t2 = time.perf_counter()
            nbytes = atomic_save_safetensors(tensors, u.output)
        else:
            gate, up, down = self._read_ffn(u)
            nw = gate.numel() + up.numel() + down.numel()
            t1 = time.perf_counter()
            chunks = expert_bin_chunks(gate, up, down)
            t2 = time.perf_counter()
            nbytes = atomic_write_bytes(u.output, chunks)
            if nbytes != u.size:
                raise SystemExit(f"[export_inkling] {u.output}: wrote {nbytes} bytes, expected {u.size}")
        t3 = time.perf_counter()
        self.mark_done(u)
        log(f"[{u.label}] read {t1 - t0:.3f}s quant {t2 - t1:.3f}s write {t3 - t2:.3f}s | "
            f"{nw / 1e6:.1f}M weights -> {nbytes / 1e6:.1f} MB")
        with self._lock:
            if self.layer_stats is not None:
                st = self.layer_stats
                st["n"] += 1
                st["read"] += t1 - t0
                st["quant"] += t2 - t1
                st["write"] += t3 - t2
                st["bytes"] += nbytes
        return "done"

    # ---- shard consumption
    def consumed_tensors(self) -> set:
        consumers = defaultdict(list)
        for u in self.units:
            for n in u.needs:
                consumers[n].append(u)
        return {n for n, us in consumers.items() if all(self.unit_done(u) for u in us)}

    def shard_consumed(self, shard: str, consumed: set) -> bool:
        return all(n.startswith(DROP_PREFIXES) or n in consumed for n in self.src.shards[shard])

    def sweep_shards(self) -> None:
        if not self.delete_consumed or not self.src.shards:
            return
        consumed = self.consumed_tensors()
        for shard in sorted(self.src.shards):
            if self.src.present(shard) and self.shard_consumed(shard, consumed):
                self.src.close_handles()
                p = self.src.dir / shard
                size = p.stat().st_size
                os.remove(p)
                self.deleted_shards.append(shard)
                log(f"[shard] deleted {shard} ({size / 1e9:.2f} GB): every tensor it holds is exported")

    # ---- driver
    def run(self) -> dict:
        man, out = self.man, self.out
        L = man["num_layers"]
        lo, hi = self.layers if self.layers else (0, L - 1)
        if not (0 <= lo <= hi < L):
            raise SystemExit(f"[export_inkling] --layers {lo}-{hi} outside 0..{L - 1}")
        known = {n for u in self.units for n in u.needs}
        unknown = [n for n in self.src.names() if n not in known and not n.startswith(DROP_PREFIXES)]
        if unknown:
            log(f"[export] WARNING: {len(unknown)} checkpoint tensors are neither exported nor dropped "
                f"(their shards are never deleted), e.g. {unknown[:5]}")

        def tally(status):
            self.counts[status] += 1

        if not self.shards_only:
            for u in self.units[:2]:
                tally(self.run_unit(u))
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for li in range(lo, hi + 1):
                lunits = self.per_layer[li]
                if layer_marker(out, li).exists():
                    log(f"[layer {li:02d}/{L}] skip (done)")
                    for _ in lunits:
                        tally("skipped")
                    continue
                self.layer_stats = {"n": 0, "read": 0.0, "quant": 0.0, "write": 0.0, "bytes": 0}
                t0 = time.perf_counter()
                todo = [u for u in lunits if not (self.shards_only and u.kind == "shell")]
                statuses = list(ex.map(self.run_unit, todo))
                for s in statuses:
                    tally(s)
                st = self.layer_stats
                self.layer_stats = None
                if all(self.unit_done(u) for u in lunits):
                    layer_marker(out, li).touch()
                    state = "complete"
                else:
                    state = "partial (" + ", ".join(f"{k}={statuses.count(k)}" for k in ("done", "skipped", "pending")) + ")"
                dt = time.perf_counter() - t0
                log(f"[layer {li:02d}/{L}] {state}: {st['n']} units written in {dt:.1f}s "
                    f"(read {st['read']:.1f}s quant {st['quant']:.1f}s write {st['write']:.1f}s; "
                    f"{st['bytes'] / 1e9:.2f} GB, {st['bytes'] / 1e9 / max(dt, 1e-9):.2f} GB/s)")
                self.sweep_shards()
        self.sweep_shards()

        complete, missing, layers_done, embed_done, head_done = completeness(man, out)
        n_consumed = 0
        if self.src.shards:
            consumed = self.consumed_tensors()
            n_consumed = sum(self.shard_consumed(s, consumed) for s in self.src.shards)
        summary = {
            "complete": complete, "layers_done": layers_done, "num_layers": L, "embed_done": embed_done,
            "head_done": head_done, "shards_consumed": n_consumed, "shards_deleted": list(self.deleted_shards),
            "pending_shards": sorted(self.pending_shards), "counts": dict(self.counts), "missing": missing,
        }
        if complete and not self.shards_only:
            write_manifest(man, out)
        elif not complete:
            log("[export] incomplete, manifest not written: " + "; ".join(missing[:6])
                + (" ..." if len(missing) > 6 else ""))
        log(f"INKLING_EXPORT layers_done={layers_done}/{L} shards_consumed={n_consumed} "
            f"pending_shards=[{','.join(summary['pending_shards'])}] embed_done={int(embed_done)} "
            f"head_done={int(head_done)} complete={int(complete)}")
        return summary


def _set_threads(workers: int) -> None:
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 1) // max(1, workers)))


def export_real(model_dir: Path, out: Path, *, layers=None, shards_only=False, skip_missing=False,
                delete_consumed=False, workers=4, strict=False) -> dict:
    model_dir, out = Path(model_dir), Path(out)
    man = load_and_validate_config(model_dir / "config.json", strict=strict)
    out.mkdir(parents=True, exist_ok=True)
    src_cfg = out / "source_config.json"
    if not src_cfg.exists():
        shutil.copy(model_dir / "config.json", src_cfg)
    first_pass = not (out / "embed.safetensors").exists() and not any(out.glob(".layer_*.done"))
    check_space(out, man, first_pass)
    _set_threads(workers)
    src = ShardSource(model_dir)
    ex = Exporter(src, man, out, layers=layers, shards_only=shards_only, skip_missing=skip_missing,
                  delete_consumed=delete_consumed, workers=workers)
    summary = ex.run()
    if summary["complete"] and not shards_only:
        for fn in SIDECARS:
            s = model_dir / fn
            if s.exists() and not (out / fn).exists():
                shutil.copy(s, out / fn)
                log(f"[sidecar] {fn}")
    return summary


def export_tiny(out: Path, workers: int = 2) -> dict:
    """Deterministic tiny model (PORT_SPEC §4) through the real export path + reference.json."""
    from inkling_ref import (MIN_ARGMAX_MARGIN, N_GEN, TINY_CONFIG, build_tiny_model, greedy_reference,
                             hf_state_to_checkpoint, load_export_as_hf, select_prompt)

    out = Path(out)
    man = load_and_validate_config(TINY_CONFIG)
    model = build_tiny_model(man)
    ckpt = hf_state_to_checkpoint(model.state_dict())
    if out.exists():  # a fixture generator, not a resumable job: never resume onto stale weights
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_config.json").write_text(json.dumps(TINY_CONFIG, indent=2))
    _set_threads(workers)
    summary = Exporter(DictSource(ckpt), man, out, workers=workers).run()
    if not summary["complete"]:
        raise SystemExit("[tiny] export incomplete")
    # reference.json from HF on the DEQUANTIZED weights (what the Rust loader will see); the prompt
    # is the fixtures' prompt (argmax margins robust for both the f32 and the int4-roundtrip model).
    prompt, seed, margin = select_prompt(man, model)
    rt_model, _ = load_export_as_hf(out)
    ref, margins = greedy_reference(rt_model, prompt, N_GEN)
    worst = min(min(margins["prompt"]), min(margins["greedy"]))
    if worst < MIN_ARGMAX_MARGIN:
        raise SystemExit(f"[tiny] dequantized model argmax margin {worst:.3f} < {MIN_ARGMAX_MARGIN}")
    (out / "reference.json").write_text(json.dumps(ref, indent=2))
    log(f"[tiny] reference.json prompt seed {seed} greedy {ref['greedy_ids']} "
        f"(min top-2 logit margin {worst:.3f})")
    log(f"[tiny] wrote {man['num_layers']} layers to {out}")
    return summary


def layers_done_check(out: Path, model_dir=None) -> bool:
    out = Path(out)
    cfg = Path(model_dir) / "config.json" if model_dir else out / "source_config.json"
    if not cfg.exists():
        raise SystemExit(f"[export_inkling] --layers-done-check: no config at {cfg} (pass --model DIR)")
    man = load_and_validate_config(cfg)
    complete, missing, layers_done, embed_done, head_done = completeness(man, out)
    if complete:
        if not (out / "manifest.json").exists():
            write_manifest(man, out)
        log(f"[layers-done-check] OK: embed, head and {layers_done}/{man['num_layers']} layers complete")
        print(json.dumps(man, indent=2))
        return True
    log(f"[layers-done-check] INCOMPLETE: layers {layers_done}/{man['num_layers']}, "
        f"embed_done={int(embed_done)} head_done={int(head_done)}")
    for m in missing:
        log("  - " + m)
    stale = [layer_marker(out, li) for li in range(man["num_layers"])
             if layer_marker(out, li).exists() and any(m.startswith(f"layer {li:02d}:") for m in missing)]
    for p in stale:  # a marker over a missing/truncated file would make the exporter skip the layer
        p.unlink()
        log(f"  - removed stale marker {p.name} (its layer has missing/truncated outputs; re-run the export)")
    return False


def _parse_layers(s: str):
    a, _, b = s.partition("-")
    lo = int(a)
    return lo, int(b) if b else lo


def main():
    ap = argparse.ArgumentParser(description="Inkling exporter (PORT_SPEC.md)")
    ap.add_argument("--validate", type=Path, help="validate a config.json against the inkling contract")
    ap.add_argument("--strict", action="store_true", help="also fail when a contract flag is absent")
    ap.add_argument("--tiny", type=Path, help="write the deterministic tiny model to this dir")
    ap.add_argument("--model", type=Path, help="local checkpoint dir (config.json + safetensors shards)")
    ap.add_argument("--out", type=Path, help="output dir")
    ap.add_argument("--layers", type=_parse_layers, help="layer range a-b (inclusive)")
    ap.add_argument("--shards-only", action="store_true", help="only the int4 expert/dense bins")
    ap.add_argument("--skip-missing-shards", action="store_true",
                    help="streaming pass: skip tensors whose shard file is absent, exit 0 with a summary")
    ap.add_argument("--delete-consumed-shards", action="store_true",
                    help="delete a source shard once every tensor it holds is exported and fsynced")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--layers-done-check", action="store_true",
                    help="assert the export at --out is complete and print the manifest (exit 1 otherwise)")
    args = ap.parse_args()

    if args.validate:
        man = load_and_validate_config(args.validate, strict=args.strict)
        print(json.dumps(man, indent=2))
        log(f"[validate] OK: inkling config ({man['num_layers']} layers, {man['num_experts']} experts, "
            f"top-{man['top_k']}, hidden {man['hidden_size']}, dense {man['dense_layers']})")
        return
    if args.layers_done_check:
        if not args.out:
            ap.error("--layers-done-check needs --out")
        sys.exit(0 if layers_done_check(args.out, args.model) else 1)
    if args.tiny:
        export_tiny(args.tiny, workers=args.workers)
        return
    if args.model and args.out:
        export_real(args.model, args.out, layers=args.layers, shards_only=args.shards_only,
                    skip_missing=args.skip_missing_shards, delete_consumed=args.delete_consumed_shards,
                    workers=args.workers, strict=args.strict)
        return
    ap.error("one of --validate, --tiny, --model/--out, or --layers-done-check is required")


if __name__ == "__main__":
    main()
