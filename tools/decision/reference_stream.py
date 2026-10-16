"""fp32 reference for a decision tree, one decoder layer resident at a time (LoRA merged per layer).

Replays the server's --decision-dump, so it measures quantization only (codec correctness is
codec_golden.py's job). Peak memory ~ one fp32 layer + the passes' activations.

  python tools/decision/reference_stream.py jev  --weights JEV_BF16 --lora JEV/adapter_vllm/adapter_model.safetensors --tree TREE --dump d.parity.jsonl --out ref.json
  python tools/decision/reference_stream.py kev  --weights QWEN35_9B_BASE --lora KEV/adapter_model.safetensors --tree TREE --dump d.parity.jsonl --out ref.json
  python tools/decision/reference_stream.py clef --weights CLEF_DIR --clef-dir CLEF_DIR --tree TREE --dump d.parity.jsonl --out ref.json

REF.json maps "<record id>.<question id>" to probabilities in API option order (noul [false, true];
choice in request order; score by level index): exactly one entry per question of the 50 records,
which is what parity_check.py --ref requires.
"""
import argparse
import gc
import json
import math
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qwen36_surgery"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from codec_golden import align, rescale_score  # noqa: E402
from records import records  # noqa: E402


class Ckpt:
    """An HF safetensors checkpoint read tensor by tensor (row by row for embeddings/lm_head)."""

    def __init__(self, d):
        self.d = Path(d)
        self.idx = json.loads((self.d / "model.safetensors.index.json").read_text())["weight_map"]
        self.prefix = "model.language_model." if any(k.startswith("model.language_model.") for k in self.idx) else "model."
        self._f = {}

    def _file(self, name):
        f = self.idx[name]
        if f not in self._f:
            self._f[f] = safe_open(self.d / f, "pt")
        return self._f[f]

    def tensor(self, name):
        return self._file(name).get_tensor(name).float()

    def rows(self, name, ids):
        s = self._file(name).get_slice(name)
        return torch.stack([s[int(i):int(i) + 1][0] for i in ids]).float()

    def release(self):
        """Close every open shard. Windows charges a mapped shard's touched pages to the process
        until its handle closes, so a 27B checkpoint (~54 GB of shards) outgrows a 32 GB host."""
        self._f.clear()


class Lora:
    """PEFT adapter: delta(module) = scale * B @ A in fp32, or None when the module has no adapter.
    `module` is the path below the decoder ("layers.3.mlp.down_proj") or "lm_head"; the adapter may
    name it with or without the "model." / "model.language_model." prefix. weight_swap.lora_modules'
    refusals (embedding LoRA, modules_to_save tensors, unpaired keys) and, when adapter_config.json is
    beside the adapter, lora_scale_from_config's (modules_to_save, DoRA, ...) hold even with an
    explicit scale: the reference must drop nothing the export would also drop."""

    def __init__(self, path, scale):
        from weight_swap import lora_modules, lora_scale_from_config
        self.f = safe_open(path, "pt") if path else None
        if self.f is not None and scale is None:
            raise ValueError("a LoRA adapter needs a scale (--lora-scale or adapter_config.json)")
        self.mods = lora_modules(self.f.keys(), path) if self.f else {}
        if path and (Path(path).parent / "adapter_config.json").exists():
            lora_scale_from_config(path)
        self.scale = scale
        self.merged = Counter()

    def delta(self, module):
        """The full merged delta, for decoder weights (one layer's worth at a time)."""
        side = self.mods.get(module)
        if side is None:
            return None
        self.merged[module] += 1
        return self.scale * (self.f.get_tensor(side["B"]).float() @ self.f.get_tensor(side["A"]).float())

    def lm_head_logits(self, vids, h):
        """scale * B[vids] @ (A @ h): the lm_head adapter's logit delta for `vids` only. The full
        [vocab, hidden] delta (~4 GB fp32 for a 248k vocab) is never materialised. None without one."""
        side = self.mods.get("lm_head")
        if side is None:
            return None
        s = self.f.get_slice(side["B"])
        b = torch.stack([s[int(i):int(i) + 1][0] for i in vids]).float()
        return self.scale * (b @ (self.f.get_tensor(side["A"]).float() @ h))


def text_config(weights_dir):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(weights_dir)
    cfg = getattr(cfg, "text_config", cfg)
    cfg._attn_implementation = "sdpa"
    return cfg


def stream_hidden(weights_dir, passes, lora=None, lora_scale=None, threads=4, allow_lm_head_lora=False):
    """Final-norm hidden [T, H] (fp32) of every pass's prompt_ids, one decoder layer resident at a time.
    Every adapter module must merge into a decoder weight; an lm_head adapter is tolerated only with
    allow_lm_head_lora (JEV, whose verbalizer head applies it in family_probs). torch runs on
    `threads` threads for the call; the caller's setting is restored after it."""
    prev = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        return _stream_hidden(weights_dir, passes, lora, lora_scale, allow_lm_head_lora)
    finally:
        torch.set_num_threads(prev)


@torch.inference_mode()
def _stream_hidden(weights_dir, passes, lora, lora_scale, allow_lm_head_lora):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    ck, cfg = Ckpt(weights_dir), text_config(weights_dir)
    lo = Lora(lora, lora_scale)
    ids = [p["readout"]["prompt_ids"] for p in passes]
    acts = [ck.rows(f"{ck.prefix}embed_tokens.weight", x).unsqueeze(0) for x in ids]
    # Qwen3.5 text positions: [4, B, T] = text row + 3 identical mrope rows (Qwen3_5TextModel.forward)
    pos = [torch.arange(a.shape[1]).view(1, 1, -1).expand(4, 1, -1) for a in acts]
    rot = M.Qwen3_5TextRotaryEmbedding(cfg)
    pe = [rot(a, p[1:]) for a, p in zip(acts, pos)]
    for i in range(cfg.num_hidden_layers):
        layer = M.Qwen3_5DecoderLayer(cfg, i).float()
        sd, merged = {}, 0
        for name in layer.state_dict():
            w = ck.tensor(f"{ck.prefix}layers.{i}.{name}")
            if name.endswith(".weight"):
                d = lo.delta(f"layers.{i}.{name[:-len('.weight')]}")
                if d is not None:
                    w = w + d
                    merged += 1
            sd[name] = w
        layer.load_state_dict(sd, strict=True)
        layer.eval()
        for j in range(len(acts)):
            # attention_mask=None: sdpa runs causal; the linear-attention layers need no mask at batch 1
            acts[j] = layer(acts[j], position_embeddings=pe[j], attention_mask=None, position_ids=pos[j][0])
        del layer, sd
        ck.release()
        gc.collect()
        print(f"layer {i} ({cfg.layer_types[i]}, {merged} LoRA merged)", flush=True)
    # every adapter module merged exactly once; lm_head only for JEV (applied in family_probs)
    want = {m for m in lo.mods if not (allow_lm_head_lora and m == "lm_head")}
    unused = sorted(want - set(lo.merged))
    if unused:
        raise SystemExit(f"ERROR {lora}: {len(unused)} adapter module(s) matched no decoder weight, e.g. {unused[:3]}")
    twice = sorted(m for m, n in lo.merged.items() if n > 1)
    if twice:
        raise SystemExit(f"ERROR {lora}: {len(twice)} adapter module(s) merged more than once, e.g. {twice[:3]}")
    norm = M.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.load_state_dict({"weight": ck.tensor(f"{ck.prefix}norm.weight")})
    return [norm(a[0]) for a in acts]


def lm_head_name(ck, weights_dir):
    """lm_head.weight, or the input embeddings only when config.json ties them
    (the same rule as decision_export._lm_head_name)."""
    if "lm_head.weight" in ck.idx:
        return "lm_head.weight"
    cfg_path = Path(weights_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    tied = cfg.get("tie_word_embeddings", (cfg.get("text_config") or {}).get("tie_word_embeddings", False))
    emb = f"{ck.prefix}embed_tokens.weight"
    if tied and emb in ck.idx:
        return emb
    raise SystemExit(f"ERROR {weights_dir}: no lm_head.weight in model.safetensors.index.json "
                     f"(and tie_word_embeddings is not set)")


@lru_cache(maxsize=1)
def _lm_head(weights_dir):
    """(Ckpt, lm_head tensor name), opened once per checkpoint: family_probs reads rows for every
    pass. Only the shard holding lm_head is ever opened through it."""
    ck = Ckpt(weights_dir)
    return ck, lm_head_name(ck, weights_dir)


def lm_rows(weights_dir, ids):
    """lm_head rows (fp32) for `ids`; the input embeddings when the checkpoint ties them."""
    ck, name = _lm_head(str(weights_dir))
    return ck.rows(name, ids)


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return (e / e.sum()).tolist()


@lru_cache(maxsize=2)
def clef_tensor_head(clef_dir):
    """Task 14's joint head over flat tensors, loaded once per clef dir."""
    from decision.clef_head import TensorHead, load_joint_head
    return TensorHead(load_joint_head(clef_dir)).eval()


def n_options(q):
    """How many probabilities the API reports for a question (option_keys)."""
    return 2 if q["type"] == "noul" else len(q["criteria"])


def single_part(rec, p):
    if len(p["parts"]) != 1:
        raise SystemExit(f"MISMATCH {rec['id']}: a pass with {len(p['parts'])} parts; this family answers one question per pass")
    return p["parts"][0][0]


def family_probs(family, rec, passes, hidden, tree, weights_dir, lora, lora_scale, clef_dir):
    """{question id: probabilities in API option order} for one record's passes."""
    prof = json.loads((Path(tree) / "decision.json").read_text())
    out = {}
    lo_ad = Lora(lora, lora_scale) if family == "jev" else None
    for p, h in zip(passes, hidden, strict=True):
        head = p["readout"]["head"]
        if family == "jev":
            j = prof["jev"]
            qid = single_part(rec, p)
            q = rec["questions"][qid]
            kind = q["type"]
            lo_slot = j["slots"][kind][0]
            vids = head["token_ids"]
            last = h[-1]
            logits = lm_rows(weights_dir, vids) @ last
            d = lo_ad.lm_head_logits(vids, last)
            if d is not None:
                logits = logits + d
            z = [(float(l) + j["bias"][lo_slot + i]) / j["temperature"][kind] for i, l in enumerate(logits)]
            pr = softmax(z)
            out[qid] = rescale_score(pr, len(q["criteria"])) if kind == "score" else pr
        elif family == "kev":
            qid = single_part(rec, p)
            dd = Path(tree) / "decision"
            dp = json.loads((dd / "head.json").read_text())["dp"]
            H = h.shape[1]

            def ld(n, shape):
                return torch.from_numpy(np.fromfile(dd / f"{n}.f32", dtype="<f4").reshape(shape))
            qv = ld("pointer_q_w", (dp, H)) @ h[head["decide_pos"]] + ld("pointer_q_b", (dp,))
            kv = h[head["option_pos"]] @ ld("pointer_k_w", (dp, H)).T + ld("pointer_k_b", (dp,))
            z = (kv @ qv) / math.sqrt(dp) / prof["kev"]["temperature"]
            out[qid] = softmax(z.tolist())
        elif family == "clef":
            th = clef_tensor_head(clef_dir)
            T = h.shape[0]
            spans = [q["span"] for q in head["questions"]] + [o["span"] for q in head["questions"] for o in q["options"]]
            mask = torch.zeros(len(spans), T)
            for r, (a, b) in enumerate(spans):
                mask[r, a:b] = 1.0 / (b - a)
            ids = p["readout"]["prompt_ids"]
            lexical = torch.stack([lm_rows(weights_dir, ids[o["span"][0]:o["span"][1]]).mean(0)
                                   for q in head["questions"] for o in q["options"]])
            option_q = torch.tensor([qi for qi, q in enumerate(head["questions"]) for _ in q["options"]])
            types = torch.tensor([q["qtype"] for q in head["questions"]])
            with torch.no_grad():
                logits = th(h, mask, lexical, option_q, types).tolist()
            for (qid, a, b) in p["parts"]:
                pr = softmax(logits[a:b])
                q = rec["questions"][qid]
                if q["type"] == "noul":  # head order [true, false] -> [false, true]
                    pr = [pr[1], pr[0]]
                elif q["type"] == "choice":  # head order = sorted keys -> request order
                    order = sorted(q["criteria"])
                    pr = [pr[order.index(k)] for k in q["criteria"]]
                out[qid] = pr
        else:
            raise ValueError(f"unknown family {family!r}")
    return out


def build_ref(family, weights_dir, dump, recs, tree, lora=None, lora_scale=None, clef_dir=None, threads=4):
    """{"<record>.<qid>": probs} over EVERY question of `recs`, in record/question order.
    A dump out of step with `recs`, a question left unanswered, a duplicate or a wrong option
    count fails: parity_check scores every reference entry, so partial output must never be written."""
    aligned = list(align(dump, recs))
    flat = [p for _, ps in aligned for p in ps]
    hidden = stream_hidden(weights_dir, flat, lora, lora_scale, threads=threads, allow_lm_head_lora=family == "jev")
    ref, k = {}, 0
    for rec, ps in aligned:
        probs = family_probs(family, rec, ps, hidden[k:k + len(ps)], tree, weights_dir, lora, lora_scale, clef_dir)
        k += len(ps)
        if set(probs) != set(rec["questions"]):
            raise SystemExit(f"MISMATCH {rec['id']}: reference answered {sorted(probs)}, record asks {sorted(rec['questions'])}")
        for qid, q in rec["questions"].items():
            pr = probs[qid]
            if len(pr) != n_options(q):
                raise SystemExit(f"MISMATCH {rec['id']}.{qid}: {len(pr)} probabilities for {n_options(q)} options")
            ref[f"{rec['id']}.{qid}"] = pr
    expected = sum(len(r["questions"]) for r in recs)
    if len(ref) != expected:
        raise SystemExit(f"MISMATCH {len(ref)} reference entries for {expected} questions")
    return ref


def resolve_lora_scale(lora, override=None):
    """adapter_config.json's scale, or `override`. lora_scale_from_config always runs, so its refusals
    (DoRA, rank/alpha patterns, lora_bias) hold even when only the scale value is overridden."""
    if not lora:
        return override
    from weight_swap import lora_scale_from_config
    scale = lora_scale_from_config(lora)
    return scale if override is None else override


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["jev", "kev", "clef"])
    ap.add_argument("--weights", required=True, help="HF safetensors checkpoint (bf16/fp32)")
    ap.add_argument("--dump", required=True,
                    help="the <dump>.parity.jsonl slice `parity_check.py --dump` wrote (exactly one run of the 50 records)")
    ap.add_argument("--tree", required=True, help="exported tree: decision.json (+ decision/ pointer weights for kev)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=None, help="override adapter_config.json")
    ap.add_argument("--clef-dir", default=None, help="clef: dir with joint_head.safetensors + joint_head_config.json")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.family == "clef" and not a.clef_dir:
        ap.error("clef needs --clef-dir")
    if a.family == "kev" and not a.lora:
        # Kev's head was trained on Base + LoRA hidden states: a reference without the adapter is
        # the stock backbone, and parity against a tree exported the same way would still pass.
        ap.error("kev needs --lora (Kev's adapter_model.safetensors) with --weights (its Base checkpoint)")
    scale = resolve_lora_scale(a.lora, a.lora_scale)
    ref = build_ref(a.family, a.weights, a.dump, records(), a.tree, a.lora, scale, a.clef_dir, threads=a.threads)
    Path(a.out).write_text(json.dumps(ref))
    print(f"wrote {len(ref)} reference distributions to {a.out}")


if __name__ == "__main__":
    main()
