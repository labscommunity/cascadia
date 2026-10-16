"""fp32 reference for a decision tree, one decoder layer resident at a time (LoRA merged per layer).

Replays the server's --decision-dump, so it measures quantization only (codec correctness is
codec_golden.py's job). Peak memory ~ one fp32 layer + the passes' activations.

  python tools/decision/reference_stream.py jev  --weights JEV_BF16 --lora JEV/adapter_vllm/adapter_model.safetensors --tree TREE --dump d.parity.jsonl --out ref.json
  python tools/decision/reference_stream.py kev  --weights QWEN35_9B_BASE --lora KEV/adapter_model.safetensors --tree TREE --dump d.parity.jsonl --out ref.json
  python tools/decision/reference_stream.py clef --weights CLEF_DIR --clef-dir CLEF_DIR --tree TREE --dump d.parity.jsonl --out ref.json
  python tools/decision/reference_stream.py jev  --weights JEV_BF16 --lora JEV/adapter_vllm/adapter_model.safetensors --tree TREE --dump d.wide.parity.jsonl --out wide_ref.json --wide

REF.json maps "<record id>.<question id>" to probabilities in API option order (noul [false, true];
choice in request order; score by level index): exactly one entry per question of the 50 records,
which is what parity_check.py --ref requires.

--wide replays WIDE_RECORDS (JEV choices past the trained slots; the slice parity_check.py --wide
wrote). Each pass of a question's plan is one fp32 choice pass over its own options (trained bias on
the first slots, none beyond), and the passes are combined with codec_golden's ports of
serve_decide's formulas: the permute mean over the orders CPython's random draws, or the tournament
combination over the reference's own finalists: the server's final (read back through the tree's
tokenizer.json) when its fp32 group passes pick the same ones, else a second fp32 sweep runs the
final over the reference's finalists (FINALISTS names the options swapped), so a server that picks
the wrong finalists differs from the reference and the gate sees it.
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
from codec_golden import (TokenizerFile, align, final_indices, groups, jev_pass_text, jev_plan_probs,  # noqa: E402
                          rescale_score, tokenizer_added, tournament_final)
from records import WIDE_RECORDS, records  # noqa: E402


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


def tree_tokenizer(tree):
    """The tree's tokenizer.json (a JEV tree ships the one its choice labels were checked against)."""
    path = Path(tree) / "tokenizer.json"
    if not path.is_file():
        raise SystemExit(f"ERROR {path} not found: a tournament's final is read back through the tree's tokenizer")
    return TokenizerFile(path)


def server_finalists(rec, qid, final, tok, jev_prof):
    """The options the server's tournament final ran over, read back from its prompt."""
    if tok is None:
        raise SystemExit(f"MISMATCH {rec['id']}.{qid}: a tournament final needs the tree's tokenizer to read its options")
    labels = jev_prof.get("choice_labels")
    if not labels:
        raise SystemExit(f"MISMATCH {rec['id']}.{qid}: decision.json has no choice_labels, yet the server ran a tournament")
    return final_indices(rec, qid, final["readout"]["prompt_ids"], tok, labels, tokenizer_added(tok)[1])


def wide_dist(j, p, h, weights_dir, lo_ad):
    """One pass of a JEV multi-pass plan over its own options by position: the trained bias where the
    choice slots reach, none beyond, and the choice temperature (serve_decide s1_pass)."""
    lo_slot, hi_slot = j["slots"]["choice"]
    vids = p["readout"]["head"]["token_ids"]
    logits = lm_rows(weights_dir, vids) @ h[-1]
    d = lo_ad.lm_head_logits(vids, h[-1])
    if d is not None:
        logits = logits + d
    return softmax([(float(l) + (j["bias"][lo_slot + i] if lo_slot + i < hi_slot else 0.0)) / j["temperature"]["choice"]
                    for i, l in enumerate(logits)])


def reference_final(rec, qid, own, tok, jev_prof):
    """A dump-style pass for the tournament final over the REFERENCE's finalists `own`: the server
    codec's layout (codec_golden.jev_pass_text, checked against the server by codec_golden --wide)
    through the tree's tokenizer, labelled from the tree's first choice labels."""
    labels, ids = jev_prof["choice_labels"], jev_prof["choice_label_ids"]
    prompt = tok(jev_pass_text(rec, qid, own, labels, tokenizer_added(tok)[1]), add_special_tokens=False).input_ids
    return {"parts": [[qid, 0, len(own)]], "strategy": "tournament", "round": 2,
            "readout": {"prompt_ids": list(prompt), "head": {"kind": "verbalizer", "token_ids": ids[:len(own)]}}}


def family_probs(family, rec, passes, hidden, tree, weights_dir, lora, lora_scale, clef_dir, tok=None,
                 own_finals=None, pending=None):
    """{question id: probabilities in API option order} for one record's passes. `tok` (JEV): reads a
    tournament's final back (tree_tokenizer). Where the reference's fp32 groups pick other finalists
    than the server ran, the reference is combined over its OWN final: `own_finals[(record, qid)]` is
    that final's distribution; without it the question is appended to `pending` as (record, qid,
    finalists) with a None answer, for build_ref's second sweep."""
    prof = json.loads((Path(tree) / "decision.json").read_text())
    out = {}
    plans = {}  # JEV qid -> (strategy, [(pass, distribution)]) of a choice past the trained slots
    lo_ad = Lora(lora, lora_scale) if family == "jev" else None
    for p, h in zip(passes, hidden, strict=True):
        head = p["readout"]["head"]
        if family == "jev":
            j = prof["jev"]
            qid = single_part(rec, p)
            q = rec["questions"][qid]
            kind = q["type"]
            lo_slot, hi_slot = j["slots"][kind]
            vids = head["token_ids"]
            last = h[-1]
            logits = lm_rows(weights_dir, vids) @ last
            d = lo_ad.lm_head_logits(vids, last)
            if d is not None:
                logits = logits + d
            strategy = p.get("strategy", "native")
            if strategy != "native":
                plans.setdefault(qid, (strategy, []))[1].append((p, wide_dist(j, p, h, weights_dir, lo_ad)))
                continue
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
    for qid, (strategy, ps) in plans.items():
        dists, fin = [d for _, d in ps], None
        if strategy == "tournament":
            fin = server_finalists(rec, qid, ps[-1][0], tok, prof["jev"])
            own = tournament_final(groups(len(rec["questions"][qid]["criteria"])), dists[:-1])
            if own != fin:
                key = (rec["id"], qid)
                if own_finals is not None and key in own_finals:
                    out[qid] = jev_plan_probs(rec, qid, strategy, dists[:-1] + [own_finals[key]], own)
                    continue
                print(f"FINALISTS {rec['id']}.{qid}: the reference's fp32 groups pick {own}, the server ran {fin} "
                      f"(server only {sorted(set(fin) - set(own))}, reference only {sorted(set(own) - set(fin))}); "
                      f"the reference uses its own fp32 final", flush=True)
                if pending is None:
                    raise SystemExit(f"MISMATCH {rec['id']}.{qid}: the reference's finalists differ from the server's "
                                     f"and no fp32 final over them was run (build_ref runs it)")
                pending.append((rec, qid, own))
                out[qid] = None
                continue
        out[qid] = jev_plan_probs(rec, qid, strategy, dists, fin)
    return out


def build_ref(family, weights_dir, dump, recs, tree, lora=None, lora_scale=None, clef_dir=None, threads=4, tok=None):
    """{"<record>.<qid>": probs} over EVERY question of `recs`, in record/question order.
    A dump out of step with `recs`, a question left unanswered, a duplicate or a wrong option
    count fails: parity_check scores every reference entry, so partial output must never be written."""
    aligned = list(align(dump, recs))
    flat = [p for _, ps in aligned for p in ps]
    hidden = stream_hidden(weights_dir, flat, lora, lora_scale, threads=threads, allow_lm_head_lora=family == "jev")
    spans, k = [], 0
    for _, ps in aligned:
        spans.append(hidden[k:k + len(ps)])
        k += len(ps)
    pending = []
    answers = [family_probs(family, rec, ps, h, tree, weights_dir, lora, lora_scale, clef_dir, tok=tok, pending=pending)
               for (rec, ps), h in zip(aligned, spans)]
    if pending:
        # A second, small fp32 sweep: the finals over the reference's own finalists, so a server that
        # picked the wrong ones is measured against what it should have run, not against itself.
        j = json.loads((Path(tree) / "decision.json").read_text())["jev"]
        finals = [reference_final(rec, qid, own, tok, j) for rec, qid, own in pending]
        hs = stream_hidden(weights_dir, finals, lora, lora_scale, threads=threads, allow_lm_head_lora=True)
        lo_ad = Lora(lora, lora_scale)
        own_finals = {(rec["id"], qid): wide_dist(j, f, h, weights_dir, lo_ad)
                      for (rec, qid, _), f, h in zip(pending, finals, hs)}
        redo = {rec["id"] for rec, _, _ in pending}
        answers = [family_probs(family, rec, ps, h, tree, weights_dir, lora, lora_scale, clef_dir, tok=tok,
                                own_finals=own_finals) if rec["id"] in redo else a
                   for (rec, ps), h, a in zip(aligned, spans, answers)]
    ref = {}
    for (rec, ps), probs in zip(aligned, answers):
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
                    help="the <dump>.parity.jsonl slice `parity_check.py --dump` wrote (exactly one run of the 50 records; "
                         "with --wide, the <dump>.wide.parity.jsonl slice of WIDE_RECORDS)")
    ap.add_argument("--tree", required=True, help="exported tree: decision.json (+ decision/ pointer weights for kev)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=None, help="override adapter_config.json")
    ap.add_argument("--clef-dir", default=None, help="clef: dir with joint_head.safetensors + joint_head_config.json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--wide", action="store_true",
                    help="jev: the dump is parity_check.py --wide's slice of WIDE_RECORDS (needs the tree's tokenizer.json)")
    a = ap.parse_args()
    if a.wide and a.family != "jev":
        ap.error("--wide is for jev only")
    if a.family == "clef" and not a.clef_dir:
        ap.error("clef needs --clef-dir")
    if a.family == "kev" and not a.lora:
        # Kev's head was trained on Base + LoRA hidden states: a reference without the adapter is
        # the stock backbone, and parity against a tree exported the same way would still pass.
        ap.error("kev needs --lora (Kev's adapter_model.safetensors) with --weights (its Base checkpoint)")
    scale = resolve_lora_scale(a.lora, a.lora_scale)
    recs, tok = (WIDE_RECORDS, tree_tokenizer(a.tree)) if a.wide else (records(), None)
    ref = build_ref(a.family, a.weights, a.dump, recs, a.tree, a.lora, scale, a.clef_dir, threads=a.threads, tok=tok)
    Path(a.out).write_text(json.dumps(ref))
    print(f"wrote {len(ref)} reference distributions to {a.out}")


if __name__ == "__main__":
    main()
