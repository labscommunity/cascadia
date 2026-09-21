#!/usr/bin/env python3
"""E4: score Inkling's SHIPPED multi-token-prediction head over a hidden-state dump.

    mtp_score.py --dump DIR --weights DIR --out results.json [--depths 8] [--variants ref,swap,...] [--threads 32]

Inputs
  --dump     directory of p{i:04}.safetensors written by examples/inkling_spec_dump.rs
             (tokens [T], embed_out [T,H], final_out [T-1,H] = layer 65 output BEFORE the final norm,
             argmax [T-1]; metadata prompt_len, family)
  --weights  directory with mtp.safetensors (HF checkpoint, model.mtp.layers.{0..7}.*),
             head.safetensors (unembed.weight, norm.weight) and embed.safetensors (embed.weight,
             embed_norm.weight) of the int4 export, manifest.json

MTP forward (from vLLM vllm/models/inkling/nvidia/mtp.py and SGLang srt/models/inkling.py, which agree;
transformers' modeling_inkling.py ignores model.mtp.* on load):
  depth k at sequence position t:
      e   = rmsnorm(embed[x_{t+k+1}], backbone embed_norm)           # the normed embedding layer 0 consumes
      hin = input_proj( cat[ hidden_norm_k(prev_t), embed_norm_k(e) ] )   # hidden first (mtp_hidden_states_first)
      h_k = DenseInklingBlock_k(hin)   over the whole sequence, causal; depth 1 and 3 global attention
                                       (8 kv heads, rel extent 1024), the others sliding (16 kv heads, 512)
      logits = unembed(h_k / mup)[:unpadded]          # NO norm on the block output (chain_hidden_post_norm=false)
  prev_t = final_norm(layer65_out[t]) for k = 0 (the target's POST-final-norm hidden, undivided by mup);
           h_{k-1}[t] (raw block output) for k >= 1.
  The prediction of depth k at position t is for token x_{t+k+2}.

Protocols
  A "teacher": depth k is fed the TRUE token x_{t+k+1} at every position (training-style, = production
     with the draft context re-extended on verified tokens).
  B "own":     depth k >= 1 is fed depth k-1's own top-1 token at every generated position (true tokens
     inside the prompt): what a cheap production chain does, its draft context never corrected.
  C "own_noctx": as B, but depths >= 1 see NO context at all (each position alone: attention over itself,
     zero conv history): the floor for an implementation that does not keep the deeper modules' KV/conv
     state current at every accepted token (depth 0 keeps its full context: it runs at every real token).
  Scored only where the target is a token the main model chose greedily AND the anchor position t >= P-1
  (the first state a drafter would see in production): t in [P-1, T-3-k].

Variants (teeth checks, depth 0 only unless --variant-depths): swap (embedding first), no_enorm (skip the
  MTP embed_norm), raw_embed (skip the backbone embed_norm), prenorm_in (feed layer65_out without the final
  norm), out_norm (final norm on the block output before the unembed), no_deint (w13 split in halves
  instead of de-interleaved), no_sconv (drop the four short convs), no_relbias (drop the relative bias),
  noctx (every position alone: no attention context, no conv history), shift0 (embedding of x_t, not x_{t+1}).
"""
import argparse, glob, json, math, os, sys, time
import torch
import torch.nn.functional as F
from safetensors import safe_open

FAMILIES = ["explain", "code", "arithmetic", "story", "lists", "tables", "rewriting", "facts", "poem", "how-to",
            "translation", "true/false"]
H = 6144; EPS = 1e-6; D_REL = 16; REL_EXTENT = 1024; WINDOW = 512; HEAD_DIM = 128; N_HEADS = 64
LOCAL_DEPTHS = {0, 2, 4, 5, 6, 7}


def rms(x, w, eps=EPS):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w


def sconv(y, w, enabled=True, noctx=False):
    """y [T, C]; w [C, 1, K]: y + causal depthwise conv (HF InklingShortConvolution, fp32).
    noctx: every row is its own sequence of length 1 (zero conv history)."""
    if not enabled:
        return y
    if noctx:
        return y * w[:, 0, -1] + y
    T, C = y.shape
    x = y.float().t().unsqueeze(0)
    out = F.conv1d(x, w, padding=w.shape[-1] - 1, groups=C)[:, :, :T]
    return out.squeeze(0).t() + y


class Depth:
    def __init__(self, f, k, variant_flags):
        p = f"model.mtp.layers.{k}."
        g = lambda n: f.get_tensor(p + n).float()
        self.k = k
        self.local = k in LOCAL_DEPTHS
        self.hidden_norm = g("hidden_norm.weight"); self.embed_norm = g("embed_norm.weight")
        self.input_proj = g("input_proj.weight")
        b = "transformer_block."
        self.attn_norm = g(b + "attn_norm.weight"); self.mlp_norm = g(b + "mlp_norm.weight")
        self.wq = g(b + "attn.wq_du.weight"); self.wk = g(b + "attn.wk_dv.weight"); self.wv = g(b + "attn.wv_dv.weight")
        self.wr = g(b + "attn.wr_du.weight"); self.wo = g(b + "attn.wo_ud.weight")
        self.q_norm = g(b + "attn.q_norm.weight"); self.k_norm = g(b + "attn.k_norm.weight")
        self.k_sconv = g(b + "attn.k_sconv.weight"); self.v_sconv = g(b + "attn.v_sconv.weight")
        self.rel_proj = g(b + "attn.rel_logits_proj.proj")
        self.attn_sconv = g(b + "attn_sconv.weight"); self.mlp_sconv = g(b + "mlp_sconv.weight")
        w13 = g(b + "mlp.w13_dn.weight")
        self.gate_i, self.up_i = w13[0::2].contiguous(), w13[1::2].contiguous()      # checkpoint: row 2i gate, 2i+1 up
        half = w13.shape[0] // 2
        self.gate_h, self.up_h = w13[:half], w13[half:]                                # wrong on purpose (no_deint)
        self.w2 = g(b + "mlp.w2_md.weight"); self.global_scale = g(b + "mlp.global_scale")
        self.kv_heads = self.wk.shape[0] // HEAD_DIM
        self.extent = self.rel_proj.shape[1]
        assert self.kv_heads == (16 if self.local else 8), (k, self.kv_heads)
        assert self.extent == (WINDOW if self.local else REL_EXTENT), (k, self.extent)

    def block(self, x, flags):
        T = x.shape[0]
        use_sconv = "no_sconv" not in flags
        nc = "noctx" in flags                                          # each position alone: no attention context, no conv history
        h = rms(x, self.attn_norm)
        q = (h @ self.wq.t()).view(T, N_HEADS, HEAD_DIM)
        kk = sconv(h @ self.wk.t(), self.k_sconv, use_sconv, nc).view(T, self.kv_heads, HEAD_DIM)
        vv = sconv(h @ self.wv.t(), self.v_sconv, use_sconv, nc).view(T, self.kv_heads, HEAD_DIM)
        r = (h @ self.wr.t()).view(T, N_HEADS, D_REL)
        q = rms(q, self.q_norm).transpose(0, 1)                       # [heads, T, d]
        kk = rms(kk, self.k_norm).transpose(0, 1)
        vv = vv.transpose(0, 1)
        rep = N_HEADS // self.kv_heads
        kk = kk.repeat_interleave(rep, dim=0); vv = vv.repeat_interleave(rep, dim=0)
        scores = q @ kk.transpose(1, 2) / HEAD_DIM                     # q/k are RMS-normed per head: 1/d scaling
        pos = torch.arange(T)
        dist = pos[:, None] - pos[None, :]                             # [q, k]
        if "no_relbias" not in flags:
            rel = (r @ self.rel_proj).transpose(0, 1)                  # [heads, T, extent]
            idx = dist.clamp(0, self.extent - 1).unsqueeze(0).expand(N_HEADS, -1, -1)
            bias = rel.gather(-1, idx).masked_fill(((dist < 0) | (dist >= self.extent)).unsqueeze(0), 0.0)
            scores = scores + bias
        mask = (dist != 0) if nc else (dist < 0)
        if self.local:
            mask = mask | (dist >= WINDOW)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        a = torch.softmax(scores, dim=-1) @ vv                         # [heads, T, d]
        a = a.transpose(0, 1).reshape(T, N_HEADS * HEAD_DIM) @ self.wo.t()
        x = x + sconv(a, self.attn_sconv, use_sconv, nc)
        h = rms(x, self.mlp_norm)
        gate, up = (self.gate_h, self.up_h) if "no_deint" in flags else (self.gate_i, self.up_i)
        m = (F.silu(h @ gate.t()) * (h @ up.t())) @ self.w2.t() * self.global_scale
        return x + sconv(m, self.mlp_sconv, use_sconv, nc)

    def forward(self, prev, e, flags=()):
        hn = rms(prev, self.hidden_norm)
        en = e if "no_enorm" in flags else rms(e, self.embed_norm)
        cat = torch.cat([en, hn], -1) if "swap" in flags else torch.cat([hn, en], -1)
        return self.block(cat @ self.input_proj.t(), flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--depths", type=int, default=8); ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--variants", default="swap,no_enorm,raw_embed,prenorm_in,out_norm,no_deint,no_sconv,no_relbias,noctx,shift0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="keep the sequences already in --out, score only new dump files")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_grad_enabled(False)
    man = json.load(open(os.path.join(a.weights, "manifest.json")))
    mup = float(man["logits_mup_width_multiplier"]); V = int(man["unpadded_vocab_size"])
    t0 = time.time()
    with safe_open(os.path.join(a.weights, "head.safetensors"), "pt") as f:
        unembed = f.get_tensor("unembed.weight").float()[:V].contiguous(); final_norm = f.get_tensor("norm.weight").float()
    with safe_open(os.path.join(a.weights, "embed.safetensors"), "pt") as f:
        embed = f.get_tensor("embed.weight"); embed_norm = f.get_tensor("embed_norm.weight").float()     # embed stays bf16
    with safe_open(os.path.join(a.weights, "mtp.safetensors"), "pt") as f:
        depths = [Depth(f, k, ()) for k in range(a.depths)]
    print("weights loaded in %.0fs" % (time.time() - t0), flush=True)
    variants = [v for v in a.variants.split(",") if v]

    def logits_top(h, flags=()):
        x = rms(h, final_norm) if "out_norm" in flags else h
        return (x / mup) @ unembed.t()

    files = sorted(glob.glob(os.path.join(a.dump, "p*.safetensors")))
    if a.limit:
        files = files[:a.limit]
    rows = []            # one dict per sequence
    if a.resume and os.path.exists(a.out):
        rows = json.load(open(a.out))
    done = {r["i"] for r in rows}
    for fi, path in enumerate(files):
        if int(os.path.basename(path)[1:5]) in done:
            continue
        t1 = time.time()
        with safe_open(path, "pt") as f:
            md = f.metadata(); toks = f.get_tensor("tokens"); eo = f.get_tensor("embed_out").float()
            fo = f.get_tensor("final_out").float(); am = f.get_tensor("argmax")
        P = int(md["prompt_len"]); fam = int(md["family"]); i = int(md["i"]); T = toks.shape[0]
        assert fo.shape[0] == T - 1 and am.shape[0] == T - 1 and eo.shape[0] == T
        assert torch.equal(am[P - 1:], toks[P:]), "argmax != generated tokens"
        # self-check of the dump + head: final norm + unembed must reproduce the dumped argmax
        chk = logits_top(rms(fo, final_norm)).argmax(-1)
        head_agree = (chk[P - 1:] == am[P - 1:]).float().mean().item()
        g = rms(fo, final_norm)                                   # post-final-norm hidden, rows 0..T-2
        rec = {"i": i, "family": fam, "P": P, "T": T, "head_agree": head_agree, "tokens": toks.tolist()}

        def emb_of(ids):
            return rms(embed[ids].float(), embed_norm)

        h0 = depths[0].forward(g, eo[1:T]); lg0 = logits_top(h0); p0 = lg0.argmax(-1)   # depth 0 is the same in both
        for proto in ("teacher", "own", "own_noctx"):
            prev = g; pred_prev = None; hits = []; preds = []
            for k in range(a.depths):
                L = T - 1 - k                                      # positions 0..T-2-k have a true x_{t+k+1}
                true_next = toks[k + 1:k + 1 + L]
                if k == 0:
                    hk, pk = h0, p0
                else:
                    if proto == "teacher":
                        e = eo[k + 1:k + 1 + L]
                    else:
                        # known when the first chain starts (anchor P-1): x_0..x_P, so true tokens for t <= P-1-k
                        own = pred_prev[:L].clone()
                        n_true = max(P - k, 0)
                        own[:n_true] = true_next[:n_true]
                        e = emb_of(own)
                    hk = depths[k].forward(prev[:L], e, ("noctx",) if proto == "own_noctx" else ())
                    pk = logits_top(hk).argmax(-1)                 # prediction for x_{t+k+2}, t = 0..L-1
                # scored anchors t in [P-1, T-3-k]
                ts = torch.arange(P - 1, max(T - 2 - k, P - 1))
                hit = (pk[ts] == toks[ts + k + 2])
                hits.append(hit.tolist()); preds.append(pk[ts].tolist())
                prev, pred_prev = hk, pk
            rec[proto] = {"hits": hits, "preds": preds}
        # vocabulary-prefix pricing at depth 0 + teeth checks at depth 0
        L = T - 1; ts = torch.arange(P - 1, max(T - 2, P - 1)); tgt = toks[ts + 2]
        lg = lg0[ts]
        rec["prefix"] = {str(n): (lg[:, :n].argmax(-1) == tgt).tolist() for n in (32768, 65536, 131072)}
        rec["target_ids"] = tgt.tolist()
        rec["variants"] = {}
        for v in variants:
            flags = (v,)
            prev_v = fo if v == "prenorm_in" else g
            e_v = embed[toks[1:1 + L]].float() if v == "raw_embed" else (eo[0:L] if v == "shift0" else eo[1:1 + L])
            hv = depths[0].forward(prev_v, e_v, flags)
            rec["variants"][v] = (logits_top(hv, flags)[ts].argmax(-1) == tgt).tolist()
        rows.append(rec)
        a1 = sum(rec["teacher"]["hits"][0]) / max(len(rec["teacher"]["hits"][0]), 1)
        print("%3d/%d i=%d fam=%s P=%d T=%d head_agree=%.3f a1=%.3f  (%.1fs)" % (
            fi + 1, len(files), i, FAMILIES[fam], P, T, head_agree, a1, time.time() - t1), flush=True)
        json.dump(rows, open(a.out, "w"))
    print("done", len(rows), "sequences")


if __name__ == "__main__":
    main()
