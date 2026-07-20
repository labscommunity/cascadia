//! dsv4 full-model assembly: MoE gate (sqrtsoftplus / noaux bias / hash),
//! clamped-SwiGLU experts, Hyper-Connections block wiring, embed and the
//! HC head. Port of `Gate` / `Expert` / `MoE` / `Block` / `ParallelHead` /
//! `Transformer` from `tools/deepseek_v4_ref/model.py` (b = 1, greedy).
//!
//! The hidden state is `hc_mult` copies per token (`[hc, dim]`); every
//! sublayer runs pre-mix (Sinkhorn coefficients from the flattened copies)
//! -> norm -> sublayer -> post-mix into fresh copies.
//!
//! Pipeline-parallel staging: a model instance holds a contiguous layer
//! slice; `embed` exists only on the first stage, `head` only on the last.
//! The inter-stage activation is the flattened copies (`[s, hc*dim]` f32 of
//! bf16 values) — it rides the existing sparse-moe `Forward` wire frame
//! unchanged (hidden_size = hc*dim). Token ids are needed only by hash-gate
//! layers (the first `n_hash_layers`, always within stage 0 for any sane
//! split), so ids never cross the wire.
//!
//! Experts here hold f32 weights (tiny fixtures / dequantized shells); the
//! production engine swaps the expert matmuls for the int4 kernel behind
//! the same interface.

use super::attn_layer::AttentionLayer;
use super::hc::hc_split_sinkhorn;
use super::math::{linear_bf16, linear_f32, rmsnorm, to_bf16};

/// softplus matching torch's default (beta = 1, threshold = 20).
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

pub struct GateWeights {
    pub weight: Vec<f32>,          // [n_experts, dim]
    pub bias: Option<Vec<f32>>,    // [n_experts] (noaux selection bias)
    pub tid2eid: Option<Vec<i32>>, // [vocab, topk] (hash layers)
}

pub struct Expert {
    pub w1: Vec<f32>, // gate [inter, dim]
    pub w3: Vec<f32>, // up   [inter, dim]
    pub w2: Vec<f32>, // down [dim, inter]
    pub inter: usize,
}

impl Expert {
    /// silu(clamp(w1 x)) * clamp(w3 x) [* route_w] -> w2, reference dtypes.
    fn forward(&self, x: &[f32], dim: usize, limit: f32, route_w: Option<f32>) -> Vec<f32> {
        let inter = self.inter;
        let mut gate = vec![0.0f32; inter];
        let mut up = vec![0.0f32; inter];
        linear_bf16(x, &self.w1, inter, dim, &mut gate);
        linear_bf16(x, &self.w3, inter, dim, &mut up);
        // .float() then clamp (limit > 0), silu * up in f32
        let mut h = vec![0.0f32; inter];
        for i in 0..inter {
            let mut g = gate[i];
            let mut u = up[i];
            if limit > 0.0 {
                u = u.clamp(-limit, limit);
                g = g.min(limit);
            }
            let s = g / (1.0 + (-g).exp()); // silu
            let mut v = s * u;
            if let Some(w) = route_w {
                v *= w;
            }
            h[i] = to_bf16(v); // .to(dtype) before w2
        }
        let mut out = vec![0.0f32; dim];
        linear_bf16(&h, &self.w2, dim, inter, &mut out);
        out
    }

    /// Batched mirror of [`Self::forward`]. Eager experts are dev/test-only, so
    /// this loops `forward` per row (bit-identical); `route_ws[r]` scales row r.
    fn forward_batch(
        &self,
        xs: &[f32],
        rows: usize,
        dim: usize,
        limit: f32,
        route_ws: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * dim];
        for r in 0..rows {
            let y = self.forward(&xs[r * dim..(r + 1) * dim], dim, limit, Some(route_ws[r]));
            out[r * dim..(r + 1) * dim].copy_from_slice(&y);
        }
        out
    }
}

/// Expert storage: eagerly-dequantized f32 (tiny fixtures / dev) or mmap'd
/// int4_bin (production — weights stay packed on disk, rows dequantized on
/// the fly). The two paths are numerically identical (see `expert_mmap`).
pub enum AnyExpert {
    Eager(Expert),
    Mmap(super::expert_mmap::MmapExpert),
}

impl AnyExpert {
    pub fn forward(&self, x: &[f32], dim: usize, limit: f32, route_w: Option<f32>) -> Vec<f32> {
        match self {
            AnyExpert::Eager(e) => e.forward(x, dim, limit, route_w),
            AnyExpert::Mmap(e) => e.forward(x, dim, limit, route_w),
        }
    }

    /// Batched FFN for `rows` activation rows routed to this expert; each int4
    /// weight row is decoded once and reused across the rows. `route_ws[r]`
    /// scales row r. See [`super::expert_mmap::MmapExpert::forward_batch`].
    pub fn forward_batch(
        &self,
        xs: &[f32],
        rows: usize,
        dim: usize,
        limit: f32,
        route_ws: &[f32],
    ) -> Vec<f32> {
        match self {
            AnyExpert::Eager(e) => e.forward_batch(xs, rows, dim, limit, route_ws),
            AnyExpert::Mmap(e) => e.forward_batch(xs, rows, dim, limit, route_ws),
        }
    }
}

pub struct HcParams {
    pub fn_w: Vec<f32>,  // [(2+hc)*hc, hc*dim]
    pub base: Vec<f32>,  // [(2+hc)*hc]
    pub scale: Vec<f32>, // [3]
}

pub struct Layer {
    pub attn: AttentionLayer,
    pub attn_norm_w: Vec<f32>,
    pub ffn_norm_w: Vec<f32>,
    pub gate: GateWeights,
    pub experts: Vec<AnyExpert>,
    pub shared: AnyExpert,
    pub hc_attn: HcParams,
    pub hc_ffn: HcParams,
}

/// Last-stage head: HC mix -> final RMSNorm -> fp32 logits matmul.
pub struct Head {
    pub norm_w: Vec<f32>,   // [dim]
    pub weight: Vec<f32>,   // [vocab, dim] f32 (reference keeps it fp32)
    pub hc_fn: Vec<f32>,    // [hc, hc*dim]
    pub hc_base: Vec<f32>,  // [hc]
    pub hc_scale: Vec<f32>, // [1]
}

pub struct ModelCfg {
    pub dim: usize,
    pub vocab: usize,
    pub hc: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
    pub topk: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub n_hash_layers: usize,
}

pub struct DsV4Model {
    pub cfg: ModelCfg,
    /// [vocab, dim] — first stage only.
    pub embed: Option<Vec<f32>>,
    pub layers: Vec<Layer>,
    /// last stage only.
    pub head: Option<Head>,
    /// per-layer hidden snapshots from the last forward ([s * hc * dim]
    /// each) — test/debug introspection mirroring the fixture
    /// `int.*.h_after_L{li}` captures.
    pub last_hiddens: Vec<Vec<f32>>,
    /// Debug: per-layer attention output (last prefill token). Bisection only.
    pub last_attn: Vec<Vec<f32>>,
    /// Debug: per-layer moe input (last prefill token). Bisection only.
    pub last_moe_in: Vec<Vec<f32>>,
    /// Global index of this stage's first layer (`layers[i]` is global layer
    /// `first_layer + i`) — maps local layer -> the on-disk expert IR path.
    pub first_layer: usize,
    /// Optional OpenVINO int4 expert backend (opt-in; see [`super::ov_expert`]).
    /// When `Some`, `moe` dispatches experts through it instead of the Rust
    /// int4 mmap kernel.
    pub ov_experts: Option<super::ov_expert::OvExperts>,
}

impl DsV4Model {
    /// pre-mix: coefficients from the flattened copies; returns the reduced
    /// (bf16-rounded) single-stream input + (post, comb) for the post-mix.
    fn hc_pre(&self, copies: &[f32], p: &HcParams) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let _t = std::time::Instant::now();
        let (hc, dim) = (self.cfg.hc, self.cfg.dim);
        let flat = copies; // [hc*dim], f32 (holding bf16 values)
        let ms: f32 = flat.iter().map(|v| v * v).sum::<f32>() / (hc * dim) as f32;
        let r = 1.0 / (ms + self.cfg.norm_eps).sqrt();
        let mix_hc = (2 + hc) * hc;
        let mut mixes = vec![0.0f32; mix_hc];
        linear_f32(flat, &p.fn_w, mix_hc, hc * dim, &mut mixes);
        for m in mixes.iter_mut() {
            *m *= r;
        }
        let c = hc_split_sinkhorn(
            &mixes,
            &p.scale,
            &p.base,
            hc,
            self.cfg.sinkhorn_iters,
            self.cfg.hc_eps,
        );
        let mut y = vec![0.0f32; dim];
        for (d, yv) in y.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for h in 0..hc {
                acc += c.pre[h] * copies[h * dim + d];
            }
            *yv = to_bf16(acc);
        }
        super::math::prof::add(super::math::prof::SINK, _t);
        (y, c.post, c.comb)
    }

    /// post-mix: new_copies[j] = bf16(post[j]*sub + sum_i comb[i][j]*residual[i]).
    fn hc_post(&self, sub: &[f32], residual: &[f32], post: &[f32], comb: &[f32]) -> Vec<f32> {
        let _t = std::time::Instant::now();
        let (hc, dim) = (self.cfg.hc, self.cfg.dim);
        let mut out = vec![0.0f32; hc * dim];
        for j in 0..hc {
            for d in 0..dim {
                let mut acc = post[j] * sub[d];
                for i in 0..hc {
                    acc += comb[i * hc + j] * residual[i * dim + d];
                }
                out[j * dim + d] = to_bf16(acc);
            }
        }
        super::math::prof::add(super::math::prof::SINK, _t);
        out
    }

    /// MoE for one token (x = ffn-normed input, bf16 values). `token_id` is
    /// required by hash-gate layers only.
    fn moe(&self, li: usize, x: &[f32], token_id: Option<usize>) -> Vec<f32> {
        let layer = &self.layers[li];
        let dim = self.cfg.dim;
        let n = layer.gate.weight.len() / dim;
        // scores in f32
        let mut scores = vec![0.0f32; n];
        linear_f32(x, &layer.gate.weight, n, dim, &mut scores);
        for s in scores.iter_mut() {
            *s = softplus(*s).sqrt(); // sqrtsoftplus
        }
        // selection
        let indices: Vec<usize> = if let Some(t2e) = &layer.gate.tid2eid {
            let tid = token_id.expect("hash-gate layer needs the token id");
            (0..self.cfg.topk)
                .map(|k| t2e[tid * self.cfg.topk + k] as usize)
                .collect()
        } else {
            let bias = layer
                .gate
                .bias
                .as_ref()
                .expect("non-hash layer needs gate bias");
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                (scores[b] + bias[b])
                    .partial_cmp(&(scores[a] + bias[a]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            order[..self.cfg.topk].to_vec()
        };
        let mut ws: Vec<f32> = indices.iter().map(|&i| scores[i]).collect();
        let sum: f32 = ws.iter().sum();
        for w in ws.iter_mut() {
            *w = *w / sum * self.cfg.route_scale;
        }
        // accumulate routed + shared in f32.
        //
        // Reference quirk, replicated deliberately: the MoE loop does
        // `y[idx] += expert(x[idx], w[idx, top])` — torch advanced-indexing
        // assignment with a DUPLICATED index is last-write-wins, so when the
        // hash gate emits the same expert twice for a token only the last
        // occurrence's contribution survives.
        let mut y = vec![0.0f32; dim];
        let _te = std::time::Instant::now();
        let glid = self.first_layer + li; // global layer id -> expert IR path
        for (k, &ei) in indices.iter().enumerate() {
            if indices[k + 1..].contains(&ei) {
                continue; // a later duplicate overwrites this contribution
            }
            let e = match &self.ov_experts {
                Some(ov) => ov.routed(glid, ei, x, ws[k]),
                None => layer.experts[ei].forward(x, dim, self.cfg.swiglu_limit, Some(ws[k])),
            };
            for d in 0..dim {
                y[d] += e[d];
            }
        }
        let sh = match &self.ov_experts {
            Some(ov) => ov.shared(glid, x),
            None => layer.shared.forward(x, dim, self.cfg.swiglu_limit, None),
        };
        for d in 0..dim {
            y[d] += sh[d];
        }
        super::math::prof::add(super::math::prof::EXPERTS, _te);
        for v in y.iter_mut() {
            *v = to_bf16(*v);
        }
        y
    }

    /// Batched MoE over `s` tokens (`xins = [s, dim]`, ffn-normed inputs), giving
    /// the same result as [`Self::moe`] on each token — bit-for-bit — but each
    /// expert's int4 weights are decoded ONCE for all its routed tokens instead
    /// of once per token (the prefill throughput win). Two phases keep the
    /// per-token f32 accumulation order (and the hash-gate duplicate
    /// last-write-wins quirk) identical to `moe`: phase 1 computes every
    /// surviving (token, expert) FFN batched per expert; phase 2 sums them per
    /// token in top-k order, then the shared expert. Falls back to the per-token
    /// path when the OpenVINO expert backend is active (no batch entry point).
    fn moe_batch(&self, li: usize, xins: &[f32], s: usize, ids: Option<&[usize]>) -> Vec<f32> {
        let layer = &self.layers[li];
        let dim = self.cfg.dim;
        debug_assert_eq!(xins.len(), s * dim);
        if self.ov_experts.is_some() {
            let mut out = vec![0.0f32; s * dim];
            for t in 0..s {
                let y = self.moe(li, &xins[t * dim..(t + 1) * dim], ids.map(|v| v[t]));
                out[t * dim..(t + 1) * dim].copy_from_slice(&y);
            }
            return out;
        }
        let n = layer.gate.weight.len() / dim;
        // --- per-token routing (identical to `moe`) ---
        let mut tok_idx: Vec<Vec<usize>> = Vec::with_capacity(s);
        let mut tok_ws: Vec<Vec<f32>> = Vec::with_capacity(s);
        for t in 0..s {
            let x = &xins[t * dim..(t + 1) * dim];
            let mut scores = vec![0.0f32; n];
            linear_f32(x, &layer.gate.weight, n, dim, &mut scores);
            for sc in scores.iter_mut() {
                *sc = softplus(*sc).sqrt(); // sqrtsoftplus
            }
            let indices: Vec<usize> = if let Some(t2e) = &layer.gate.tid2eid {
                let tid = ids.expect("hash-gate layer needs the token ids")[t];
                (0..self.cfg.topk)
                    .map(|k| t2e[tid * self.cfg.topk + k] as usize)
                    .collect()
            } else {
                let bias = layer
                    .gate
                    .bias
                    .as_ref()
                    .expect("non-hash layer needs gate bias");
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| {
                    (scores[b] + bias[b])
                        .partial_cmp(&(scores[a] + bias[a]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                order[..self.cfg.topk].to_vec()
            };
            let mut ws: Vec<f32> = indices.iter().map(|&i| scores[i]).collect();
            let sum: f32 = ws.iter().sum();
            for w in ws.iter_mut() {
                *w = *w / sum * self.cfg.route_scale;
            }
            tok_idx.push(indices);
            tok_ws.push(ws);
        }
        // --- phase 1: batched expert compute for surviving (token, expert) ---
        let _te = std::time::Instant::now();
        let mut per_expert: std::collections::HashMap<usize, Vec<(usize, f32)>> =
            std::collections::HashMap::new();
        for t in 0..s {
            let idx = &tok_idx[t];
            for k in 0..idx.len() {
                let ei = idx[k];
                if idx[k + 1..].contains(&ei) {
                    continue; // a later duplicate overwrites this contribution
                }
                per_expert.entry(ei).or_default().push((t, tok_ws[t][k]));
            }
        }
        let mut expert_out: std::collections::HashMap<(usize, usize), Vec<f32>> =
            std::collections::HashMap::new();
        for (&ei, toks) in per_expert.iter() {
            let rows = toks.len();
            let mut xb = vec![0.0f32; rows * dim];
            let mut ws = vec![0.0f32; rows];
            for (r, &(t, w)) in toks.iter().enumerate() {
                xb[r * dim..(r + 1) * dim].copy_from_slice(&xins[t * dim..(t + 1) * dim]);
                ws[r] = w;
            }
            let yb = layer.experts[ei].forward_batch(&xb, rows, dim, self.cfg.swiglu_limit, &ws);
            for (r, &(t, _)) in toks.iter().enumerate() {
                expert_out.insert((ei, t), yb[r * dim..(r + 1) * dim].to_vec());
            }
        }
        // shared expert over ALL tokens (route_w None ≡ scale 1.0, an exact no-op).
        let ones = vec![1.0f32; s];
        let shared = layer
            .shared
            .forward_batch(xins, s, dim, self.cfg.swiglu_limit, &ones);
        super::math::prof::add(super::math::prof::EXPERTS, _te);
        // --- phase 2: per-token accumulate in `moe`'s exact order ---
        let mut out = vec![0.0f32; s * dim];
        for t in 0..s {
            let mut y = vec![0.0f32; dim];
            let idx = &tok_idx[t];
            for k in 0..idx.len() {
                let ei = idx[k];
                if idx[k + 1..].contains(&ei) {
                    continue;
                }
                let e = &expert_out[&(ei, t)];
                for d in 0..dim {
                    y[d] += e[d];
                }
            }
            let sh = &shared[t * dim..(t + 1) * dim];
            for d in 0..dim {
                y[d] += sh[d];
            }
            for v in y.iter_mut() {
                *v = to_bf16(*v);
            }
            out[t * dim..(t + 1) * dim].copy_from_slice(&y);
        }
        out
    }

    fn block_ffn_stage(
        &self,
        li: usize,
        copies: &mut Vec<f32>,
        token_id: Option<usize>,
    ) -> Vec<f32> {
        let layer = &self.layers[li];
        let (y, post, comb) = self.hc_pre(copies, &layer.hc_ffn);
        let mut xin = y;
        rmsnorm(&mut xin, &layer.ffn_norm_w, self.cfg.norm_eps);
        let out = self.moe(li, &xin, token_id);
        *copies = self.hc_post(&out, copies, &post, &comb);
        xin // debug: moe input for bisection
    }

    /// Clear all per-generation state (KV rings, compressor windows,
    /// indexer caches) — call before a new sequence.
    pub fn reset(&mut self) {
        for l in self.layers.iter_mut() {
            l.attn.reset();
        }
        self.last_hiddens.clear();
    }

    /// First-stage entry: ids -> per-token HC copies.
    pub fn embed_ids(&self, ids: &[usize]) -> Vec<Vec<f32>> {
        let (hc, dim) = (self.cfg.hc, self.cfg.dim);
        let embed = self.embed.as_ref().expect("embed only on the first stage");
        ids.iter()
            .map(|&id| {
                let e = &embed[id * dim..(id + 1) * dim];
                let mut c = vec![0.0f32; hc * dim];
                for h in 0..hc {
                    c[h * dim..(h + 1) * dim].copy_from_slice(e);
                }
                c
            })
            .collect()
    }

    /// Run this stage's layers over a prefill sequence of HC copies.
    /// `ids` is required iff the stage contains hash-gate layers.
    pub fn forward_layers_prefill(
        &mut self,
        mut copies: Vec<Vec<f32>>,
        ids: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        let dim = self.cfg.dim;
        let s = copies.len();
        self.last_hiddens.clear();
        self.last_attn.clear();
        for li in 0..self.layers.len() {
            let mut posts = Vec::with_capacity(s);
            let mut combs = Vec::with_capacity(s);
            let mut xin = vec![0.0f32; s * dim];
            for t in 0..s {
                let (mut y, post, comb) = self.hc_pre(&copies[t], &self.layers[li].hc_attn);
                rmsnorm(&mut y, &self.layers[li].attn_norm_w, self.cfg.norm_eps);
                xin[t * dim..(t + 1) * dim].copy_from_slice(&y);
                posts.push(post);
                combs.push(comb);
            }
            let attn_out = self.layers[li].attn.prefill(&xin, s);
            self.last_attn
                .push(attn_out[(s - 1) * dim..s * dim].to_vec());
            for t in 0..s {
                copies[t] = self.hc_post(
                    &attn_out[t * dim..(t + 1) * dim],
                    &copies[t],
                    &posts[t],
                    &combs[t],
                );
            }
            // FFN block with the MoE batched across the whole sequence: each
            // expert's int4 weights are decoded once for all its routed tokens,
            // not once per prompt token. hc_pre / rmsnorm / hc_post stay per
            // token (the Sinkhorn mix is per token, like attention).
            let mut xins = vec![0.0f32; s * dim];
            let mut posts = Vec::with_capacity(s);
            let mut combs = Vec::with_capacity(s);
            for t in 0..s {
                let (y, post, comb) = self.hc_pre(&copies[t], &self.layers[li].hc_ffn);
                let mut xin = y;
                rmsnorm(&mut xin, &self.layers[li].ffn_norm_w, self.cfg.norm_eps);
                xins[t * dim..(t + 1) * dim].copy_from_slice(&xin);
                posts.push(post);
                combs.push(comb);
            }
            let moe_out = self.moe_batch(li, &xins, s, ids);
            for t in 0..s {
                let new = self.hc_post(
                    &moe_out[t * dim..(t + 1) * dim],
                    &copies[t],
                    &posts[t],
                    &combs[t],
                );
                copies[t] = new;
            }
            self.last_moe_in.push(xins[(s - 1) * dim..s * dim].to_vec());
            self.last_hiddens
                .push(copies.iter().flatten().copied().collect());
        }
        copies
    }

    /// Run this stage's layers for one decode token at absolute `pos`.
    pub fn forward_layers_decode(
        &mut self,
        mut copies: Vec<f32>,
        pos: usize,
        id: Option<usize>,
    ) -> Vec<f32> {
        self.last_hiddens.clear();
        for li in 0..self.layers.len() {
            let _tl = std::time::Instant::now();
            let (mut y, post, comb) = self.hc_pre(&copies, &self.layers[li].hc_attn);
            rmsnorm(&mut y, &self.layers[li].attn_norm_w, self.cfg.norm_eps);
            let attn_out = self.layers[li].attn.decode(&y, pos);
            copies = self.hc_post(&attn_out, &copies, &post, &comb);
            let _ = self.block_ffn_stage(li, &mut copies, id);
            super::math::prof::add(super::math::prof::WALL, _tl);
            self.last_hiddens.push(copies.clone());
        }
        super::math::prof::dump("decode");
        super::ov_expert::stats::dump();
        copies
    }

    /// Last-stage exit: logits from the final token's copies (HC head mix ->
    /// final norm -> f32 head matmul; no bf16 rounding on logits).
    pub fn logits(&self, copies: &[f32]) -> Vec<f32> {
        let (hc, dim) = (self.cfg.hc, self.cfg.dim);
        let head = self.head.as_ref().expect("head only on the last stage");
        let ms: f32 = copies.iter().map(|v| v * v).sum::<f32>() / (hc * dim) as f32;
        let r = 1.0 / (ms + self.cfg.norm_eps).sqrt();
        let mut mixes = vec![0.0f32; hc];
        linear_f32(copies, &head.hc_fn, hc, hc * dim, &mut mixes);
        let mut y = vec![0.0f32; dim];
        for h in 0..hc {
            let pre = 1.0 / (1.0 + (-(mixes[h] * r * head.hc_scale[0] + head.hc_base[h])).exp())
                + self.cfg.hc_eps;
            for d in 0..dim {
                y[d] += pre * copies[h * dim + d];
            }
        }
        for v in y.iter_mut() {
            *v = to_bf16(*v); // hc_head returns .to(dtype)
        }
        rmsnorm(&mut y, &head.norm_w, self.cfg.norm_eps);
        let mut logits = vec![0.0f32; self.cfg.vocab];
        linear_f32(&y, &head.weight, self.cfg.vocab, dim, &mut logits);
        logits
    }

    /// Single-stage convenience (first + last): prefill `ids`, return logits.
    pub fn prefill(&mut self, ids: &[usize]) -> Vec<f32> {
        let copies = self.embed_ids(ids);
        let copies = self.forward_layers_prefill(copies, Some(ids));
        self.logits(&copies[ids.len() - 1])
    }

    /// Single-stage convenience: one greedy decode step at absolute `pos`.
    pub fn decode(&mut self, id: usize, pos: usize) -> Vec<f32> {
        let copies = self.embed_ids(&[id]).pop().unwrap();
        let copies = self.forward_layers_decode(copies, pos, Some(id));
        self.logits(&copies)
    }
}
