//! Sparse gather attention with per-head sink logits — port of upstream
//! `sparse_attn` (`kernel.py` / `kernels_ref.sparse_attn`).
//!
//! For each (position, head): gather the KV rows named by `topk_idxs`
//! (-1 = masked slot), score q . kv * scale, softmax where the learnable
//! per-head `attn_sink` joins the DENOMINATOR only (an always-present
//! "attend to nothing" option), then prob-weight the same gathered rows.
//! q heads share the single KV latent (V4 is MQA-on-latent: kv rows serve
//! as both K and V).

use super::math::to_bf16;

/// One query position, all heads.
///
/// * `q`:    `[h, d]`
/// * `kv`:   `[n, d]` (row-major; the gather space)
/// * `sink`: `[h]`
/// * `idxs`: `[topk]`, entries in `[0, n)` or `-1`
/// * `out`:  `[h, d]`, bf16-rounded
pub fn sparse_attn_pos(
    q: &[f32],
    kv: &[f32],
    sink: &[f32],
    idxs: &[i32],
    h: usize,
    d: usize,
    scale: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(q.len(), h * d);
    debug_assert_eq!(sink.len(), h);
    debug_assert_eq!(out.len(), h * d);
    let topk = idxs.len();
    let mut scores = vec![f32::NEG_INFINITY; topk];
    for hi in 0..h {
        let qh = &q[hi * d..(hi + 1) * d];
        let mut smax = f32::NEG_INFINITY;
        for (t, &ix) in idxs.iter().enumerate() {
            if ix < 0 {
                scores[t] = f32::NEG_INFINITY;
                continue;
            }
            let row = &kv[ix as usize * d..(ix as usize + 1) * d];
            let mut acc = 0.0f32;
            for k in 0..d {
                acc += qh[k] * row[k];
            }
            let s = acc * scale;
            scores[t] = s;
            smax = smax.max(s);
        }
        let m = smax.max(sink[hi]);
        let mut denom = (sink[hi] - m).exp();
        for s in scores.iter_mut() {
            if s.is_finite() {
                *s = (*s - m).exp();
                denom += *s;
            } else {
                *s = 0.0;
            }
        }
        let oh = &mut out[hi * d..(hi + 1) * d];
        oh.fill(0.0);
        for (t, &ix) in idxs.iter().enumerate() {
            if ix < 0 {
                continue;
            }
            let p = scores[t] / denom;
            let row = &kv[ix as usize * d..(ix as usize + 1) * d];
            for k in 0..d {
                oh[k] += p * row[k];
            }
        }
        for v in oh.iter_mut() {
            *v = to_bf16(*v);
        }
    }
}
