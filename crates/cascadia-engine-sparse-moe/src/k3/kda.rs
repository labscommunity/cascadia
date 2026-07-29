//! KDA — Kimi Delta Attention, the gated delta rule used by 69 of K3's 93
//! layers. Unlike softmax attention there is no KV cache: each head carries a
//! fixed `[K, V]` state matrix plus a short causal convolution window, so
//! memory is O(1) in context length.
//!
//! Per token, per head (all f32):
//!
//! ```text
//! S = S * exp(g)                       // per-K-dim decay, g in log space
//! S = S + (beta*k) (x) (v - S^T k)     // delta rule
//! o = S^T q
//! ```
//!
//! `q`/`k` arrive L2-normalised and `q` pre-scaled by `K^-0.5`; `beta` is
//! already `sigmoid(b_proj(x))`. The decay `g` comes from [`kda_gate`].
//!
//! Transcribed from `fla::ops::kda` (`naive_recurrent_kda`, `gate.py`) via
//! `tools/kimi_k3_ref`; golden-tested bit-exact against it.

/// KDA decay gate.
///
/// K3 sets `gate_lower_bound = -5.0`, which selects the bounded branch
/// `g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))`. With no lower
/// bound the default is `g = -exp(A_log) * softplus(g_raw + dt_bias)`.
///
/// - `g_raw`, `out`: `[H * D]` for one token
/// - `a_log`: `[H]`, `dt_bias`: `[H * D]`
pub fn kda_gate(
    g_raw: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    lower_bound: Option<f32>,
    heads: usize,
    dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(g_raw.len(), heads * dim);
    debug_assert_eq!(dt_bias.len(), heads * dim);
    debug_assert_eq!(a_log.len(), heads);
    debug_assert_eq!(out.len(), heads * dim);
    for (h, &al) in a_log.iter().enumerate() {
        let a = al.exp();
        for d in 0..dim {
            let i = h * dim + d;
            let g = g_raw[i] + dt_bias[i];
            out[i] = match lower_bound {
                Some(lb) => lb * (1.0 / (1.0 + (-(a * g)).exp())),
                // softplus, guarded for large x
                None => {
                    let sp = if g > 20.0 { g } else { (1.0 + g.exp()).ln() };
                    -a * sp
                }
            };
        }
    }
}

/// Causal depthwise convolution (kernel `k`) followed by SiLU, with a carried
/// window of the `k-1` previous inputs.
///
/// - `x`: `[t * d]`, row-major by position
/// - `weight`: `[d * k]`, row-major by channel
/// - `state`: `[d, k-1]` (channel-major, oldest first); updated in place
/// - `out`: `[t * d]`
///
/// Note the state layout differs from fla's `ShortConvolution`, which caches a
/// `[d, k]` rolling window INCLUDING the current token. Same convolution,
/// different buffer; the reference uses this convention too.
pub fn short_conv(
    x: &[f32],
    weight: &[f32],
    state: &mut [f32],
    d: usize,
    k: usize,
    out: &mut [f32],
) {
    let t = x.len() / d;
    debug_assert_eq!(weight.len(), d * k);
    debug_assert_eq!(state.len(), d * (k - 1));
    debug_assert_eq!(out.len(), t * d);

    for pos in 0..t {
        for c in 0..d {
            let mut acc = 0.0f32;
            // taps 0..k-1 are the oldest..newest inputs ending at `pos`
            for j in 0..k {
                let back = k - 1 - j; // how many positions before `pos`
                let v = if back == 0 {
                    x[pos * d + c]
                } else if pos >= back {
                    x[(pos - back) * d + c]
                } else {
                    // reach into the carried window: index from its end
                    let s = (k - 1) - (back - pos);
                    state[c * (k - 1) + s]
                };
                acc += v * weight[c * k + j];
            }
            out[pos * d + c] = acc * (1.0 / (1.0 + (-acc).exp())); // silu
        }
    }

    // roll the window forward: keep the last k-1 inputs of (state ++ x)
    let keep = k - 1;
    let mut next = vec![0.0f32; keep * d];
    for s in 0..keep {
        // absolute index into the virtual (state ++ x) stream, from its end
        let from_end = keep - s; // 1 = newest
        let idx = t as isize - from_end as isize;
        for c in 0..d {
            next[c * keep + s] = if idx >= 0 {
                x[idx as usize * d + c]
            } else {
                state[c * keep + (keep as isize + idx) as usize]
            };
        }
    }
    state.copy_from_slice(&next);
}

/// One token of the gated delta rule, updating `state` in place.
///
/// - `q`, `k`, `g`: `[heads * kdim]`; `v`, `out`: `[heads * vdim]`
/// - `beta`: `[heads]`
/// - `state`: `[heads * kdim * vdim]`, row-major `[h][kd][vd]`
#[allow(clippy::too_many_arguments)]
pub fn kda_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    heads: usize,
    kdim: usize,
    vdim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(state.len(), heads * kdim * vdim);
    debug_assert_eq!(out.len(), heads * vdim);

    let mut sk = vec![0.0f32; vdim];
    for h in 0..heads {
        let s = &mut state[h * kdim * vdim..(h + 1) * kdim * vdim];
        let qh = &q[h * kdim..(h + 1) * kdim];
        let kh = &k[h * kdim..(h + 1) * kdim];
        let vh = &v[h * vdim..(h + 1) * vdim];
        let gh = &g[h * kdim..(h + 1) * kdim];
        let b = beta[h];

        // S *= exp(g), and accumulate S^T k on the decayed state
        sk.fill(0.0);
        for kd in 0..kdim {
            let decay = gh[kd].exp();
            let kk = kh[kd];
            let row = &mut s[kd * vdim..(kd + 1) * vdim];
            for vd in 0..vdim {
                row[vd] *= decay;
                sk[vd] += kk * row[vd];
            }
        }

        // S += (beta*k) (x) (v - S^T k), then o = S^T q
        let oh = &mut out[h * vdim..(h + 1) * vdim];
        oh.fill(0.0);
        for kd in 0..kdim {
            let bk = b * kh[kd];
            let qq = qh[kd];
            let row = &mut s[kd * vdim..(kd + 1) * vdim];
            for vd in 0..vdim {
                row[vd] += bk * (vh[vd] - sk[vd]);
                oh[vd] += qq * row[vd];
            }
        }
    }
}

/// L2-normalise each head's row in place, then scale by `scale`.
pub fn l2norm_heads(x: &mut [f32], heads: usize, dim: usize, scale: f32) {
    for h in 0..heads {
        let r = &mut x[h * dim..(h + 1) * dim];
        let n = r.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let inv = if n > 0.0 { scale / n } else { 0.0 };
        for v in r.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_lower_bound_is_bounded_and_negative() {
        let mut out = [0.0f32; 4];
        let g = [10.0f32, -10.0, 0.0, 3.0];
        let a = [0.5f32, 0.5];
        let b = [0.0f32; 4];
        kda_gate(&g, &a, &b, Some(-5.0), 2, 2, &mut out);
        for &v in &out {
            assert!((-5.0..=0.0).contains(&v), "gate {v} escaped [-5, 0]");
        }
    }

    #[test]
    fn zero_beta_leaves_state_decayed_only() {
        // beta = 0 removes the delta term, so S just decays and o = S^T q.
        let (h, kd, vd) = (1, 2, 2);
        let mut s = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; vd];
        kda_step(
            &[1.0, 0.0],
            &[1.0, 1.0],
            &[9.0, 9.0],
            &[0.0, 0.0],
            &[0.0],
            &mut s,
            h,
            kd,
            vd,
            &mut out,
        );
        assert_eq!(s, vec![1.0, 2.0, 3.0, 4.0]); // exp(0) = 1, no update
        assert_eq!(out, vec![1.0, 2.0]); // q picks row 0
    }

    #[test]
    fn conv_state_roundtrip_matches_single_pass() {
        // Splitting a sequence and carrying the window must equal one pass.
        let (d, k) = (3usize, 4usize);
        let t = 7usize;
        let x: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.37).sin()).collect();
        let w: Vec<f32> = (0..d * k).map(|i| (i as f32 * 0.11).cos()).collect();

        let mut st_a = vec![0.0f32; d * (k - 1)];
        let mut all = vec![0.0f32; t * d];
        short_conv(&x, &w, &mut st_a, d, k, &mut all);

        let mut st_b = vec![0.0f32; d * (k - 1)];
        let mut p1 = vec![0.0f32; 4 * d];
        let mut p2 = vec![0.0f32; 3 * d];
        short_conv(&x[..4 * d], &w, &mut st_b, d, k, &mut p1);
        short_conv(&x[4 * d..], &w, &mut st_b, d, k, &mut p2);

        for i in 0..4 * d {
            assert!((all[i] - p1[i]).abs() < 1e-6, "head mismatch at {i}");
        }
        for i in 0..3 * d {
            assert!(
                (all[4 * d + i] - p2[i]).abs() < 1e-6,
                "tail mismatch at {i}"
            );
        }
    }
}
