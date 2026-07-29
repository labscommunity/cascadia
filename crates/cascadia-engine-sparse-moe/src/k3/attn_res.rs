//! Attention Residuals — K3's cross-block residual mixing.
//!
//! Each layer keeps a GROWING stack of per-block residual vectors and mixes
//! over all of them (plus the running prefix sum) with a learned softmax:
//!
//! ```text
//! v       = [block_residual[0..nb] , prefix_sum]        // nb+1 rows of H
//! k       = v * rsqrt(mean(v^2) + eps)                  // RMS-normalise, no weight yet
//! score_w = norm_w * proj_w                             // proj is Linear(H, 1, bias=false)
//! probs   = softmax_j( sum_h k[j][h] * score_w[h] )
//! out     = sum_j probs[j] * v[j]
//! ```
//!
//! Applied TWICE per layer (before attention and before the MLP) with
//! independent (proj, norm) pairs, plus once more at model level after the
//! layer loop. A new entry is appended to the stack every
//! `attn_res_block_size` layers.
//!
//! This is not a carried anchor: because the mixture reads EVERY prior block,
//! a pipeline rank boundary cannot avoid shipping the whole stack — the
//! inter-stage activation has to carry `prefix_sum` plus the block stack
//! (the dsv4 Hyper-Connections situation, not the glm5 boundary-snapping one).

/// Mix one token's block stack with its prefix sum.
///
/// - `prefix_sum`: `[H]`
/// - `blocks`: `nb` rows of `[H]`, laid out contiguously (`nb * H`)
/// - `proj_w`, `norm_w`: `[H]`
/// - `out`: `[H]`
///
/// `nb == 0` is legal (the first layer, before any block boundary): the
/// mixture is then over the single prefix-sum row and reduces to identity.
pub fn apply_attn_res(
    prefix_sum: &[f32],
    blocks: &[f32],
    proj_w: &[f32],
    norm_w: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    let h = prefix_sum.len();
    debug_assert_eq!(proj_w.len(), h);
    debug_assert_eq!(norm_w.len(), h);
    debug_assert_eq!(out.len(), h);
    debug_assert_eq!(blocks.len() % h, 0);
    let nb = blocks.len() / h;

    // score for each of the nb+1 rows; the prefix sum is the last row
    let mut scores = Vec::with_capacity(nb + 1);
    let row = |j: usize| -> &[f32] {
        if j < nb {
            &blocks[j * h..(j + 1) * h]
        } else {
            prefix_sum
        }
    };
    for j in 0..=nb {
        let r = row(j);
        let ms = r.iter().map(|&v| v * v).sum::<f32>() / h as f32;
        let inv = (ms + eps).sqrt().recip();
        let mut s = 0.0f32;
        for i in 0..h {
            s += r[i] * inv * norm_w[i] * proj_w[i];
        }
        scores.push(s);
    }

    // softmax over the rows
    let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut denom = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        denom += *s;
    }

    // weighted sum of the ORIGINAL (un-normalised) rows
    out.fill(0.0);
    for (j, &sc) in scores.iter().enumerate() {
        let w = sc / denom;
        let r = row(j);
        for i in 0..h {
            out[i] += w * r[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stack_is_identity() {
        // With no blocks the softmax has a single row -> weight 1 on prefix_sum.
        let ps = [1.0f32, -2.0, 3.0, 0.5];
        let mut out = [0.0f32; 4];
        apply_attn_res(&ps, &[], &[0.3; 4], &[1.1; 4], 1e-5, &mut out);
        for (o, p) in out.iter().zip(&ps) {
            assert!((o - p).abs() < 1e-6, "got {out:?} want {ps:?}");
        }
    }

    #[test]
    fn output_is_a_convex_combination() {
        // Every output element must lie within the range of the input rows.
        let h = 4;
        let ps = [1.0f32, 1.0, 1.0, 1.0];
        let blocks = [0.0f32, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0];
        let mut out = [0.0f32; 4];
        apply_attn_res(&ps, &blocks, &[0.7; 4], &[1.0; 4], 1e-5, &mut out);
        for i in 0..h {
            assert!(
                (0.0..=2.0).contains(&out[i]),
                "out[{i}] = {} escaped",
                out[i]
            );
        }
    }
}
