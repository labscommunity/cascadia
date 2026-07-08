//! Hyper-Connections mixing coefficients (`hc_split_sinkhorn`) — exact port
//! of the upstream tilelang kernel via `kernels_ref.hc_split_sinkhorn`.
//!
//! Input `mixes` per token: `[(2 + hc) * hc]` f32 (from
//! `F.linear(x_flat, hc_fn) * rsqrt(mean(x_flat^2) + eps)`).
//! Outputs per token: `pre[hc]`, `post[hc]`, `comb[hc][hc]`:
//!   pre  = sigmoid(m[..hc]      * scale[0] + base[..hc])       + eps
//!   post = 2*sigmoid(m[hc..2hc] * scale[1] + base[hc..2hc])
//!   comb = row-softmax(m[2hc..] * scale[2] + base[2hc..]) + eps,
//!          col-normalise, then (iters-1) x [row-normalise, col-normalise],
//!          every normalise dividing by (sum + eps).

#[inline]
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

pub struct HcCoeffs {
    pub pre: Vec<f32>,  // [hc]
    pub post: Vec<f32>, // [hc]
    pub comb: Vec<f32>, // [hc * hc], row-major
}

/// One token's coefficients. `mixes` is `[(2+hc)*hc]`, `scale` is `[3]`,
/// `base` is `[(2+hc)*hc]`.
pub fn hc_split_sinkhorn(
    mixes: &[f32],
    scale: &[f32],
    base: &[f32],
    hc: usize,
    iters: usize,
    eps: f32,
) -> HcCoeffs {
    debug_assert_eq!(mixes.len(), (2 + hc) * hc);
    debug_assert_eq!(scale.len(), 3);
    debug_assert_eq!(base.len(), (2 + hc) * hc);

    let pre: Vec<f32> = (0..hc)
        .map(|j| sigmoid(mixes[j] * scale[0] + base[j]) + eps)
        .collect();
    let post: Vec<f32> = (0..hc)
        .map(|j| 2.0 * sigmoid(mixes[hc + j] * scale[1] + base[hc + j]))
        .collect();

    let mut comb = vec![0.0f32; hc * hc];
    for j in 0..hc {
        for k in 0..hc {
            let i = j * hc + k;
            comb[i] = mixes[2 * hc + i] * scale[2] + base[2 * hc + i];
        }
    }
    // row softmax + eps
    for j in 0..hc {
        let row = &mut comb[j * hc..(j + 1) * hc];
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v = *v / sum + eps;
        }
    }
    // col normalise, then (iters-1) x [row, col]
    col_norm(&mut comb, hc, eps);
    for _ in 0..iters.saturating_sub(1) {
        row_norm(&mut comb, hc, eps);
        col_norm(&mut comb, hc, eps);
    }
    HcCoeffs { pre, post, comb }
}

fn row_norm(c: &mut [f32], hc: usize, eps: f32) {
    for j in 0..hc {
        let row = &mut c[j * hc..(j + 1) * hc];
        let s: f32 = row.iter().sum();
        for v in row.iter_mut() {
            *v /= s + eps;
        }
    }
}

fn col_norm(c: &mut [f32], hc: usize, eps: f32) {
    for k in 0..hc {
        let mut s = 0.0f32;
        for j in 0..hc {
            s += c[j * hc + k];
        }
        for j in 0..hc {
            c[j * hc + k] /= s + eps;
        }
    }
}
