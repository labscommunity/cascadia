//! GLM SwiGLU feed-forward: `down(silu(gate·x) · up·x)`, no bias.
//!
//! The shared building block for routed experts (`moe_intermediate` = 2048),
//! the shared expert (`moe_intermediate · n_shared`), and the dense first-k
//! layers (`intermediate` = 12288). Numeric contract (matches
//! `tools/glm5_ref::swiglu_ref`): bf16 after each linear; the `silu·up` product
//! stays f32 (`silu(g)·u`).

use crate::dsv4::expert_mmap::MmapExpert;
use crate::dsv4::math::{linear_bf16, linear_bf16_w};

/// SiLU / swish: `x·sigmoid(x)`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU FFN for one token. `wg`/`wu` are `[inter, hidden]`, `wd` is
/// `[hidden, inter]` (bf16 bits). Returns `[hidden]`.
pub fn swiglu(
    x: &[f32],
    wg: &[u16],
    wu: &[u16],
    wd: &[u16],
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), hidden);
    assert_eq!(wg.len(), inter * hidden);
    assert_eq!(wu.len(), inter * hidden);
    assert_eq!(wd.len(), hidden * inter);
    let mut g = vec![0.0f32; inter];
    linear_bf16_w(x, wg, inter, hidden, &mut g);
    let mut u = vec![0.0f32; inter];
    linear_bf16_w(x, wu, inter, hidden, &mut u);
    // fuse silu(gate)·up into g (f32, no bf16 rounding of the product).
    for (gi, &ui) in g.iter_mut().zip(&u) {
        *gi = silu(*gi) * ui;
    }
    let mut y = vec![0.0f32; hidden];
    linear_bf16_w(&g, wd, hidden, inter, &mut y);
    y
}

/// SwiGLU FFN with **f32** weights (int4-dequantized experts). Identical
/// numeric contract to [`swiglu`] — bf16 after each linear, `silu·up` in f32 —
/// but the projection weights are dequantized f32 (int4 values are not exactly
/// bf16-representable, so they are kept f32, not re-rounded). `wg`/`wu` are
/// `[inter, hidden]`, `wd` is `[hidden, inter]`.
pub fn swiglu_f32w(
    x: &[f32],
    wg: &[f32],
    wu: &[f32],
    wd: &[f32],
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), hidden);
    assert_eq!(wg.len(), inter * hidden);
    assert_eq!(wu.len(), inter * hidden);
    assert_eq!(wd.len(), hidden * inter);
    let mut g = vec![0.0f32; inter];
    linear_bf16(x, wg, inter, hidden, &mut g);
    let mut u = vec![0.0f32; inter];
    linear_bf16(x, wu, inter, hidden, &mut u);
    for (gi, &ui) in g.iter_mut().zip(&u) {
        *gi = silu(*gi) * ui;
    }
    let mut y = vec![0.0f32; hidden];
    linear_bf16(&g, wd, hidden, inter, &mut y);
    y
}

/// SwiGLU FFN for an mmap'd int4 expert: the same contract as [`swiglu_f32w`]
/// (bf16 after each GEMV, `silu·up` in f32), but each projection is the fused
/// int4 dequant-dot from disk. Agrees with the eager path within a few bf16 ULP
/// (the fused kernel reorders the f32 summation).
pub fn swiglu_mmap(m: &MmapExpert, x: &[f32]) -> Vec<f32> {
    let mut h = m.gemv_gate(x);
    let u = m.gemv_up(x);
    for (hi, &ui) in h.iter_mut().zip(&u) {
        *hi = silu(*hi) * ui;
    }
    m.gemv_down(&h)
}
