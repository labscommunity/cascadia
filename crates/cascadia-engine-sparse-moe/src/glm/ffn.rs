//! GLM SwiGLU feed-forward: `down(silu(gate·x) · up·x)`, no bias.
//!
//! The shared building block for routed experts (`moe_intermediate` = 2048),
//! the shared expert (`moe_intermediate · n_shared`), and the dense first-k
//! layers (`intermediate` = 12288). Numeric contract (matches
//! `tools/glm5_ref::swiglu_ref`): bf16 after each linear; the `silu·up` product
//! stays f32 (`silu(g)·u`).

use crate::dsv4::math::linear_bf16_w;

/// SiLU / swish: `x·sigmoid(x)`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU FFN for one token. `wg`/`wu` are `[inter, hidden]`, `wd` is
/// `[hidden, inter]` (bf16 bits). Returns `[hidden]`.
pub fn swiglu(x: &[f32], wg: &[u16], wu: &[u16], wd: &[u16], hidden: usize, inter: usize) -> Vec<f32> {
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
