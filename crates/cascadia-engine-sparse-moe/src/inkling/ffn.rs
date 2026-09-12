//! SwiGLU feed-forward for Inkling experts: `down(silu(gate·x) · up·x)`, no
//! bias — exactly the glm kernels (bf16 after each linear, `silu·up` in f32),
//! re-exported so the MoE / dense paths share one numeric contract and the
//! same `AnyExpert` storage dispatch (bf16 goldens, eager int4-dequant f32,
//! mmap'd int4 bins).
//!
//! Plus the checkpoint-layout helper: Inkling stores gate/up fused as `w13`
//! with INTERLEAVED rows (`row 2i = gate_i`, `row 2i+1 = up_i`; transformers'
//! `Interleave` op de-interleaves then chunks), which [`deinterleave_w13`]
//! splits into the `[inter, hidden]` gate / up matrices the kernels take.

pub use crate::glm::ffn::{swiglu, swiglu_f32w, swiglu_mmap};
pub use crate::glm::moe::{AnyExpert, ExpertW};

/// Split an interleaved `w13` matrix (`[2·inter, hidden]` row-major, even rows
/// gate, odd rows up) into `(gate, up)`, each `[inter, hidden]`. Element type
/// generic so it serves bf16 bits (`u16`) and f32 alike.
pub fn deinterleave_w13<T: Copy>(w13: &[T], inter: usize, hidden: usize) -> (Vec<T>, Vec<T>) {
    assert_eq!(
        w13.len(),
        2 * inter * hidden,
        "deinterleave_w13: len != 2 * inter * hidden"
    );
    let mut gate = Vec::with_capacity(inter * hidden);
    let mut up = Vec::with_capacity(inter * hidden);
    for (i, row) in w13.chunks_exact(hidden).enumerate() {
        if i % 2 == 0 {
            gate.extend_from_slice(row);
        } else {
            up.extend_from_slice(row);
        }
    }
    (gate, up)
}

#[cfg(test)]
mod tests {
    use super::deinterleave_w13;

    #[test]
    fn w13_even_rows_are_gate_odd_rows_are_up() {
        // inter 2, hidden 2: rows g0, u0, g1, u1.
        let w13 = [1u16, 1, 2, 2, 3, 3, 4, 4];
        let (g, u) = deinterleave_w13(&w13, 2, 2);
        assert_eq!(g, vec![1, 1, 3, 3]);
        assert_eq!(u, vec![2, 2, 4, 4]);
    }
}
