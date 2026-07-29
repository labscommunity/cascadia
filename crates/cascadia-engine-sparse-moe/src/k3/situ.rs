//! SiTU — K3's model-global activation (`hidden_act = "situ"`).
//!
//! ```text
//! situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)
//! ```
//!
//! Upstream (`SituAndMul`) splits one `[gate | up]` concatenation in half,
//! computes in f32 and casts back. K3 config: `beta = 4.0`,
//! `linear_beta = 25.0`. Used by routed experts, the shared experts, and the
//! dense layer-0 MLP alike — it is not MoE-specific.

/// One element of the SiTU product. `linear_beta = None` leaves `up` untouched.
#[inline]
pub fn situ_elem(gate: f32, up: f32, beta: f32, linear_beta: Option<f32>) -> f32 {
    let a = beta * (gate / beta).tanh() * (1.0 / (1.0 + (-gate).exp()));
    let u = match linear_beta {
        Some(lb) => lb * (up / lb).tanh(),
        None => up,
    };
    a * u
}

/// `out[i] = situ(gate[i], up[i])`. `gate`, `up` and `out` are all length `d`.
pub fn situ(gate: &[f32], up: &[f32], out: &mut [f32], beta: f32, linear_beta: Option<f32>) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for i in 0..out.len() {
        out[i] = situ_elem(gate[i], up[i], beta, linear_beta);
    }
}

/// In-place over a `[gate | up]` concatenation of width `2*d`; the result lands
/// in the first `d` entries. Mirrors upstream's split-in-half convention.
pub fn situ_split(gate_up: &[f32], out: &mut [f32], beta: f32, linear_beta: Option<f32>) {
    let d = gate_up.len() / 2;
    debug_assert_eq!(out.len(), d);
    let (g, u) = gate_up.split_at(d);
    situ(g, u, out, beta, linear_beta);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_gate_is_zero() {
        // tanh(0) = 0 -> the whole product vanishes regardless of `up`.
        assert_eq!(situ_elem(0.0, 3.0, 4.0, Some(25.0)), 0.0);
    }

    #[test]
    fn linear_beta_saturates_up() {
        // For |up| >> linear_beta the up factor approaches +/- linear_beta.
        let v = situ_elem(100.0, 1.0e6, 4.0, Some(25.0));
        let gate_part = 4.0f32 * (100.0f32 / 4.0).tanh() * (1.0 / (1.0 + (-100.0f32).exp()));
        assert!((v - gate_part * 25.0).abs() < 1e-2, "got {v}");
    }

    #[test]
    fn none_linear_beta_passes_up_through() {
        let g = 1.5f32;
        let a = 4.0 * (g / 4.0).tanh() * (1.0 / (1.0 + (-g).exp()));
        assert!((situ_elem(g, 2.0, 4.0, None) - a * 2.0).abs() < 1e-6);
    }
}
