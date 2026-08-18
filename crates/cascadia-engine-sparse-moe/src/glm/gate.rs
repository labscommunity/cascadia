//! GLM-5.2 MoE router gate: DeepSeek-V3-style sigmoid scoring with `noaux_tc`
//! selection bias, deterministic top-k, norm-topk, and `routed_scaling_factor`.
//!
//! Contract (mirrors `tools/glm5_ref/kernels_ref.py::moe_gate`, golden-tested):
//! - `scores = sigmoid(logits)`
//! - select `top_k` experts by `(scores + bias)` descending, ties broken toward
//!   the LOWER expert id — a fully deterministic order (torch.topk's tie order
//!   is unspecified, so we define our own and the reference matches it).
//! - weights are the ORIGINAL `scores` of the selected experts (not the biased
//!   selection score), normalised to sum 1 when `top_k > 1`, then `* scale`.
//!
//! GLM's `n_group = 1, topk_group = 1` means there is no group masking — this is
//! a plain top-k over all experts.

/// Selected experts + their dispatch weights, in canonical selection order
/// (selection-score descending, ties broken toward the lower id).
#[derive(Debug, Clone, PartialEq)]
pub struct GateOut {
    pub idx: Vec<u32>,
    pub weight: Vec<f32>,
}

/// One token's gate. `logits` and `bias` are both length `n_experts`.
pub fn moe_gate(
    logits: &[f32],
    bias: &[f32],
    top_k: usize,
    scale: f32,
    norm_topk: bool,
) -> GateOut {
    let e = logits.len();
    assert_eq!(bias.len(), e, "gate: bias/logits length mismatch");
    assert!(top_k <= e, "gate: top_k ({top_k}) > n_experts ({e})");

    // sigmoid scores, then the noaux_tc selection score (scores + bias).
    let scores: Vec<f32> = logits.iter().map(|&l| 1.0 / (1.0 + (-l).exp())).collect();
    let choice: Vec<f32> = scores.iter().zip(bias).map(|(&s, &b)| s + b).collect();

    // Deterministic top-k: expert ids sorted by choice DESC, ties -> lower id.
    let mut order: Vec<u32> = (0..e as u32).collect();
    order.sort_by(|&a, &b| {
        choice[b as usize]
            .partial_cmp(&choice[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    order.truncate(top_k);

    // Weights from the ORIGINAL scores of the selected experts.
    let mut weight: Vec<f32> = order.iter().map(|&i| scores[i as usize]).collect();
    if norm_topk && top_k > 1 {
        let denom: f32 = weight.iter().sum::<f32>() + 1e-20;
        for w in &mut weight {
            *w /= denom;
        }
    }
    for w in &mut weight {
        *w *= scale;
    }

    GateOut { idx: order, weight }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tie_breaks_toward_lower_id_and_normalises() {
        // Two experts with an exact selection-score tie: lower id wins the slot.
        // logits 0 -> sigmoid 0.5 for all; bias picks the ordering.
        let logits = [0.0, 0.0, 0.0, 0.0];
        let bias = [0.1, 0.1, 0.0, 0.0]; // ids 0 and 1 tie at choice=0.6
        let out = moe_gate(&logits, &bias, 2, 2.5, true);
        assert_eq!(out.idx, vec![0, 1]); // tie -> lower ids, in order
                                         // both selected scores are 0.5 -> normalised 0.5 each -> *2.5 = 1.25
        assert!((out.weight[0] - 1.25).abs() < 1e-6);
        assert!((out.weight[1] - 1.25).abs() < 1e-6);
    }

    #[test]
    fn no_norm_keeps_raw_scaled_scores() {
        let logits = [10.0, -10.0]; // sigmoid ~1.0, ~0.0
        let bias = [0.0, 0.0];
        let out = moe_gate(&logits, &bias, 1, 2.0, true); // top_k=1 -> no norm
        assert_eq!(out.idx, vec![0]);
        assert!((out.weight[0] - 2.0 * (1.0 / (1.0 + (-10.0f32).exp()))).abs() < 1e-6);
    }
}
