//! Inkling MoE router — `InklingTopkRouter` (sigmoid gate, biased top-k
//! selection, shared-expert sink normalisation, `route_scale · global_scale`).
//!
//! Contract (`PORT_SPEC.md` §3, mirrors the HF forward):
//!
//! ```text
//!   s        = sigmoid(logits)                       logits = [n_routed + n_shared]
//!   sel      = top_k over (s[..n_routed] + bias)     ties -> lower expert id
//!   den      = Σ_{i ∈ sel} s[i] + Σ_{t < n_shared} s[n_routed + t]
//!   w_i      = s[i] / den · route_scale · global_scale        (i ∈ sel)
//!   gamma_t  = s[n_routed + t] / den · route_scale · global_scale
//! ```
//!
//! HF writes the normalisation as `exp(logsigmoid(l) - logsumexp(logsigmoid(l)))`
//! over the selected + shared logits, which is exactly `s / den`. The bias
//! (`e_score_correction_bias`) steers SELECTION only; the weights use the raw
//! sigmoid scores. `torch.topk(sorted=False)` leaves the selection order
//! unspecified, so we fix it: selection score descending, ties toward the
//! lower id (the glm convention). A NaN selection score (a NaN logit — a
//! corrupted activation) is never selected: it ranks as `-inf`, and the sort
//! uses `f32::total_cmp` so the comparator stays a total order (`sort_by`
//! panics on an inconsistent one).

/// One token's routing: the `top_k` selected routed expert ids, their weights,
/// and the `n_shared` shared-expert gammas — all in canonical order.
#[derive(Debug, Clone, PartialEq)]
pub struct GateOut {
    pub idx: Vec<usize>,
    pub w: Vec<f32>,
    pub gammas: Vec<f32>,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `logits` is `[n_routed + n_shared]` (the router GEMV output, shared rows
/// last), `bias` is `[n_routed]`.
pub fn inkling_gate(
    logits: &[f32],
    bias: &[f32],
    top_k: usize,
    n_shared: usize,
    route_scale: f32,
    global_scale: f32,
) -> GateOut {
    let n_total = logits.len();
    assert!(
        n_total > n_shared,
        "inkling_gate: logits ({n_total}) must cover n_shared ({n_shared}) + routed"
    );
    let n_routed = n_total - n_shared;
    assert_eq!(bias.len(), n_routed, "inkling_gate: bias len != n_routed");
    assert!(
        top_k <= n_routed,
        "inkling_gate: top_k ({top_k}) > n_routed ({n_routed})"
    );

    let scores: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    // Selection score; NaN is unselectable (ranks below every finite score).
    let choice: Vec<f32> = scores[..n_routed]
        .iter()
        .zip(bias)
        .map(|(&s, &b)| {
            let c = s + b;
            if c.is_nan() {
                f32::NEG_INFINITY
            } else {
                c
            }
        })
        .collect();

    // Deterministic top-k: ids by choice DESC (total order), ties -> lower id.
    let mut order: Vec<usize> = (0..n_routed).collect();
    order.sort_by(|&a, &b| choice[b].total_cmp(&choice[a]).then_with(|| a.cmp(&b)));
    order.truncate(top_k);

    // Shared-expert sink: selected + shared scores share one denominator.
    let mut den = 0.0f32;
    for &i in &order {
        den += scores[i];
    }
    for &s in &scores[n_routed..] {
        den += s;
    }
    let scale = |s: f32| s / den * route_scale * global_scale;
    let w: Vec<f32> = order.iter().map(|&i| scale(scores[i])).collect();
    let gammas: Vec<f32> = scores[n_routed..].iter().map(|&s| scale(s)).collect();

    GateOut {
        idx: order,
        w,
        gammas,
    }
}
