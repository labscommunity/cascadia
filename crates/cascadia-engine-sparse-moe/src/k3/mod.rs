//! Kimi-K3 shell — hybrid KDA (linear attention) + gated NoPE MLA, LatentMoE
//! experts in a 3584-dim latent, SiTU activation, and Attention Residuals.
//!
//! Sibling of [`crate::dsv4`] and [`crate::glm`] rather than a generalisation of
//! them: 69 of K3's 93 layers are linear attention, so there is no MLA-shaped
//! core to share. Pure leaves are reused across families as usual — `dsv4::math`
//! for the bf16/linear/rmsnorm contract and [`crate::glm::gate::moe_gate`] for
//! the router, whose sigmoid + `noaux_tc` + norm-topk semantics K3 matches
//! exactly (K3 just uses `routed_scaling_factor = 1.0`).
//!
//! Every primitive here is golden-tested against `tools/kimi_k3_ref`, which is
//! itself validated against the vendored upstream sources.

pub mod attn;
pub mod attn_res;
pub mod expert_fp4;
pub mod kda;
pub mod loader;
pub mod model;
pub mod moe;
pub mod prof;
pub mod residency;
pub mod situ;
pub mod stage;
