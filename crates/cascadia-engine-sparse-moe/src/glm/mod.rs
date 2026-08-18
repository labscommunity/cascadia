//! GLM-5.2 (`glm5`) Rust shell — DeepSeek-V3-style MLA + DSA sparse-MoE.
//! See `docs/architectures/glm5.md` for the full spec and the reuse map.
//!
//! Shares the pure numeric leaves (bf16 rounding, GEMV, RMSNorm, interleaved
//! rope) with the dsv4 shell via `crate::dsv4::{math, rope}` — additive, no
//! dsv4 edits. GLM differs above the leaves: sigmoid + `noaux_tc` routing (not
//! sqrtsoftplus), plain residual (no Hyper-Connections), raw-position DSA (not
//! block-compressed), interleaved rope with YaRN disabled (`original_seq_len=0`,
//! `base=8e6`).
//!
//! Every primitive is golden-tested 1:1 against the CPU reference in
//! `tools/glm5_ref/` (regenerate fixtures with
//! `tools/glm5_ref/gen_fixtures.py`).
//!
//! Build order (each lands with passing goldens):
//!   1. router gate                                  <- this milestone (M1)
//!   2. rope table + MLA attention + DSA indexer
//!   3. dense/MoE block + full model greedy parity
//!   4. engine/manifest wiring (`shell_backend = "rust_glm"`)

/// True when `name` is set to a value that means "on".
///
/// The `CASCADIA_GLM5_*` switches were presence-only (`var_os(..).is_some()` /
/// `var(..).is_ok()`), so `FLAG=0` — the obvious way to turn one off — switched
/// it ON. That is worst for `CASCADIA_GLM5_OV_EXPERTS`, whose docs say `=1` and
/// whose backend is experimental and measured slower than the Rust kernel: an
/// operator disabling it would enable it. Treat unset, empty, `0`, `false`,
/// `no` and `off` as off; anything else (including `1`) as on.
pub fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim();
            !(v.is_empty()
                || v == "0"
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("no")
                || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod env_flag_tests {
    use super::env_flag;

    #[test]
    fn zero_and_false_mean_off_not_on() {
        let k = "CASCADIA_GLM5_ENV_FLAG_TEST";
        for off in ["0", "false", "FALSE", "no", "off", "", "  "] {
            std::env::set_var(k, off);
            assert!(!env_flag(k), "{off:?} must be off");
        }
        for on in ["1", "true", "yes", "on", "anything"] {
            std::env::set_var(k, on);
            assert!(env_flag(k), "{on:?} must be on");
        }
        std::env::remove_var(k);
        assert!(!env_flag(k), "unset must be off");
    }
}

pub mod attn;
pub mod ffn;
pub mod gate;
pub mod grammar;
pub mod indexer;
pub mod kv_cache;
pub mod loader;
pub mod lookahead;
pub mod model;
pub mod moe;
pub mod mtp;
pub mod ov_expert;
pub mod prof;
pub mod residency;
pub mod rope;
pub mod stage;
