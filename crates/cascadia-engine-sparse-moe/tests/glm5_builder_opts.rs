//! Config-threaded glm5 tunables.
//!
//! Hosts that load the engine in-process cannot pass these through the
//! environment (`set_var` is `unsafe` under edition 2024 and is process-global),
//! so the operator-facing knobs travel as config instead, with env as the
//! fallback when a field is left `None`.
//!
//! These tests assert the values REACH the resolved `StageOpts` — not merely
//! that the fields exist. A field that parses and then goes nowhere is the exact
//! failure this plumbing exists to prevent, and a "defaults to None" test cannot
//! catch it.
//!
//! The env-default assertions compare against `StageOpts::from_env()` rather
//! than hardcoded booleans, so they hold regardless of what
//! `CASCADIA_GLM5_*` happens to be set in the runner's environment.

use cascadia_engine_sparse_moe::glm::stage::StageOpts;
use cascadia_engine_sparse_moe::SparseMoEBuilderConfig;

#[test]
fn unset_fields_fall_back_to_the_env_defaults() {
    let env = StageOpts::from_env();
    let opts = StageOpts::resolve(None, None, None, None, None, None, None, None, None, None);
    assert_eq!(
        opts.bf16_head, env.bf16_head,
        "bf16_head must follow env when unset"
    );
    assert_eq!(
        opts.bf16_kv, env.bf16_kv,
        "bf16_kv must follow env when unset"
    );
    assert_eq!(opts.experts_mode, None);
    assert_eq!(opts.lookahead, None);
    assert_eq!(opts.prefix_cache_depth, None);
}

#[test]
fn f32_head_is_the_opt_out_not_the_default() {
    // bf16 head is the production default (~1.2x decode); CASCADIA_GLM5_F32_HEAD
    // is the opt-OUT. A `bf16_head` field defaulting to false would be a silent
    // perf regression on every deployment, so the polarity is pinned here.
    assert!(
        !StageOpts::resolve(
            Some(true),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )
        .bf16_head
    );
    assert!(
        StageOpts::resolve(
            Some(false),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )
        .bf16_head
    );
}

#[test]
fn explicit_config_overrides_the_env_default() {
    let opts = StageOpts::resolve(
        Some(true),
        Some(true),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    assert!(!opts.bf16_head, "f32_head = true means bf16_head = false");
    assert!(opts.bf16_kv);
}

#[test]
fn load_staged_knobs_reach_stage_opts() {
    // The three knobs whose env reads live INSIDE load_staged. If these do not
    // arrive on StageOpts they are dead config.
    let opts = StageOpts::resolve(
        None,
        None,
        Some("eager".to_string()),
        Some(true),
        Some(8),
        None,
        None,
        None,
        None,
        None,
    );
    assert_eq!(opts.experts_mode.as_deref(), Some("eager"));
    assert_eq!(opts.lookahead, Some(true));
    assert_eq!(opts.prefix_cache_depth, Some(8));
}

#[test]
fn ov_attn_knobs_reach_stage_opts() {
    // `ov_attn_min_rows` in particular: the tiny parity model has W=8, so a
    // config that could not lower the 64 floor would leave the parity harness
    // exercising nothing.
    let opts = StageOpts::resolve(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(true),
        Some("CPU".to_string()),
        Some(8),
    );
    assert_eq!(opts.ov_attn, Some(true));
    assert_eq!(opts.ov_attn_device.as_deref(), Some("CPU"));
    assert_eq!(opts.ov_attn_min_rows, Some(8));
}

#[test]
fn config_fields_default_to_none() {
    let cfg = SparseMoEBuilderConfig::new("/tmp/glm5", "CPU");
    assert_eq!(cfg.max_seq, None);
    assert_eq!(cfg.prefix_cache_depth, None);
    assert_eq!(cfg.bf16_kv, None);
    assert_eq!(cfg.f32_head, None);
    assert_eq!(cfg.lookahead, None);
    assert_eq!(cfg.experts_mode, None);
    assert_eq!(cfg.ov_attn, None);
    assert_eq!(cfg.ov_attn_device, None);
    assert_eq!(cfg.ov_attn_min_rows, None);
    // NOTE: there is deliberately no `cfg.profile`. CASCADIA_GLM5_PROFILE is read
    // from a OnceLock process global inside the decode loop, which no builder
    // config can reach, so it stays env-only rather than becoming a field that
    // parses and silently does nothing.
}
