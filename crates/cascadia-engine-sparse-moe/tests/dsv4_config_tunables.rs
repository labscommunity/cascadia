//! Config-threaded dsv4 tunables: `max_seq` and `experts_mode`.
//!
//! Mirrors `glm5_builder_opts.rs`'s approach: these are pure precedence
//! resolvers (`resolve_dsv4_max_seq`, `resolve_experts_mode`) that take the
//! env read as a parameter rather than touching `std::env` themselves, so
//! precedence is testable without mutating process-global state (`set_var`
//! is `unsafe` under edition 2024, and racy under a parallel test runner
//! regardless — see `glm5_builder_opts.rs`'s header comment for the same
//! rationale). A full `load_staged`/`load_staged_with_experts` round trip
//! against the tiny fixture is exercised separately to prove the override
//! actually reaches the loader, not just the resolver.

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::loader::ExpertsMode;
use cascadia_engine_sparse_moe::dsv4::stage::{resolve_experts_mode, Dsv4Runner};
use cascadia_engine_sparse_moe::engine::resolve_dsv4_max_seq;

fn export_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_export")
}

// ---- resolve_dsv4_max_seq ----

#[test]
fn max_seq_config_wins_over_env() {
    assert_eq!(resolve_dsv4_max_seq(Some(2048), Some("8192")), 2048);
}

#[test]
fn max_seq_falls_back_to_env_when_config_absent() {
    assert_eq!(resolve_dsv4_max_seq(None, Some("8192")), 8192);
}

#[test]
fn max_seq_zero_config_falls_through_to_env() {
    // `Some(0)` must mean "unset", not "zero context" — same polarity as the
    // env-side `.filter(|&n| n > 0)`.
    assert_eq!(resolve_dsv4_max_seq(Some(0), Some("8192")), 8192);
}

#[test]
fn max_seq_absent_config_and_env_falls_through_to_default() {
    assert_eq!(
        resolve_dsv4_max_seq(None, None),
        cascadia_engine_sparse_moe::dsv4::stage::DSV4_DEFAULT_MAX_SEQ
    );
}

#[test]
fn max_seq_zero_config_and_unparseable_env_falls_through_to_default() {
    assert_eq!(
        resolve_dsv4_max_seq(Some(0), Some("not-a-number")),
        cascadia_engine_sparse_moe::dsv4::stage::DSV4_DEFAULT_MAX_SEQ
    );
}

#[test]
fn max_seq_zero_env_falls_through_to_default() {
    assert_eq!(
        resolve_dsv4_max_seq(None, Some("0")),
        cascadia_engine_sparse_moe::dsv4::stage::DSV4_DEFAULT_MAX_SEQ
    );
}

// ---- resolve_experts_mode ----

#[test]
fn experts_override_wins_over_env_and_heuristic() {
    // n_routed_experts = 40 (>32) would push the heuristic to Mmap, and the
    // env says Mmap too — an explicit Eager override must still win.
    assert_eq!(
        resolve_experts_mode(Some("eager"), Some("mmap"), 40),
        ExpertsMode::Eager
    );
    // n_routed_experts = 4 (<=32) would push the heuristic to Eager — an
    // explicit Mmap override must still win.
    assert_eq!(
        resolve_experts_mode(Some("mmap"), None, 4),
        ExpertsMode::Mmap
    );
}

#[test]
fn experts_none_reproduces_env_then_heuristic() {
    // None + env set: env wins over the heuristic.
    assert_eq!(
        resolve_experts_mode(None, Some("mmap"), 4),
        ExpertsMode::Mmap
    );
    assert_eq!(
        resolve_experts_mode(None, Some("eager"), 40),
        ExpertsMode::Eager
    );
    // None + no env: the >32 size heuristic decides.
    assert_eq!(resolve_experts_mode(None, None, 40), ExpertsMode::Mmap);
    assert_eq!(resolve_experts_mode(None, None, 4), ExpertsMode::Eager);
}

#[test]
fn experts_unrecognized_override_falls_back_to_env_then_heuristic() {
    // A garbage override must not panic, and must not silently pin a mode —
    // it defers to env, then the heuristic, exactly like `None`.
    assert_eq!(
        resolve_experts_mode(Some("bogus"), Some("mmap"), 4),
        ExpertsMode::Mmap
    );
    assert_eq!(
        resolve_experts_mode(Some("bogus"), None, 40),
        ExpertsMode::Mmap
    );
    assert_eq!(
        resolve_experts_mode(Some("bogus"), None, 4),
        ExpertsMode::Eager
    );
}

// ---- load_staged / load_staged_with_experts round trip ----

/// `load_staged` (the 14-call-site, signature-frozen entry point) must
/// delegate to `load_staged_with_experts(..., None)` and reproduce identical
/// behavior — the regression guard for every existing caller.
#[test]
fn load_staged_matches_load_staged_with_experts_none() {
    let dir = export_dir();
    let mut a = Dsv4Runner::load_staged(&dir, 64, 0, 1, 0, 0).expect("load_staged");
    let mut b = Dsv4Runner::load_staged_with_experts(&dir, 64, 0, 1, 0, 0, None)
        .expect("load_staged_with_experts(None)");
    assert_eq!(a.hidden_size(), b.hidden_size());
    assert_eq!(a.max_seq(), b.max_seq());

    // Same fixed input token through both must produce identical hidden
    // output — proves the None path resolved to the same experts mode
    // (the tiny fixture's 8 routed experts push the heuristic to Eager
    // either way, so this also guards against a stray override leaking in).
    let ha = a.embed_token(0);
    let hb = b.embed_token(0);
    let out_a = a.forward_layers(ha, 0, Some(0));
    let out_b = b.forward_layers(hb, 0, Some(0));
    assert_eq!(out_a, out_b, "load_staged must reproduce today's behavior");
}

/// The override must reach the loader, not just the resolver: both explicit
/// modes must load the tiny fixture without error (heuristic default for
/// its 8 routed experts is Eager, so `Some("mmap")` here specifically
/// exercises the override winning over what the heuristic would pick).
#[test]
fn explicit_experts_override_loads_via_the_engine_facing_entry_point() {
    let dir = export_dir();
    Dsv4Runner::load_staged_with_experts(&dir, 64, 0, 1, 0, 0, Some("eager"))
        .expect("explicit eager override");
    Dsv4Runner::load_staged_with_experts(&dir, 64, 0, 1, 0, 0, Some("mmap"))
        .expect("explicit mmap override");
}
