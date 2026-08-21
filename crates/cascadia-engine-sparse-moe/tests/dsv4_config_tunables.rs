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
    // Trimmed: the override arrives from a config file, where trailing
    // whitespace is an operator typo, not a distinct mode.
    assert_eq!(
        resolve_experts_mode(Some("  mmap  "), None, 4),
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

/// `load_staged` (signature-frozen for its 24 call sites, all of them tests
/// now that the builder uses `load_staged_with_experts`) must delegate to
/// `load_staged_with_experts(..., None)` and reproduce identical behavior.
///
/// Rank 0 of 2 over an explicit `[1,3)` deliberately. At `rank=0, total=1,
/// 0, 0` both argument pairs are swap-degenerate — `total.max(1)` /
/// `rank.min(total-1)` collapse `(1,0)` back to `(0,1)`, and `layer_end > 0`
/// is false either way — so a delegation passing `(total, rank)` or
/// `(layer_end, layer_start)` would compile and go unnoticed. These values
/// distinguish both swaps while keeping this the FIRST stage, which is what
/// `embed_token` below requires.
#[test]
fn load_staged_matches_load_staged_with_experts_none() {
    let dir = export_dir();
    let mut a = Dsv4Runner::load_staged(&dir, 64, 0, 2, 1, 3).expect("load_staged");
    let mut b = Dsv4Runner::load_staged_with_experts(&dir, 64, 0, 2, 1, 3, None)
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

/// The override must reach the LOADER, not just the resolver — and the runner
/// must have loaded in the mode asked for. Asserting only that both calls
/// return `Ok` proves nothing: both modes load this fixture fine, so dropping
/// the `experts_override` argument entirely would keep such a test green.
#[test]
fn explicit_experts_override_reaches_the_loader() {
    let dir = export_dir();
    // The fixture's 8 routed experts put the heuristic at Eager, so Mmap here
    // is the override winning over what would otherwise be chosen.
    let mmap = Dsv4Runner::load_staged_with_experts(&dir, 64, 0, 1, 0, 0, Some("mmap"))
        .expect("explicit mmap override");
    assert_eq!(mmap.experts_mode(), ExpertsMode::Mmap);

    let eager = Dsv4Runner::load_staged_with_experts(&dir, 64, 0, 1, 0, 0, Some("eager"))
        .expect("explicit eager override");
    assert_eq!(eager.experts_mode(), ExpertsMode::Eager);

    // No override: the heuristic decides, and it must not silently be Mmap.
    let derived = Dsv4Runner::load_staged(&dir, 64, 0, 1, 0, 0).expect("no override");
    assert_eq!(derived.experts_mode(), ExpertsMode::Eager);
}
