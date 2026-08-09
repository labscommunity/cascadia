//! Tiny-model greedy-parity battery for the OV attention prefill route (T3
//! Task 7) — the MERGE GATE deciding whether Tasks 1-6's offload is
//! trustworthy. Every case drives a REAL compiled `attn_ov/` graph (Task 2's
//! exporter, Task 5's `OvAttn`) against the `--tiny` DSA model (Task 3:
//! `index_n_heads=2, index_head_dim=16, index_topk=8`, so `W=8`) and requires
//! the greedy token stream to exactly match the SAME run with `ov_attn` off —
//! AND requires the OV path to have actually fired (`used=true`,
//! `layers_ov>0`), never merely fallen back everywhere while tokens matched
//! trivially (that exact failure shipped once already in this plan; a
//! mutation experiment caught it, not a review).
//!
//! # Required invocation
//!
//! ```text
//! cargo test --release -p cascadia-engine-sparse-moe --features openvino \
//!   --test glm5_attn_ov_parity -- --test-threads=1 --nocapture
//! ```
//!
//! `--release`: see "debug-assert vs release" below. `--test-threads=1`:
//! fixture generation shells out to Python + a nested `cargo test` and is
//! idempotent but not designed for concurrent first-run races; serializing
//! costs nothing on a battery this small.
//!
//! # Decision 1 — missing fixtures FAIL, never skip
//!
//! Fixture-gated tests elsewhere in this repo `eprintln!("SKIP: …")` and
//! `return` when their fixture directory is absent, and it is unverified
//! whether CI ever generates it (carried forward from Task 3's review). This
//! file is the merge gate for the whole feature — a silent skip here would
//! green-light an unvalidated offload path, which is this project's
//! single most common defect (~8 in one cycle per the brief). So
//! [`ensure_fixture`] does not skip: absent a fixture, it GENERATES one via
//! `std::process::Command` (the checked-in `tools/export_glm5.py --tiny` +
//! `tools/glm5_attn_ov.py --export`, plus the rope-table dump the exporter's
//! own drift gate requires), and PANICS with an actionable message — command,
//! exit code, stdout, stderr — if that generation fails for any reason
//! (`python3` missing, `openvino` package missing, exporter refusing the
//! shape). `GLM5_TINY_ATTN_BOUNDARY_DIR` / `GLM5_TINY_ATTN_WINDOWED_DIR` let
//! a CI stage point at a prebuilt fixture instead; if THAT path doesn't
//! check out, it still panics — never a silent fallback to auto-generation
//! or a skip.
//!
//! # Decision 2 — run under `--release`
//!
//! Task 4's `commit_prefill_rows` rounds OV-produced `lc`/`rc` to the bf16
//! grid and `debug_assert`s the round was a no-op — catching a GPU plugin
//! that silently elides the graph's own bf16 `Convert` (observed once,
//! Task 2). `cargo test` builds debug by default, so a defeated contract
//! would abort THAT TEST's thread with a panic instead of producing the
//! greedy-parity mismatch this harness exists to surface. A prior reviewer
//! confirmed the underlying write is recoverable (rows land but `len` isn't
//! advanced, so a later Rust pass overwrites — no KV corruption), so this is
//! a choice about what SIGNAL the harness produces on that failure mode, not
//! about safety. Chosen: `--release`. On CPU (this battery's target — see
//! below) the bf16 contract already passed the device canary AND the
//! exporter's own effective-config readback for both fixture shapes (see the
//! task-7 report), so the debug_assert is not expected to fire here at all;
//! `--release` is chosen for what it means as policy (a defeated contract
//! reads as a token mismatch, the gate's actual job) rather than because the
//! panic is expected to be exercised by this run.
//!
//! # Device
//!
//! CPU only (`ov_attn_device = "CPU"`) — this exercises the same graphs,
//! commit path, and indexer recompute as the GPU route; only the GPU
//! plugin's own behavior is out of reach here, covered separately by the
//! on-node canary (spec Sec9 gate 0 / Task 5).
//!
//! # Fixture shapes — why two
//!
//! The tiny model's `W=8` (`index_topk`), but `tools/glm5_attn_ov.py`'s
//! exporter refuses `p_max==0` ("every window would have zero past
//! capacity"), so `rows_cap` can be AT MOST `W-1=7` for any valid export.
//! That makes `rows==W` structurally ineligible (`rows > rows_cap`) for
//! EVERY possible export, not a reachable "should route" case — see
//! `rows_battery_at_the_w_boundary`. One graph shape cannot serve both "a
//! single window using the whole `rows_cap` budget" and "multiple windows
//! with `base>0`" (`p_max + rows_cap == W` is a hard invariant — Rust's
//! `check_stamp` rejects a stamp that violates it as `CorruptDims`), so two
//! shapes are exported:
//!
//! - **boundary** (`rows_cap=7, p_max=1`): single-window battery at `base=0`.
//! - **windowed** (`rows_cap=2, p_max=6`): everything needing `base>0` —
//!   multi-window chaining, prefix-cache resume, mixed-lineage resume.
#![cfg(feature = "openvino")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use cascadia_engine_sparse_moe::glm::loader::load_model;
use cascadia_engine_sparse_moe::glm::stage::{GlmRunner, StageOpts};
use cascadia_engine_sparse_moe::staged::StagedRunner;

const MAX_SEQ: usize = 32;
const VOCAB: usize = 16; // tiny model's vocab_size (export_glm5.py::export_tiny)
const NUM_LAYERS: u32 = 3; // TINY_INDEXER_NUM_LAYERS

/// A deterministic token stream of length `n`, avoiding token 0 (the tiny
/// model's `eos_token_ids`) so a run of `n_gen` never stops early for a
/// reason unrelated to what's under test.
fn prompt_of(n: usize) -> Vec<u32> {
    (0..n).map(|i| 1 + (i % (VOCAB - 2)) as u32).collect()
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

// ============================================================================
// Fixture generation — see "Decision 1" in the module doc. Never skips.
// ============================================================================

fn repo_root() -> PathBuf {
    // crate root is <repo>/crates/cascadia-engine-sparse-moe
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run_or_fail(cmd: &mut Command, what: &str) {
    let out = cmd.output().unwrap_or_else(|e| {
        panic!(
            "glm5_attn_ov_parity: could not spawn {what} ({cmd:?}): {e}\n\
             This harness generates its own fixtures and refuses to skip on a \
             missing one (T3 Task 7 mandate) — python3 must be on PATH."
        )
    });
    if !out.status.success() {
        panic!(
            "glm5_attn_ov_parity: {what} FAILED (exit {:?})\n--- stdout ---\n{}\n--- stderr ---\n{}",
            out.status.code(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }
}

fn stamp_rows_matches(dir: &Path, rows_cap: usize) -> bool {
    let Ok(bytes) = std::fs::read(dir.join("attn_ov/export_stamp.json")) else {
        return false;
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    v.get("rows").and_then(|x| x.as_u64()) == Some(rows_cap as u64)
}

/// Build (or reuse) a tiny-model `attn_ov` export whose graph's row capacity
/// is exactly `rows_cap`. `env_override`, if set, is used AS-IS (a prebuilt
/// fixture dir for a CI stage that generates fixtures separately) — a bad
/// path there still panics; it never silently falls back to generating one
/// underneath the caller-specified location.
fn ensure_fixture(dir: &Path, rows_cap: usize, env_override: &str) -> PathBuf {
    if let Ok(prebuilt) = std::env::var(env_override) {
        let d = PathBuf::from(prebuilt);
        assert!(
            stamp_rows_matches(&d, rows_cap),
            "glm5_attn_ov_parity: {env_override}={d:?} has no attn_ov/export_stamp.json with \
             rows={rows_cap} — this harness never falls back to auto-generating over an \
             explicit override; fix the env var or unset it"
        );
        return d;
    }

    // Idempotent reuse: an existing, shape-matching export_stamp.json is
    // trusted without re-running Python (fast local iteration). `OvAttn::from_opts`
    // itself independently re-checks the stamp's `manifest_sha256` at USE
    // time, so a stale export still can't silently pass — see ov_attn.rs.
    if stamp_rows_matches(dir, rows_cap) {
        return dir.to_path_buf();
    }

    std::fs::create_dir_all(dir).expect("glm5_attn_ov_parity: create fixture dir");
    if !dir.join("manifest.json").exists() {
        run_or_fail(
            Command::new("python3")
                .arg(repo_root().join("tools/export_glm5.py"))
                .arg("--tiny")
                .arg(dir),
            "tiny model export (python3 tools/export_glm5.py --tiny)",
        );
    }

    // The exporter's drift gate hard-requires an exact rope table dump
    // (dsv4/rope.rs::freqs_dump) matching this model's (qk_rope, W, theta) —
    // a nested `cargo test` invocation is the documented way to produce one
    // (see that module's doc comment).
    let rope_dump = dir.join("rope_freqs_dump.bin");
    run_or_fail(
        Command::new("cargo")
            .current_dir(repo_root())
            .args([
                "test",
                "-p",
                "cascadia-engine-sparse-moe",
                "--lib",
                "dump_real_dims_freqs",
                "--",
                "--nocapture",
            ])
            .env("GLM5_ROPE_DUMP_ROT_DIMS", "4") // tiny model's qk_rope_head_dim
            .env("GLM5_ROPE_DUMP_SEQLEN", "8") // == W (index_topk)
            .env("GLM5_ROPE_DUMP_THETA", "8000000") // tiny model's rope_theta
            .env("GLM5_ROPE_FREQS_DUMP", &rope_dump),
        "rope table dump (cargo test --lib dump_real_dims_freqs)",
    );

    run_or_fail(
        Command::new("python3")
            .arg(repo_root().join("tools/glm5_attn_ov.py"))
            .arg("--export")
            .arg(dir)
            .arg("--rows-cap")
            .arg(rows_cap.to_string())
            .arg("--rope-table-dump")
            .arg(&rope_dump),
        "attn_ov export (python3 tools/glm5_attn_ov.py --export)",
    );

    assert!(
        stamp_rows_matches(dir, rows_cap),
        "glm5_attn_ov_parity: attn_ov export reported success but export_stamp.json at {dir:?} \
         is still missing or has the wrong `rows` — a bug in the exporter or this harness, not \
         a fixture that's merely absent"
    );
    dir.to_path_buf()
}

fn boundary_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/glm5_attn_ov_parity_boundary");
    ensure_fixture(&dir, 7, "GLM5_TINY_ATTN_BOUNDARY_DIR")
}

fn windowed_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/glm5_attn_ov_parity_windowed");
    ensure_fixture(&dir, 2, "GLM5_TINY_ATTN_WINDOWED_DIR")
}

// ============================================================================
// Runner + log-based evidence that the OV path fired (see the mandatory
// assertion in the task-7 brief). `GlmRunner` exposes no public accessor for
// a call's per-window tally, so `event="ov_attn_prefill"` (emitted once per
// `forward_layers_batch`, unconditionally whenever `ov_attn` is configured —
// stage.rs) is the only observable from outside the crate; same pattern as
// `glm5_prefill_pipeline.rs`'s existing `capture_logs`/`log.contains(..)`.
// ============================================================================

fn runner(dir: &Path, ov_on: bool, min_rows: u32, prefix_cache: bool) -> GlmRunner {
    GlmRunner::load_staged(
        dir,
        MAX_SEQ,
        0,
        1,
        0,
        0,
        StageOpts {
            ov_attn: Some(ov_on),
            ov_attn_device: Some("CPU".to_string()),
            ov_attn_min_rows: Some(min_rows), // tiny W=8 << production default 64
            prefix_cache_depth: if prefix_cache { Some(4) } else { None },
            ..Default::default()
        },
    )
    .expect("load tiny attn-ov fixture")
}

/// Capture everything `tracing` emits while `f` runs and return its result
/// alongside the captured text.
fn capture_logs<F: FnOnce() -> R, R>(f: F) -> (R, String) {
    use std::io;
    use std::sync::Mutex as StdMutex;

    #[derive(Clone)]
    struct BufWriter(Arc<StdMutex<Vec<u8>>>);
    impl io::Write for BufWriter {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let buf = Arc::new(StdMutex::new(Vec::<u8>::new()));
    let writer = BufWriter(buf.clone());
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(move || writer.clone())
        .with_ansi(false)
        .finish();
    let r = tracing::subscriber::with_default(subscriber, f);
    let bytes = buf.lock().unwrap().clone();
    (r, String::from_utf8_lossy(&bytes).into_owned())
}

/// Every `event="ov_attn_prefill"` line, in call order.
fn ov_prefill_lines(log: &str) -> Vec<&str> {
    log.lines()
        .filter(|l| l.contains("event=\"ov_attn_prefill\""))
        .collect()
}

/// The mandatory assertion (task-7 brief): the OV path ACTUALLY fired for
/// every one of this call's layers, not merely fell back while tokens
/// happened to match.
fn assert_used_all_layers(line: &str) {
    assert!(line.contains("used=true"), "expected used=true: {line}");
    assert!(
        line.contains(&format!("layers_ov={NUM_LAYERS}")),
        "expected layers_ov={NUM_LAYERS} (every layer routed): {line}"
    );
    assert!(
        line.contains("layers_rust=0"),
        "expected layers_rust=0: {line}"
    );
    assert!(
        line.contains("skipped_reason=\"none\""),
        "expected skipped_reason=\"none\": {line}"
    );
}

fn assert_fell_back(line: &str, reason: &str) {
    assert!(line.contains("used=false"), "expected used=false: {line}");
    assert!(
        line.contains(&format!("skipped_reason=\"{reason}\"")),
        "expected skipped_reason=\"{reason}\": {line}"
    );
}

/// Drive `prompt` through `forward_layers_batch` in `chunk`-row windows,
/// simulating the wire's real windowing (the single-box `generate` path
/// delivers a whole prompt in one call, which is not representative of the
/// production ≤256-row batching this seam is actually shaped for). Returns
/// `[rows, hidden]`, position order — bit-identical to a single one-shot
/// call regardless of chunking (each layer's KV write is sequential in row
/// order either way; Task 6 proved batch-vs-per-token identity, and this is
/// the same argument one level up).
fn drive_prefill_chunks(r: &mut GlmRunner, prompt: &[u32], chunk: usize) -> Vec<f32> {
    let hsz = r.hidden_size();
    let rows = prompt.len();
    let mut batch = vec![0.0f32; rows * hsz];
    for (i, &t) in prompt.iter().enumerate() {
        batch[i * hsz..(i + 1) * hsz].copy_from_slice(&r.embed_token(t));
    }
    let mut out = vec![0.0f32; rows * hsz];
    let mut base = 0usize;
    while base < rows {
        let n = chunk.min(rows - base);
        let h = r.forward_layers_batch(batch[base * hsz..(base + n) * hsz].to_vec(), base, n);
        out[base * hsz..(base + n) * hsz].copy_from_slice(&h);
        base += n;
    }
    out
}

/// Greedy decode `n_gen` tokens starting from pre-final-norm hidden `hlast`
/// at absolute position `pos` (the position `hlast` already advanced the KV
/// to). Always the plain Rust per-token path — `forward_layers`/`embed_token`
/// never consult `ov_attn` (see `stage.rs`), which is what makes this the
/// right tool for "continue on Rust after an OV/restore prefix".
fn greedy_decode_from(r: &mut GlmRunner, hlast: &[f32], pos: usize, n_gen: usize) -> Vec<u32> {
    let mut logits = r.head_logits(hlast);
    let mut out = Vec::with_capacity(n_gen);
    let mut p = pos;
    for _ in 0..n_gen {
        let tok = argmax(&logits);
        out.push(tok);
        let h = r.embed_token(tok);
        let h = r.forward_layers(h, p, Some(tok));
        logits = r.head_logits(&h);
        p += 1;
    }
    out
}

// ============================================================================
// The battery
// ============================================================================

/// `rows ∈ {1, W−1, W, W+1}` at `base=0` (spec §9.1a). `W=8`; the exporter's
/// own `p_max>=1` requirement (module doc) makes `rows_cap<=W-1=7` for ANY
/// valid export, so `rows==W` and `rows==W+1` are BOTH structurally
/// ineligible — negative controls proving the eligibility gate correctly
/// excludes an over-capacity window, not cases where OV is expected to fire.
/// I could not construct a variant where `rows==W` routes to OV: no export
/// shape satisfies it (see the module doc's "why two" section), so this is
/// reported rather than papered over with a weaker assertion.
#[test]
fn rows_battery_at_the_w_boundary() {
    let dir = boundary_dir();
    let n_gen = 3usize;
    for rows in [1usize, 7, 8, 9] {
        let prompt = prompt_of(rows);

        let want = {
            let mut off = runner(&dir, false, 1, false);
            off.generate_argmax(&prompt, n_gen)
        };

        let (got, log) = capture_logs(|| {
            let mut on = runner(&dir, true, 1, false);
            on.generate_argmax(&prompt, n_gen)
        });

        assert_eq!(
            got, want,
            "rows={rows}: greedy tokens diverged with ov_attn on vs off"
        );

        let lines = ov_prefill_lines(&log);
        assert_eq!(
            lines.len(),
            1,
            "rows={rows}: expected exactly one batched-prefill call; log:\n{log}"
        );
        if rows <= 7 {
            assert_used_all_layers(lines[0]);
        } else {
            assert_fell_back(lines[0], "window_too_large");
        }
    }
}

/// Multi-window chaining (spec §9.1a: "at least 3 windows") AND the first
/// decode token after an OV prefill fills the FULL `W` budget (`n = W+1` —
/// the first position where the DSA indexer's real top-k pruning runs on
/// keys the OV route append via `indexer_append_normed`, not the trivial
/// "everything selected" case every `base+rows<=W` window gets). Combined
/// into one test deliberately: reaching `n=W` entirely via OV REQUIRES
/// multiple windows here (`rows==W` is not a single-window-reachable case —
/// see `rows_battery_at_the_w_boundary`), so this is the natural way to
/// reach `n=W+1`'s decode step with an all-OV lineage.
#[test]
fn multiwindow_chunks_then_first_decode_past_the_budget() {
    let dir = windowed_dir();
    let prompt = prompt_of(8); // == W: fills the OV budget exactly, 4 windows of 2
    let chunk = 2usize; // == windowed export's rows_cap
    let n_gen = 2usize; // n = W+1, W+2

    let want = {
        let mut off = runner(&dir, false, 1, false);
        let h = drive_prefill_chunks(&mut off, &prompt, chunk);
        let hlast = h[(prompt.len() - 1) * off.hidden_size()..].to_vec();
        greedy_decode_from(&mut off, &hlast, prompt.len(), n_gen)
    };

    let (got, log) = capture_logs(|| {
        let mut on = runner(&dir, true, 1, false);
        let h = drive_prefill_chunks(&mut on, &prompt, chunk);
        let hlast = h[(prompt.len() - 1) * on.hidden_size()..].to_vec();
        greedy_decode_from(&mut on, &hlast, prompt.len(), n_gen)
    });

    assert_eq!(
        got, want,
        "post-multiwindow decode diverged with ov_attn on vs off"
    );

    let lines = ov_prefill_lines(&log);
    assert_eq!(
        lines.len(),
        4,
        "expected 4 chunked windows (0,2,4,6); log:\n{log}"
    );
    for (i, line) in lines.iter().enumerate() {
        assert_used_all_layers(line);
        assert!(
            line.contains(&format!("base={}", i * chunk)),
            "window {i}: {line}"
        );
        assert!(
            line.contains(&format!("rows={chunk}")),
            "window {i}: {line}"
        );
    }
}

/// A prefix-cache resume: cache a prompt prefix's KV, restore it into a
/// freshly-`reset` runner, and prefill the SUFFIX. The suffix window's
/// `base=6` comes from `restore_prefix`, not natural sequential growth —
/// this proves OV eligibility (`self.attn.len()`) and the past-KV read
/// (`past_rows()`) are keyed off the LAYER's actual restored state, not some
/// runner-local "always grown sequentially" assumption.
#[test]
fn prefix_cache_resume_then_ov_serves_the_suffix() {
    let dir = windowed_dir();
    let prompt = prompt_of(8);
    let prefix_len = 6usize; // > rows_cap(2): this window falls back on its own
    let n_gen = 2usize;
    const KEY: u64 = 1;

    let resumed_run = |ov_on: bool| {
        let mut r = runner(&dir, ov_on, 1, true);
        let (prefix, suffix) = prompt.split_at(prefix_len);
        let h0 = drive_prefill_chunks(&mut r, prefix, prefix_len); // one big window, falls back
        let _ = h0;
        r.cache_prefix(KEY);
        r.reset();
        let restored = r.restore_prefix(KEY);
        assert_eq!(restored, Some(prefix_len), "prefix restore length");

        let hsz = r.hidden_size();
        let mut sb = vec![0.0f32; suffix.len() * hsz];
        for (i, &t) in suffix.iter().enumerate() {
            sb[i * hsz..(i + 1) * hsz].copy_from_slice(&r.embed_token(t));
        }
        let h = r.forward_layers_batch(sb, prefix_len, suffix.len());
        let hlast = h[(suffix.len() - 1) * hsz..].to_vec();
        greedy_decode_from(&mut r, &hlast, prompt.len(), n_gen)
    };

    let want = resumed_run(false);
    let (got, log) = capture_logs(|| resumed_run(true));

    assert_eq!(
        got, want,
        "resumed+OV-suffix generation diverged with ov_attn on vs off"
    );

    let lines = ov_prefill_lines(&log);
    assert_eq!(
        lines.len(),
        2,
        "expected the >rows_cap prefix window (falls back) + the resumed suffix window; log:\n{log}"
    );
    assert_fell_back(lines[0], "window_too_large");
    assert_used_all_layers(lines[1]);
    assert!(lines[1].contains("base=6"), "{}", lines[1]);
}

/// Mixed lineage (spec §5): KV rows produced by the OV route, snapshotted
/// (`cache_prefix`) and restored (`restore_prefix`) into a freshly-`reset`
/// runner, then continued on the plain Rust decode path
/// (`forward_layers`/`embed_token` never consult `ov_attn` — stage.rs).
/// Proves `commit_prefill_rows`'s bf16-grid write plus the DSA indexer's
/// key append are both faithfully captured by `AttentionLayer::snapshot`/
/// `restore` (they are NOT special-cased for OV-origin rows — see attn.rs)
/// and correctly consumed by ordinary Rust attention afterward.
#[test]
fn resume_after_ov_prefill_is_consumed_correctly_by_rust_decode() {
    let dir = windowed_dir();
    let prefix = prompt_of(4); // 2 OV windows of 2 rows each: base 0, base 2
    let n_gen = 3usize;
    const KEY: u64 = 2;

    let want = load_model(&dir, MAX_SEQ)
        .expect("load_model")
        .generate(&prefix, n_gen);

    let (got, log) = capture_logs(|| {
        let mut r = runner(&dir, true, 1, true);
        let h = drive_prefill_chunks(&mut r, &prefix, 2);
        r.cache_prefix(KEY);
        r.reset();
        let restored = r.restore_prefix(KEY);
        assert_eq!(restored, Some(prefix.len()), "prefix restore length");

        let hsz = r.hidden_size();
        let hlast = h[(prefix.len() - 1) * hsz..].to_vec();
        greedy_decode_from(&mut r, &hlast, prefix.len(), n_gen)
    });

    assert_eq!(
        got, want,
        "mixed-lineage (OV-prefill -> snapshot -> restore -> Rust decode) diverged from the \
         dense-Rust reference"
    );

    let lines = ov_prefill_lines(&log);
    assert_eq!(
        lines.len(),
        2,
        "expected 2 OV windows (base 0, base 2) before the restore; log:\n{log}"
    );
    for line in &lines {
        assert_used_all_layers(line);
    }
}
