//! Batched-prefill across the pipeline over the REAL loopback transport, for
//! M ∈ {1,2,4} ranks (M=4 exercises middle-relay ranks). The whole prompt is
//! pushed through each rank's `forward_layers_batch` (per-position attention +
//! batch-union MoE) in one shot and relayed downstream as a batch frame; the
//! last rank's head logits at the final position must equal the single-process
//! `GlmModel::prefill`. This proves the distributed prefill carries batch-union
//! correctly — the throughput path a real N-node run uses.
//!
//! Requires the 8-layer fixture (`glm5_export_ml`); see glm5_pipeline_ml.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use cascadia_engine_sparse_moe::dist::{
    recv_forward_batch_body_server, recv_kind_server, send_forward_batch_prefill,
    send_forward_batch_prefill_nosample, FrameKind,
};
use cascadia_engine_sparse_moe::glm::loader::load_model;
use cascadia_engine_sparse_moe::glm::stage::{GlmRunner, StageOpts};
use cascadia_engine_sparse_moe::staged::StagedRunner;
use cascadia_engine_sparse_moe::SamplingConfig;
use cascadia_transport::{ActivationClient, ActivationServer};
use tokio::sync::Mutex;

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Push `prompt` through an `m`-rank pipeline as a single batched-prefill and
/// return the last rank's head logits at the final prompt position.
async fn pipeline_prefill(dir: &Path, max_seq: usize, m: usize, prompt: &[u32]) -> Vec<f32> {
    let rows = prompt.len();
    let mut ranks: Vec<GlmRunner> = (0..m)
        .map(|r| {
            GlmRunner::load_staged(dir, max_seq, r as u32, m as u32, 0, 0, Default::default())
                .expect("load rank")
        })
        .collect();
    for r in &mut ranks {
        r.reset();
    }
    let hsz = ranks[0].hidden_size() as u32;

    // m-1 loopback links: rank i (client) -> rank i+1 (server).
    let mut clients = Vec::new();
    let mut servers = Vec::new();
    for _ in 0..m.saturating_sub(1) {
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let server = Arc::new(Mutex::new(server));
        let sc = server.clone();
        let atask = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client
            .connect_with_timeout(Duration::from_secs(5))
            .await
            .unwrap();
        atask.await.unwrap();
        clients.push(Arc::new(Mutex::new(client)));
        servers.push(server);
    }

    // rank 0: embed the whole prompt, run its layer slice as one batch.
    let mut batch = vec![0.0f32; rows * hsz as usize];
    for (r, &t) in prompt.iter().enumerate() {
        batch[r * hsz as usize..(r + 1) * hsz as usize].copy_from_slice(&ranks[0].embed_token(t));
    }
    let mut h = ranks[0].forward_layers_batch(batch, 0, rows);

    // relay the batch hop by hop; each rank runs its slice batched.
    let cfg = SamplingConfig::default();
    for i in 0..m - 1 {
        let client = clients[i].clone();
        let cfg2 = cfg.clone();
        let hsend = h;
        let rows32 = rows as u32;
        let send = tokio::spawn(async move {
            send_forward_batch_prefill(&client, 0, rows32, &cfg2, &hsend, [1, rows32, hsz])
                .await
                .unwrap();
        });
        let k = recv_kind_server(&servers[i]).await.unwrap();
        assert_eq!(
            k,
            Some(FrameKind::ForwardBatchPrefill),
            "relay {i}: expected ForwardBatchPrefill"
        );
        let (_start, count, _s, hw, _shape) =
            recv_forward_batch_body_server(&servers[i]).await.unwrap();
        assert_eq!(count as usize, rows);
        send.await.unwrap();
        h = ranks[i + 1].forward_layers_batch(hw, 0, rows);
    }

    // last rank: head logits at the final position.
    let last = (rows - 1) * hsz as usize;
    ranks[m - 1].head_logits(&h[last..last + hsz as usize])
}

/// Push `prompt` through an `m`-rank pipeline as a SEQUENCE of `window`-row
/// frames — the shape a prompt longer than `MAX_BATCH_COUNT` takes — and return
/// the last rank's head logits at the final position. Every window but the last
/// travels as `ForwardBatchPrefillNoSample`; `base` accumulates across them so
/// each rank's KV grows exactly as it would token-by-token.
async fn pipeline_prefill_windowed(
    dir: &Path,
    max_seq: usize,
    m: usize,
    prompt: &[u32],
    window: usize,
) -> Vec<f32> {
    let mut ranks: Vec<GlmRunner> = (0..m)
        .map(|r| {
            GlmRunner::load_staged(dir, max_seq, r as u32, m as u32, 0, 0, Default::default())
                .expect("load rank")
        })
        .collect();
    for r in &mut ranks {
        r.reset();
    }
    let hsz = ranks[0].hidden_size() as u32;

    let mut clients = Vec::new();
    let mut servers = Vec::new();
    for _ in 0..m.saturating_sub(1) {
        let mut server = ActivationServer::new("127.0.0.1", 0);
        server.start().await.unwrap();
        let port = server.port();
        let server = Arc::new(Mutex::new(server));
        let sc = server.clone();
        let atask = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
        let mut client = ActivationClient::new("127.0.0.1", port);
        client
            .connect_with_timeout(Duration::from_secs(5))
            .await
            .unwrap();
        atask.await.unwrap();
        clients.push(Arc::new(Mutex::new(client)));
        servers.push(server);
    }

    let cfg = SamplingConfig::default();
    let mut base = 0usize;
    let mut out = Vec::new();
    for chunk in prompt.chunks(window) {
        let rows = chunk.len();
        let is_last = base + rows == prompt.len();

        let mut batch = vec![0.0f32; rows * hsz as usize];
        for (r, &t) in chunk.iter().enumerate() {
            batch[r * hsz as usize..(r + 1) * hsz as usize]
                .copy_from_slice(&ranks[0].embed_token(t));
        }
        let mut h = ranks[0].forward_layers_batch(batch, base, rows);

        for i in 0..m - 1 {
            let client = clients[i].clone();
            let cfg2 = cfg.clone();
            let hsend = h;
            let rows32 = rows as u32;
            let base32 = base as u32;
            let send = tokio::spawn(async move {
                let shape = [1, rows32, hsz];
                if is_last {
                    send_forward_batch_prefill(&client, base32, rows32, &cfg2, &hsend, shape)
                        .await
                        .unwrap();
                } else {
                    send_forward_batch_prefill_nosample(
                        &client, base32, rows32, &cfg2, &hsend, shape,
                    )
                    .await
                    .unwrap();
                }
            });
            let k = recv_kind_server(&servers[i]).await.unwrap();
            let want = if is_last {
                FrameKind::ForwardBatchPrefill
            } else {
                FrameKind::ForwardBatchPrefillNoSample
            };
            assert_eq!(k, Some(want), "relay {i} at base {base}: wrong frame kind");
            let (start, count, _s, hw, _shape) =
                recv_forward_batch_body_server(&servers[i]).await.unwrap();
            assert_eq!(start as usize, base, "relay {i}: wrong window base");
            assert_eq!(count as usize, rows);
            send.await.unwrap();
            h = ranks[i + 1].forward_layers_batch(hw, base, rows);
        }

        if is_last {
            let last = (rows - 1) * hsz as usize;
            out = ranks[m - 1].head_logits(&h[last..last + hsz as usize]);
        }
        base += rows;
    }
    out
}

/// A prompt longer than `MAX_BATCH_COUNT` (256) cannot ride one frame, so it is
/// prefilled as a sequence of windows. This pins the part that can silently
/// corrupt output: KV continuity across windows on EVERY rank, over the real
/// frames, for M ∈ {1,2,4}. A dropped window, a wrong `base`, or a rank that
/// fails to accumulate would all change the final logits.
///
/// Deliberately NOT covered here, because neither is reachable from a test:
/// `PipelineEngine`'s own window-loop arithmetic (its constructor is private and
/// rank 0 needs a tokenizer the fixture doesn't ship), and the "no mid-prompt row
/// enters the last rank's rep-penalty history" property (`last_rank_history` is
/// private). The latter is structurally implied by the NoSample frame kind,
/// which this test does assert is the one carrying every non-final window.
#[tokio::test]
async fn glm5_multi_window_prefill_matches_single_process() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export_ml");
    // 300 rows over a 16-token vocab: two windows at the real 256 cap.
    let prompt: Vec<u32> = (0..300u32).map(|i| i % 16).collect();
    let max_seq = 320usize;
    let window = 256usize;
    assert!(
        prompt.len() > window,
        "prompt must span more than one window or this test proves nothing"
    );

    let want = load_model(&dir, max_seq)
        .expect("load_model")
        .prefill(&prompt);
    let want_tok = argmax(&want);

    for m in [1usize, 2, 4] {
        let got = pipeline_prefill_windowed(&dir, max_seq, m, &prompt, window).await;
        assert_eq!(got.len(), want.len(), "M={m}: logit length mismatch");
        assert_eq!(
            got, want,
            "M={m}: windowed prefill diverged from single-process prefill"
        );
        assert_eq!(argmax(&got), want_tok, "M={m}: first-token argmax mismatch");
    }
}

#[tokio::test]
async fn glm5_batched_prefill_pipeline_matches_single_process() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export_ml");
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let max_seq = 32usize;

    // Single-process batched prefill is the reference (itself bit-exact vs
    // per-token, proven by glm5_model).
    let want = load_model(&dir, max_seq)
        .expect("load_model")
        .prefill(&prompt);
    let want_tok = argmax(&want);

    for m in [1usize, 2, 4] {
        let got = pipeline_prefill(&dir, max_seq, m, &prompt).await;
        assert_eq!(got.len(), want.len(), "M={m}: logit length mismatch");
        assert_eq!(
            got, want,
            "M={m}: pipeline batched prefill diverged from single-process prefill"
        );
        assert_eq!(argmax(&got), want_tok, "M={m}: first-token argmax mismatch");
    }
}

/// Drive the `m` ranks as one batched prefill of `prompt` at KV `base`, passing
/// the hidden between ranks directly (the wire moves the same bytes — verified
/// by the test above — so this isolates the per-rank KV/base behaviour).
fn drive_prefill(ranks: &mut [GlmRunner], prompt: &[u32], base: usize) -> Vec<f32> {
    let rows = prompt.len();
    let hsz = ranks[0].hidden_size();
    let mut batch = vec![0.0f32; rows * hsz];
    for (r, &t) in prompt.iter().enumerate() {
        batch[r * hsz..(r + 1) * hsz].copy_from_slice(&ranks[0].embed_token(t));
    }
    let mut h = ranks[0].forward_layers_batch(batch, base, rows);
    for i in 1..ranks.len() {
        h = ranks[i].forward_layers_batch(h, base, rows);
    }
    let last = (rows - 1) * hsz;
    ranks[ranks.len() - 1].head_logits(&h[last..last + hsz])
}

/// Distributed KV-prefix cache parity: caching a base prompt's KV on every rank,
/// restoring it, and prefilling only the SUFFIX (base = base-len) is
/// bit-identical to a full prefill of the whole prompt across the same ranks —
/// for 1/2/4-way shards. This is the correctness core of the distributed prefix
/// cache (`cache_prefix`/`restore_prefix` per rank + `base` suffill).
#[test]
fn glm5_prefix_cache_pipeline_matches_full() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export_ml");
    std::env::set_var("CASCADIA_GLM5_PREFIX_CACHE", "4");
    let base: Vec<u32> = vec![1, 2, 3];
    let ext: Vec<u32> = vec![1, 2, 3, 4, 5, 6]; // `base` is a prefix of `ext`
    let max_seq = 32usize;
    const KEY: u64 = 42;

    for m in [1usize, 2, 4] {
        let mut ranks: Vec<GlmRunner> = (0..m)
            .map(|r| {
                GlmRunner::load_staged(&dir, max_seq, r as u32, m as u32, 0, 0, Default::default())
                    .expect("load")
            })
            .collect();
        assert!(
            ranks[0].prefix_cache_enabled(),
            "prefix cache enabled by env"
        );

        // Reference: full prefill of `ext`.
        for r in &mut ranks {
            r.reset();
        }
        let want = drive_prefill(&mut ranks, &ext, 0);

        // Cached: prefill `base`, cache each rank's slice; reset; restore each
        // rank; prefill the suffix at base = base.len().
        for r in &mut ranks {
            r.reset();
        }
        let _ = drive_prefill(&mut ranks, &base, 0);
        for r in &mut ranks {
            r.cache_prefix(KEY);
        }
        for r in &mut ranks {
            r.reset();
        }
        for r in &mut ranks {
            assert_eq!(
                r.restore_prefix(KEY),
                Some(base.len()),
                "M={m}: restored length"
            );
        }
        let got = drive_prefill(&mut ranks, &ext[base.len()..], base.len());

        assert_eq!(
            got, want,
            "M={m}: distributed prefix-cache reuse diverged from full prefill"
        );
    }
}

/// CI coverage for the FLOWN config: the bf16 embed/lm_head path (production
/// default) must run end-to-end through the staged pipeline and stay within bf16
/// tolerance of the exact-f32 reference. This is the "bounded logit error within
/// the model's bf16 contract" guarantee — NOT byte-identity (which is prompt- and
/// scale-dependent and would be a flaky assertion). m=1 so rank 0 holds BOTH the
/// bf16 embed and the bf16 lm_head.
#[test]
fn glm5_bf16_head_pipeline_bounded_vs_f32() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5_export_ml");
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let max_seq = 32usize;

    let mut f32r = [
        GlmRunner::load_staged(&dir, max_seq, 0, 1, 0, 0, StageOpts::default()).expect("load f32"),
    ];
    f32r[0].reset();
    let want = drive_prefill(&mut f32r, &prompt, 0);

    let mut bf16r = [GlmRunner::load_staged(
        &dir,
        max_seq,
        0,
        1,
        0,
        0,
        StageOpts {
            bf16_head: true,
            bf16_kv: false,
            ..Default::default()
        },
    )
    .expect("load bf16-head")];
    bf16r[0].reset();
    let got = drive_prefill(&mut bf16r, &prompt, 0);

    assert_eq!(got.len(), want.len(), "bf16-head logit length mismatch");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "bf16-head produced non-finite logits"
    );
    // Weights round to bf16 (~2^-8 rel), accumulation stays f32 -> logits move a
    // few percent at most. A generous 15% bound catches gross breakage without
    // flaking on the tiny fixture's argmax sensitivity.
    let max_rel = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
        .fold(0.0f32, f32::max);
    assert!(
        max_rel < 0.15,
        "bf16-head logits diverged from f32 beyond bf16 tolerance: max rel {max_rel}"
    );
}
