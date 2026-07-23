//! GLM-5.2 distributed-shard correctness check on ONE box (e.g. a single dev box) before
//! shipping to the real fleet. Runs the model across M pipeline ranks over the
//! REAL loopback transport (ActivationServer/Client on 127.0.0.1) — same split
//! (`index_aligned_split`), per-rank slice load, cross-rank hidden handoff, and
//! wire frames as N physical nodes, just co-located. Asserts the M-rank output
//! is token-identical to the single-process (total=1) run.
//!
//!   cargo run --release --example glm5_shardcheck -- <model_dir> [n_gen] [M]
//!   CASCADIA_GLM5_EXPERTS=mmap cargo run --release --example glm5_shardcheck -- <dir> 16 4
//!
//! If M-rank == single-process, the sharding is proven on the real model and
//! deploying to M nodes is just 127.0.0.1 → real hostnames.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cascadia_engine_sparse_moe::dist::{
    recv_forward_body_server, recv_kind_server, send_forward, FrameKind,
};
use cascadia_engine_sparse_moe::glm::stage::{GlmRunner, GLM5_DEFAULT_MAX_SEQ};
use cascadia_engine_sparse_moe::staged::StagedRunner;
use cascadia_engine_sparse_moe::SamplingConfig;
use cascadia_transport::{ActivationClient, ActivationServer};
use tokenizers::Tokenizer;
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

/// Greedy generation across `m` pipeline ranks over `m-1` loopback links.
async fn pipeline_generate(
    dir: &Path,
    max_seq: usize,
    m: usize,
    prompt: &[u32],
    n_gen: usize,
) -> Vec<u32> {
    let mut ranks: Vec<GlmRunner> = (0..m)
        .map(|r| GlmRunner::load_staged(dir, max_seq, r as u32, m as u32, 0, 0).expect("load rank"))
        .collect();
    for run in ranks.iter_mut() {
        run.reset();
    }
    let hsz = ranks[0].hidden_size() as u32;

    // m-1 loopback links, each an accept()ed ActivationServer + connected client.
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
            .connect_with_timeout(Duration::from_secs(10))
            .await
            .unwrap();
        atask.await.unwrap();
        clients.push(Arc::new(Mutex::new(client)));
        servers.push(server);
    }

    let cfg = SamplingConfig::default();
    async fn step(
        ranks: &mut [GlmRunner],
        clients: &[Arc<Mutex<ActivationClient>>],
        servers: &[Arc<Mutex<ActivationServer>>],
        tok: u32,
        pos: usize,
        hsz: u32,
        cfg: &SamplingConfig,
    ) -> u32 {
        let m = ranks.len();
        let mut h = ranks[0].embed_token(tok);
        h = ranks[0].forward_layers(h, pos, None);
        for i in 0..m - 1 {
            let client = clients[i].clone();
            let cfg2 = cfg.clone();
            let hsend = h;
            let send = tokio::spawn(async move {
                send_forward(&client, pos as u32, &cfg2, &hsend, [1, 1, hsz])
                    .await
                    .unwrap();
            });
            let k = recv_kind_server(&servers[i]).await.unwrap();
            assert_eq!(k, Some(FrameKind::Forward));
            let (_p, _c, hw, _s) = recv_forward_body_server(&servers[i]).await.unwrap();
            send.await.unwrap();
            h = ranks[i + 1].forward_layers(hw, pos, None);
        }
        argmax(&ranks[m - 1].head_logits(&h))
    }

    let mut pos = 0usize;
    let mut next = 0u32;
    for &t in prompt {
        next = step(&mut ranks, &clients, &servers, t, pos, hsz, &cfg).await;
        pos += 1;
    }
    let mut out = vec![next];
    for _ in 1..n_gen {
        next = step(&mut ranks, &clients, &servers, next, pos, hsz, &cfg).await;
        pos += 1;
        out.push(next);
    }
    out
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: glm5_shardcheck <model_dir> [n_gen=12] [M=4]");
        std::process::exit(2);
    }
    let dir = PathBuf::from(&args[1]);
    let n_gen: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);
    let m: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let max_seq: usize = std::env::var("CASCADIA_GLM5_MAX_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(GLM5_DEFAULT_MAX_SEQ);

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).ok();
    let prompt: Vec<u32> = match tok.as_ref() {
        Some(t) => t
            .encode("The capital of France is", true)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec(),
        None => vec![1, 2, 3, 4],
    };
    println!(
        "[shardcheck] dir={} max_seq={max_seq} M={m} n_gen={n_gen} prompt_ids={}",
        dir.display(),
        prompt.len()
    );

    // 1) single-process baseline (total=1). Dropped before the M-rank run so peak
    //    RAM stays ~one full model.
    let want: Vec<u32> = {
        let t0 = Instant::now();
        let mut solo = GlmRunner::load_staged(&dir, max_seq, 0, 1, 0, 0)?;
        println!(
            "[shardcheck] single-process loaded in {:.0}s",
            t0.elapsed().as_secs_f64()
        );
        let g0 = Instant::now();
        let out = solo.generate_argmax(&prompt, n_gen);
        println!(
            "[shardcheck] single: {out:?}  ({:.1}s)",
            g0.elapsed().as_secs_f64()
        );
        out
    };

    // 2) M-rank pipeline over loopback.
    let t1 = Instant::now();
    let got = pipeline_generate(&dir, max_seq, m, &prompt, n_gen).await;
    println!(
        "[shardcheck] {m}-rank: {got:?}  ({:.1}s)",
        t1.elapsed().as_secs_f64()
    );

    if let Some(t) = tok.as_ref() {
        if let Ok(txt) = t.decode(&got, true) {
            println!("[shardcheck] {m}-rank text: {txt:?}");
        }
    }

    if got == want {
        println!(
            "[shardcheck] ✓ {m}-rank output == single-process — sharding correct on the real model"
        );
        Ok(())
    } else {
        eprintln!("[shardcheck] ✗ MISMATCH\n  single: {want:?}\n  {m}-rank: {got:?}");
        Err("distributed shard diverged from single-process".into())
    }
}
