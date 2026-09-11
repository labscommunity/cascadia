//! Inkling expert-parallel dispatch (`inkling::ep`): the ExpertDispatch /
//! ExpertResult frames over a real loopback transport, the manifest-free
//! placement + per-worker banks, a driver with two expert workers over
//! loopback TCP reproducing the single-process model BIT FOR BIT (token by
//! token and batched prefill, greedy ids == the HF reference), and the
//! worker-error path.
//!
//! Requires the tiny export (`tools/export_inkling.py --tiny`); the tests that
//! need it skip when it is absent.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cascadia_engine::Engine;
use cascadia_engine_sparse_moe::dist::{
    recv_expert_dispatch_body_server, recv_expert_result_body_client, recv_kind_client,
    recv_kind_server, send_expert_dispatch, send_expert_result_err, send_expert_result_ok,
    FrameKind, EXPERT_PAD, MAX_BATCH_COUNT,
};
use cascadia_engine_sparse_moe::dsv4::loader::ExpertsMode;
use cascadia_engine_sparse_moe::inkling::ep::{
    expert_home, load_expert_bank, EpClient, ExpertBank, ExpertWorkerEngine,
};
use cascadia_engine_sparse_moe::inkling::loader::{load_model_with, read_manifest};
use cascadia_engine_sparse_moe::inkling::model::argmax;
use cascadia_engine_sparse_moe::inkling::stage::InklingRunner;
use cascadia_engine_sparse_moe::staged::StagedRunner;
use cascadia_transport::{ActivationClient, ActivationServer};
use tokio::sync::Mutex;

fn export_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export")
}

/// `(prompt_ids, greedy_ids)` from the export's HF reference, if present.
fn reference() -> Option<(Vec<u32>, Vec<u32>)> {
    let p = export_dir().join("reference.json");
    if !p.exists() {
        eprintln!("inkling_export/reference.json missing; skipping (run export_inkling.py --tiny)");
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let ids = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    Some((ids("prompt_ids"), ids("greedy_ids")))
}

/// One accepted loopback pair: (worker-side server, driver-side client).
async fn loopback() -> (Arc<Mutex<ActivationServer>>, Arc<Mutex<ActivationClient>>) {
    let mut server = ActivationServer::new("127.0.0.1", 0);
    server.start().await.unwrap();
    let port = server.port();
    let server = Arc::new(Mutex::new(server));
    let sc = server.clone();
    let accept = tokio::spawn(async move { sc.lock().await.accept().await.unwrap() });
    let mut client = ActivationClient::new("127.0.0.1", port);
    client
        .connect_with_timeout(Duration::from_secs(5))
        .await
        .unwrap();
    accept.await.unwrap();
    (server, Arc::new(Mutex::new(client)))
}

/// A multi-thread runtime: the worker threads drive their engines with
/// `run_async` from plain threads, which needs the runtime's I/O driver to run
/// on its own workers.
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

/// Spawn one `ExpertWorkerEngine` thread per bank, each on its own loopback
/// connection, looping `step()` exactly the way `run_relay_loop` does until
/// the connection-fatal `Err` after the driver closes. Returns the driver-side
/// clients (in bank order) and the threads (each yields its frames served).
fn spawn_workers(
    rt: &tokio::runtime::Runtime,
    banks: Vec<ExpertBank>,
) -> (
    Vec<Arc<Mutex<ActivationClient>>>,
    Vec<std::thread::JoinHandle<u64>>,
) {
    let handle = rt.handle().clone();
    let mut clients = Vec::new();
    let mut threads = Vec::new();
    for bank in banks {
        let (server, client) = rt.block_on(loopback());
        clients.push(client);
        let wi = bank.index();
        let mut engine = ExpertWorkerEngine::new(bank, server, handle.clone());
        threads.push(std::thread::spawn(move || loop {
            match engine.step() {
                Ok(_) => {}
                Err(e) if e.is_connection_fatal() => break engine.frames_served(),
                Err(e) => panic!("worker {wi} step: {e}"),
            }
        }));
    }
    (clients, threads)
}

fn close_all(rt: &tokio::runtime::Runtime, clients: &[Arc<Mutex<ActivationClient>>]) {
    for c in clients {
        rt.block_on(async { c.lock().await.close().await });
    }
}

// ────────────────────────────── 1. frame round trip ──────────────────────────────

#[tokio::test]
async fn expert_frames_round_trip_over_loopback_with_padding_ok_and_error() {
    let (server, client) = loopback().await;
    let (rows, k, h) = (3u32, 2u32, 4u32);
    let hidden: Vec<f32> = (0..rows * h).map(|i| i as f32 * 0.5 - 1.0).collect();
    // Row 0 needs one expert, row 1 two, row 2 none from this worker.
    let ids = vec![7, EXPERT_PAD, 0, 5, EXPERT_PAD, EXPERT_PAD];

    let c2 = client.clone();
    let (h2, ids2) = (hidden.clone(), ids.clone());
    let send = tokio::spawn(async move {
        send_expert_dispatch(&c2, 42, rows, k, h, &h2, &ids2)
            .await
            .unwrap();
    });
    assert_eq!(
        recv_kind_server(&server).await.unwrap(),
        Some(FrameKind::ExpertDispatch)
    );
    let body = recv_expert_dispatch_body_server(&server).await.unwrap();
    send.await.unwrap();
    assert_eq!(body.layer, 42);
    assert_eq!((body.rows, body.k), (rows, k));
    assert_eq!(body.hidden_shape, [rows, h, 1]);
    assert_eq!(body.hidden_size(), h as usize);
    assert_eq!(body.hidden, hidden);
    assert_eq!(body.ids_shape, [rows, k, 1]);
    assert_eq!(body.ids, ids);

    // ok reply: [rows, k, h] with zeros where the slot was a pad.
    let mut out = vec![0.0f32; (rows * k * h) as usize];
    for (s, id) in ids.iter().enumerate() {
        if *id != EXPERT_PAD {
            for j in 0..h as usize {
                out[s * h as usize + j] = (*id as f32) * 10.0 + j as f32;
            }
        }
    }
    let s2 = server.clone();
    let out2 = out.clone();
    let reply = tokio::spawn(async move {
        send_expert_result_ok(&s2, rows, k, h, &out2).await.unwrap();
    });
    assert_eq!(
        recv_kind_client(&client).await.unwrap(),
        Some(FrameKind::ExpertResult)
    );
    let got = recv_expert_result_body_client(&client).await.unwrap();
    reply.await.unwrap();
    let (data, shape) = got.expect("status 0");
    assert_eq!(shape, [rows, k, h]);
    assert_eq!(data, out);

    // error reply: status 1 + message, then the link is still usable.
    let s2 = server.clone();
    let reply = tokio::spawn(async move {
        send_expert_result_err(&s2, "worker 1/2: layer 3: does not own expert 4")
            .await
            .unwrap();
    });
    assert_eq!(
        recv_kind_client(&client).await.unwrap(),
        Some(FrameKind::ExpertResult)
    );
    let got = recv_expert_result_body_client(&client).await.unwrap();
    reply.await.unwrap();
    assert_eq!(
        got.unwrap_err(),
        "worker 1/2: layer 3: does not own expert 4"
    );
    let s2 = server.clone();
    let reply = tokio::spawn(async move { send_expert_result_err(&s2, "").await.unwrap() });
    assert_eq!(
        recv_kind_client(&client).await.unwrap(),
        Some(FrameKind::ExpertResult)
    );
    assert_eq!(
        recv_expert_result_body_client(&client)
            .await
            .unwrap()
            .unwrap_err(),
        ""
    );
    reply.await.unwrap();

    // Sender-side guards: no empty / oversized frames, no k = 0, no shape lies.
    assert!(send_expert_dispatch(&client, 0, 0, 1, h, &[], &[])
        .await
        .is_err());
    assert!(send_expert_dispatch(
        &client,
        0,
        MAX_BATCH_COUNT + 1,
        1,
        h,
        &vec![0.0; ((MAX_BATCH_COUNT + 1) * h) as usize],
        &vec![0; (MAX_BATCH_COUNT + 1) as usize]
    )
    .await
    .is_err());
    assert!(
        send_expert_dispatch(&client, 0, 1, 0, h, &vec![0.0; h as usize], &[])
            .await
            .is_err()
    );
    assert!(
        send_expert_dispatch(&client, 0, 1, 1, h, &vec![0.0; h as usize + 1], &[0])
            .await
            .is_err()
    );
    assert!(send_expert_result_ok(&server, 1, 1, h, &[0.0; 3])
        .await
        .is_err());
}

// ──────────────────────────── 2. placement + expert banks ────────────────────────────

#[test]
fn placement_homes_every_expert_on_exactly_one_worker_and_banks_hold_exactly_those() {
    let dir = export_dir();
    let Ok(m) = read_manifest(&dir) else {
        eprintln!("inkling_export/manifest.json missing; skipping");
        return;
    };
    let n_ids = m.num_experts + m.n_shared_experts;
    for w in 1..=4u32 {
        // Every id (routed + shared) lands on exactly one of the W workers.
        for id in 0..n_ids {
            let homes = (0..w)
                .filter(|&k| expert_home(id, w as usize) == k as usize)
                .count();
            assert_eq!(homes, 1, "expert {id} with W={w}");
        }
        let mut union: Vec<Vec<usize>> = vec![Vec::new(); m.num_layers];
        for k in 0..w {
            let bank = load_expert_bank(&dir, k, w, ExpertsMode::Eager).expect("bank");
            assert_eq!((bank.index(), bank.count()), (k, w));
            assert_eq!(bank.num_layers(), m.num_layers);
            assert_eq!(
                (
                    bank.hidden(),
                    bank.inter(),
                    bank.n_routed(),
                    bank.n_shared()
                ),
                (
                    m.hidden_size,
                    m.moe_intermediate,
                    m.num_experts,
                    m.n_shared_experts
                )
            );
            for (li, u) in union.iter_mut().enumerate() {
                let owned = bank.owned_ids(li);
                if m.dense_layers.contains(&li) {
                    assert!(!bank.is_moe_layer(li));
                    assert!(owned.is_empty(), "dense layer {li} holds experts");
                    continue;
                }
                assert!(bank.is_moe_layer(li));
                let want: Vec<usize> = (0..n_ids)
                    .filter(|&id| expert_home(id, w as usize) == k as usize)
                    .collect();
                assert_eq!(owned, want, "worker {k} of {w}, layer {li}");
                for &id in &owned {
                    assert!(bank.expert(li, id).is_some());
                }
                assert!(bank.expert(li, n_ids).is_none());
                u.extend(owned);
            }
        }
        for (li, u) in union.iter_mut().enumerate() {
            if !m.dense_layers.contains(&li) {
                u.sort_unstable();
                assert_eq!(*u, (0..n_ids).collect::<Vec<_>>());
            }
        }
    }
    assert!(load_expert_bank(&dir, 2, 2, ExpertsMode::Eager).is_err());
    assert!(load_expert_bank(&dir, 0, 0, ExpertsMode::Eager).is_err());
}

// ─────────────────── 3. driver + 2 workers over loopback TCP ───────────────────

#[test]
fn driver_with_two_workers_over_loopback_matches_single_process_bit_for_bit() {
    driver_vs_single_process("eager", ExpertsMode::Eager);
}

/// The same over mmap'd int4 experts: the worker's decode path is the
/// overlapped whole-bin read + `swiglu_from`, its batch path the mmap kernel —
/// both must equal the local mmap runner / model byte for byte.
#[test]
fn driver_with_two_workers_matches_single_process_on_mmap_experts() {
    driver_vs_single_process("mmap", ExpertsMode::Mmap);
}

fn driver_vs_single_process(mode: &str, experts: ExpertsMode) {
    let Some((prompt, want)) = reference() else {
        return;
    };
    let dir = export_dir();
    let m = read_manifest(&dir).unwrap();
    let rt = runtime();
    const W: u32 = 2;
    let banks: Vec<ExpertBank> = (0..W)
        .map(|k| load_expert_bank(&dir, k, W, experts).unwrap())
        .collect();
    let (clients, threads) = spawn_workers(&rt, banks);
    let ep = Arc::new(EpClient::new(
        clients.clone(),
        rt.handle().clone(),
        m.hidden_size,
        m.num_experts,
        m.n_shared_experts,
    ));
    assert_eq!(ep.n_workers(), W as usize);
    let max_seq = 320;
    let mut driver = InklingRunner::load_staged(
        &dir,
        max_seq,
        0,
        1,
        0,
        0,
        Some(mode.into()),
        Some(ep.clone()),
    )
    .expect("driver");
    let mut local = InklingRunner::load_staged(&dir, max_seq, 0, 1, 0, 0, Some(mode.into()), None)
        .expect("local runner");
    let mut model = load_model_with(&dir, max_seq, experts).expect("model");
    let n_layers = m.num_layers as u32;

    // Token by token: hidden states + logits bit-identical, greedy == reference.
    driver.reset();
    local.reset();
    model.reset();
    let (mut t_ep, mut t_local) = (Duration::ZERO, Duration::ZERO);
    let mut steps = 0u32;
    let mut got = Vec::new();
    let mut next = 0u32;
    let mut pos = 0usize;
    let mut feed = |tok: u32, pos: usize| -> u32 {
        let h = driver.embed_token(tok);
        let t0 = Instant::now();
        let hd = driver.forward_layers(h.clone(), pos, None);
        t_ep += t0.elapsed();
        let t0 = Instant::now();
        let hl = local.forward_layers(h, pos, None);
        t_local += t0.elapsed();
        steps += 1;
        assert_eq!(
            hd, hl,
            "hidden state at pos {pos}: EP driver vs local runner"
        );
        let ld = driver.head_logits(&hd);
        let lm = model.forward_token(tok);
        assert_eq!(ld, lm, "logits at pos {pos}: EP driver vs load_model");
        argmax(&ld) as u32
    };
    for &t in &prompt {
        next = feed(t, pos);
        pos += 1;
    }
    got.push(next);
    for _ in 1..want.len() {
        next = feed(next, pos);
        pos += 1;
        got.push(next);
    }
    assert_eq!(got, want, "EP driver greedy ids vs HF reference");
    eprintln!(
        "inkling EP (2 workers, loopback TCP, {mode} experts): per-layer wall time \
         {:.1} us (driver) vs {:.1} us (single process); {steps} decode steps × {n_layers} layers",
        t_ep.as_secs_f64() * 1e6 / (steps * n_layers) as f64,
        t_local.as_secs_f64() * 1e6 / (steps * n_layers) as f64,
    );

    // Batched prefill: bit-identical to the local batch AND to load_model's
    // prefill; the argmax is the reference's first token.
    driver.reset();
    local.reset();
    model.reset();
    let rows = prompt.len();
    let hs = m.hidden_size;
    let mut batch = vec![0.0f32; rows * hs];
    for (r, &t) in prompt.iter().enumerate() {
        batch[r * hs..(r + 1) * hs].copy_from_slice(&driver.embed_token(t));
    }
    let t0 = Instant::now();
    let hd = driver.forward_layers_batch(batch.clone(), 0, rows);
    let t_ep_b = t0.elapsed();
    let t0 = Instant::now();
    let hl = local.forward_layers_batch(batch, 0, rows);
    let t_local_b = t0.elapsed();
    assert_eq!(hd, hl, "batched prefill hidden: EP driver vs local runner");
    let ld = driver.head_logits(&hd[(rows - 1) * hs..]);
    let lm = model.prefill(&prompt);
    assert_eq!(ld, lm, "batched prefill logits: EP driver vs load_model");
    assert_eq!(argmax(&ld) as u32, want[0]);
    eprintln!(
        "inkling EP batched prefill ({mode}, {rows} rows): per-layer wall time {:.1} us (driver) vs \
         {:.1} us (single process)",
        t_ep_b.as_secs_f64() * 1e6 / n_layers as f64,
        t_local_b.as_secs_f64() * 1e6 / n_layers as f64,
    );

    // A prefill longer than one frame (MAX_BATCH_COUNT rows) is chunked by the
    // driver: still bit-identical to the local batch.
    let long_rows = MAX_BATCH_COUNT as usize + 37;
    assert!(long_rows <= max_seq);
    driver.reset();
    local.reset();
    let mut seed = 0x2545_F491u32;
    let mut long = vec![0.0f32; long_rows * hs];
    for r in 0..long_rows {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let tok = (seed >> 8) % m.vocab_size as u32;
        long[r * hs..(r + 1) * hs].copy_from_slice(&driver.embed_token(tok));
    }
    let hd = driver.forward_layers_batch(long.clone(), 0, long_rows);
    let hl = local.forward_layers_batch(long, 0, long_rows);
    assert_eq!(
        hd, hl,
        "chunked ({long_rows}-row) prefill: EP driver vs local"
    );

    // Protocol cost in isolation: one decode row to both workers with the
    // routed + shared experts of the first MoE layer — the dispatch every MoE
    // layer makes per token — averaged over many rounds.
    let li = (0..m.num_layers)
        .find(|li| !m.dense_layers.contains(li))
        .unwrap() as u32;
    let x = vec![0.01f32; hs];
    let ids: Vec<(usize, f32)> = (0..m.top_k)
        .map(|i| (i, 0.5))
        .chain((0..m.n_shared_experts).map(|s| (m.num_experts + s, 0.25)))
        .collect();
    let rounds = 200u32;
    ep.dispatch(li, &x, std::slice::from_ref(&ids)).unwrap();
    let t0 = Instant::now();
    for _ in 0..rounds {
        ep.dispatch(li, &x, std::slice::from_ref(&ids)).unwrap();
    }
    eprintln!(
        "inkling EP dispatch RTT ({mode}): {:.1} us per MoE layer (1 row, {} experts over {W} \
         workers, loopback TCP)",
        t0.elapsed().as_secs_f64() * 1e6 / rounds as f64,
        ids.len()
    );

    // Teardown: closing the driver's connections is a clean close on each
    // worker, which ends its step loop with a connection-fatal Err.
    drop(driver);
    drop(ep);
    close_all(&rt, &clients);
    for (wi, t) in threads.into_iter().enumerate() {
        let frames = t.join().expect("worker thread");
        assert!(frames > 0, "worker {wi} served no frames");
    }
}

// ───────────────────────────── 4. worker failure ─────────────────────────────

#[test]
fn a_worker_asked_for_an_expert_it_does_not_own_fails_the_dispatch_with_its_message() {
    let dir = export_dir();
    let Ok(m) = read_manifest(&dir) else {
        eprintln!("inkling_export/manifest.json missing; skipping");
        return;
    };
    let rt = runtime();
    // The one worker holds shard 0 of 2 (even ids only) — but the client is
    // told it is the only worker, so odd ids are routed to it too.
    let bank = load_expert_bank(&dir, 0, 2, ExpertsMode::Eager).unwrap();
    let li = (0..m.num_layers)
        .find(|li| !m.dense_layers.contains(li))
        .expect("a MoE layer");
    let owned = bank.owned_ids(li);
    let hs = m.hidden_size;
    let (clients, threads) = spawn_workers(&rt, vec![bank]);
    let ep = EpClient::new(
        clients.clone(),
        rt.handle().clone(),
        hs,
        m.num_experts,
        m.n_shared_experts,
    );
    let x: Vec<f32> = (0..hs).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();

    let odd = (0..m.num_experts + m.n_shared_experts)
        .find(|id| !owned.contains(id))
        .unwrap();
    let err = ep
        .dispatch(li as u32, &x, &[vec![(owned[0], 0.5), (odd, 0.25)]])
        .expect_err("an unowned expert must fail the dispatch");
    assert!(
        err.contains("expert worker 0") && err.contains(&format!("layer {li}")),
        "error must name the worker and the layer: {err}"
    );
    assert!(
        err.contains(&format!("does not own expert {odd}")),
        "error must carry the worker's message: {err}"
    );

    // The worker kept serving: a dispatch it can serve succeeds and equals the
    // local accumulation over the same experts, bit for bit.
    let bank = load_expert_bank(&dir, 0, 2, ExpertsMode::Eager).unwrap();
    let per_row: Vec<(usize, f32)> = owned
        .iter()
        .map(|&id| (id, 0.125 * id as f32 + 0.3))
        .collect();
    let got = ep
        .dispatch(li as u32, &x, std::slice::from_ref(&per_row))
        .expect("owned experts");
    let mut want = vec![0.0f32; hs];
    for &(id, w) in &per_row {
        let y = bank
            .expert(li, id)
            .unwrap()
            .forward(&x, hs, m.moe_intermediate);
        for (o, &yi) in want.iter_mut().zip(&y) {
            *o += w * yi;
        }
    }
    assert_eq!(got, want);
    assert!(want.iter().any(|&v| v != 0.0));

    // Out-of-range ids and a dense layer are rejected too (driver-side and
    // worker-side respectively), and the link survives both.
    assert!(ep
        .dispatch(
            li as u32,
            &x,
            &[vec![(m.num_experts + m.n_shared_experts, 1.0)]]
        )
        .is_err());
    if let Some(&dense) = m.dense_layers.first() {
        let e = ep
            .dispatch(dense as u32, &x, &[vec![(owned[0], 1.0)]])
            .expect_err("dense layer");
        assert!(e.contains("dense"), "{e}");
    }
    assert_eq!(ep.dispatch(li as u32, &x, &[per_row]).unwrap(), want);

    drop(ep);
    close_all(&rt, &clients);
    for t in threads {
        assert!(t.join().unwrap() >= 2);
    }
}
