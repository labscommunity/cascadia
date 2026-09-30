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
    send_restore_prefix, FrameKind, EXPERT_PAD, MAX_BATCH_COUNT,
};
use cascadia_engine_sparse_moe::dsv4::loader::ExpertsMode;
use cascadia_engine_sparse_moe::inkling::ep::{
    expert_home, load_expert_bank, load_expert_bank_with_placement, EpClient, ExpertBank,
    ExpertWorkerEngine,
};
use cascadia_engine_sparse_moe::inkling::ep_placement::{EpPlacement, EpWorkerCost};
use cascadia_engine_sparse_moe::inkling::loader::InklingManifest;
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
    let m = read_manifest(&dir).expect("checked-in inkling_export manifest");
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
    driver_vs_single_process_placed(mode, experts, false, 2);
}

fn replicated_placement(m: &InklingManifest, workers: usize) -> EpPlacement {
    EpPlacement {
        version: 1,
        hidden_size: m.hidden_size,
        moe_intermediate: m.moe_intermediate,
        num_experts: m.num_experts,
        n_shared_experts: m.n_shared_experts,
        expert_bytes: 3 * m.hidden_size as u64 * m.moe_intermediate as u64 * 9 / 16,
        workers: (0..workers)
            .map(|wi| EpWorkerCost {
                name: format!("worker-{wi}"),
                expert_capacity_bytes: 1 << 30,
                read_us: 10.0,
                compute_us: 1.0,
                dispatch_us: 2.0,
            })
            .collect(),
        layers: (0..m.num_layers)
            .map(|li| {
                if m.dense_layers.contains(&li) {
                    return vec![];
                }
                (0..m.num_experts + m.n_shared_experts)
                    .map(|id| {
                        if id >= m.num_experts {
                            (0..workers).collect()
                        } else {
                            vec![(id + li) % workers]
                        }
                    })
                    .collect()
            })
            .collect(),
    }
}

#[test]
fn replicated_shared_experts_and_layer_placement_preserve_full_model_bits() {
    driver_vs_single_process_placed("mmap", ExpertsMode::Mmap, true, 3);
}

#[test]
fn twelve_workers_preserve_logits_prefill_reset_and_replicated_shared_experts() {
    driver_vs_single_process_placed("mmap", ExpertsMode::Mmap, true, 12);
}

#[test]
fn placement_checks_capacity_coverage_dimensions_and_costs() {
    let m = read_manifest(&export_dir()).unwrap();
    let p = replicated_placement(&m, 3);
    p.validate(&m, 3).unwrap();
    let li = m.dense_layers.len();
    let mut bad = p.clone();
    bad.layers[li][0].clear();
    assert!(bad.validate(&m, 3).unwrap_err().contains("no owner"));
    let mut bad = p.clone();
    bad.layers[li][0] = vec![3];
    assert!(bad.validate(&m, 3).is_err());
    let mut bad = p.clone();
    bad.layers[li][0] = vec![1, 1];
    assert!(bad.validate(&m, 3).is_err());
    let mut bad = p.clone();
    bad.workers[0].expert_capacity_bytes = 1;
    assert!(bad.validate(&m, 3).unwrap_err().contains("exceed budget"));
    let mut bad = p.clone();
    bad.workers[0].read_us = f64::NAN;
    assert!(bad.validate(&m, 3).is_err());
    let mut bad = p.clone();
    bad.hidden_size += 32;
    assert!(bad.validate(&m, 3).is_err());
    assert!(p.validate(&m, 2).is_err());
    let mut bad = p.clone();
    bad.expert_bytes = 1;
    assert!(
        load_expert_bank_with_placement(&export_dir(), 0, 3, ExpertsMode::Mmap, Some(&bad))
            .is_err()
    );
}

#[test]
fn replicas_avoid_fixed_owner_collisions_and_account_for_worker_cost() {
    let m = read_manifest(&export_dir()).unwrap();
    let mut p = replicated_placement(&m, 3);
    let li = m.dense_layers.len();
    p.layers[li][0] = vec![0];
    p.layers[li][1] = vec![0];
    // Shared replicas must avoid rank 0, where two fixed routed reads land.
    let shared = m.num_experts;
    let rows = vec![vec![(shared, 0.2), (0, 0.1), (1, 0.3), (shared + 1, 0.4)]];
    let owners = p.assign(li, &rows).unwrap();
    assert_eq!(
        (owners[0], owners[1], owners[shared], owners[shared + 1]),
        (0, 0, 1, 2)
    );
    p.workers[1].read_us = 1000.0;
    let owners = p.assign(li, &rows).unwrap();
    assert_eq!(owners[shared], 2);
    assert_ne!(owners[shared + 1], 1);
    assert_eq!(owners, p.assign(li, &rows).unwrap());
}

#[test]
fn dispatch_compacts_empty_rows_and_preserves_duplicate_slot_order() {
    let rt = runtime();
    let mut clients = Vec::new();
    let mut tasks = Vec::new();
    for wi in 0..2 {
        let (server, client) = rt.block_on(loopback());
        clients.push(client);
        tasks.push(rt.spawn(async move {
            assert_eq!(
                recv_kind_server(&server).await.unwrap(),
                Some(FrameKind::ExpertDispatch)
            );
            let body = recv_expert_dispatch_body_server(&server).await.unwrap();
            assert_eq!(body.rows, 2, "empty rows must not reach worker {wi}");
            assert_eq!(
                body.hidden,
                if wi == 0 {
                    vec![3.0, 4.0, 7.0, 8.0]
                } else {
                    vec![1.0, 2.0, 7.0, 8.0]
                }
            );
            let mut out = vec![0.0; body.ids.len() * 2];
            for (slot, &id) in body.ids.iter().enumerate() {
                if id == EXPERT_PAD {
                    continue;
                }
                for j in 0..2 {
                    out[slot * 2 + j] =
                        body.hidden[(slot / body.k as usize) * 2 + j] + id as f32 * 100.0;
                }
            }
            send_expert_result_ok(&server, body.rows, body.k, 2, &out)
                .await
                .unwrap();
        }));
    }
    let ep = EpClient::new(clients.clone(), rt.handle().clone(), 2, 2, 0);
    let rows = vec![
        vec![(1, 0.5)],
        vec![(0, 0.25)],
        vec![],
        vec![(1, 0.5), (0, 0.25), (1, -0.125)],
    ];
    let out = ep
        .dispatch(0, &[1., 2., 3., 4., 5., 6., 7., 8.], &rows)
        .unwrap();
    assert_eq!(
        out,
        vec![
            50.5,
            51.,
            0.75,
            1.,
            0.,
            0.,
            (0.5 * 107. + 0.25 * 7.) - 0.125 * 107.,
            (0.5 * 108. + 0.25 * 8.) - 0.125 * 108.
        ]
    );
    for t in tasks {
        rt.block_on(t).unwrap();
    }
    close_all(&rt, &clients);
}

fn driver_vs_single_process_placed(mode: &str, experts: ExpertsMode, placed: bool, w: u32) {
    let (prompt, want) = reference().expect("checked-in inkling_export reference");
    let dir = export_dir();
    let m = read_manifest(&dir).unwrap();
    let rt = runtime();
    let placement = placed.then(|| Arc::new(replicated_placement(&m, w as usize)));
    let banks: Vec<ExpertBank> = (0..w)
        .map(|k| {
            load_expert_bank_with_placement(&dir, k, w, experts, placement.as_deref()).unwrap()
        })
        .collect();
    let (clients, threads) = spawn_workers(&rt, banks);
    let mut ep = EpClient::new(
        clients.clone(),
        rt.handle().clone(),
        m.hidden_size,
        m.num_experts,
        m.n_shared_experts,
    );
    if let Some(p) = placement {
        ep = ep.with_placement(p, &m).unwrap();
    }
    let ep = Arc::new(ep);
    assert_eq!(ep.n_workers(), w as usize);
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
        "inkling EP ({w} workers, loopback TCP, {mode} experts): per-layer wall time \
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
        "inkling EP dispatch RTT ({mode}): {:.1} us per MoE layer (1 row, {} experts over {w} \
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
        // A tiny fixture has fewer routed experts than a 12-node fleet; a
        // shared-only replica may legitimately receive no selected traffic.
        if w <= 3 {
            assert!(frames > 0, "worker {wi} served no frames");
        }
    }
}

// ───────────────────────────── 4. worker failure ─────────────────────────────

#[test]
fn a_worker_asked_for_an_expert_it_does_not_own_fails_the_dispatch_with_its_message() {
    let dir = export_dir();
    let m = read_manifest(&dir).expect("checked-in inkling_export manifest");
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

/// With TWO real workers, a dispatch that one worker cannot serve must still
/// drain the OTHER worker's reply (the driver awaits every reply with
/// `join_all`, not `try_join_all`), so the surviving connection stays frame
/// aligned and the NEXT dispatch over the same connections returns correct
/// data. If the failed worker's reply were left unread, the next layer's
/// dispatch would read a stale reply and return a silently wrong answer.
#[test]
fn two_workers_one_fails_a_dispatch_but_the_survivor_stays_frame_aligned() {
    let dir = export_dir();
    let m = read_manifest(&dir).expect("checked-in inkling_export manifest");
    let rt = runtime();
    let hs = m.hidden_size;
    let li = (0..m.num_layers)
        .find(|li| !m.dense_layers.contains(li))
        .expect("a MoE layer");

    // EpClient derives W=2 from its two clients, so expert_home(id, 2) = id % 2.
    // Worker 0 is an honest shard 0 of 2 (owns every even id). Worker 1 is
    // loaded as shard 1 of *4* (owns id % 4 == 1), so an odd id with id % 4 == 3
    // homes to it under W=2 yet it does NOT own it — worker 1 fails a dispatch
    // that worker 0 serves.
    let w0 = load_expert_bank(&dir, 0, 2, ExpertsMode::Eager).unwrap();
    let w1 = load_expert_bank(&dir, 1, 4, ExpertsMode::Eager).unwrap();
    let owned0 = w0.owned_ids(li);
    let owned1 = w1.owned_ids(li);
    let (clients, threads) = spawn_workers(&rt, vec![w0, w1]);
    let ep = EpClient::new(
        clients.clone(),
        rt.handle().clone(),
        hs,
        m.num_experts,
        m.n_shared_experts,
    );
    assert_eq!(ep.n_workers(), 2);

    let x: Vec<f32> = (0..hs).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1).collect();
    let even = *owned0.iter().find(|&&id| id % 2 == 0).expect("an even id");
    let bad = (0..m.num_experts)
        .find(|id| id % 4 == 3)
        .expect("an odd id worker-1-of-4 does not own");
    assert!(!owned1.contains(&bad));

    // Worker 0 succeeds, worker 1 fails: the dispatch is an Err naming worker 1.
    let err = ep
        .dispatch(li as u32, &x, &[vec![(even, 0.5), (bad, 0.25)]])
        .expect_err("worker 1 cannot serve the id");
    assert!(
        err.contains("expert worker 1") && err.contains(&format!("does not own expert {bad}")),
        "error must name worker 1 and carry its message: {err}"
    );

    // The survivor stayed frame aligned: a subsequent fully-owned dispatch to
    // BOTH workers is bit-identical to the local accumulation. If worker 0's
    // earlier reply had been left in the socket this would read it as the new
    // result and mismatch.
    let odd = *owned1
        .iter()
        .find(|&&id| id % 2 == 1)
        .expect("an odd id worker 1 owns");
    let per_row: Vec<(usize, f32)> = vec![(even, 0.5), (odd, 0.25)];
    let got = ep
        .dispatch(li as u32, &x, std::slice::from_ref(&per_row))
        .expect("both workers serve the second dispatch");
    let b0 = load_expert_bank(&dir, 0, 2, ExpertsMode::Eager).unwrap();
    let b1 = load_expert_bank(&dir, 1, 4, ExpertsMode::Eager).unwrap();
    let mut want = vec![0.0f32; hs];
    for &(id, w) in &per_row {
        let bank = if id % 2 == 0 { &b0 } else { &b1 };
        let y = bank
            .expert(li, id)
            .unwrap()
            .forward(&x, hs, m.moe_intermediate);
        for (o, &yi) in want.iter_mut().zip(&y) {
            *o += w * yi;
        }
    }
    assert_eq!(
        got, want,
        "post-failure dispatch must be bit-identical to local accumulation"
    );
    assert!(want.iter().any(|&v| v != 0.0));

    drop(ep);
    close_all(&rt, &clients);
    for (wi, t) in threads.into_iter().enumerate() {
        let frames = t.join().expect("worker thread");
        assert!(frames > 0, "worker {wi} served no frames");
    }
}

/// Generalizes the frame-alignment invariant to THREE workers with TWO
/// survivors and one failure in the same dispatch. `join_all` must drain BOTH
/// surviving replies (workers 0 and 1), not just the first — otherwise the
/// next dispatch reads a stale reply off whichever survivor was left unread.
#[test]
fn three_workers_one_fails_and_both_survivors_stay_frame_aligned() {
    let dir = export_dir();
    let m = read_manifest(&dir).expect("checked-in inkling_export manifest");
    let rt = runtime();
    let hs = m.hidden_size;
    let li = (0..m.num_layers)
        .find(|li| !m.dense_layers.contains(li))
        .expect("a MoE layer");

    // EpClient derives W=3, so expert_home(id, 3) = id % 3. Workers 0 and 1 are
    // honest shards 0 and 1 of 3. Worker 2 is loaded as shard 2 of *6* (owns
    // id % 6 == 2), so an id with id % 3 == 2 and id % 6 == 5 homes to worker 2
    // under W=3 yet it does NOT own it — worker 2 fails a dispatch that workers
    // 0 and 1 both serve.
    let w0 = load_expert_bank(&dir, 0, 3, ExpertsMode::Eager).unwrap();
    let w1 = load_expert_bank(&dir, 1, 3, ExpertsMode::Eager).unwrap();
    let w2 = load_expert_bank(&dir, 2, 6, ExpertsMode::Eager).unwrap();
    let (owned0, owned1, owned2) = (w0.owned_ids(li), w1.owned_ids(li), w2.owned_ids(li));
    let (clients, threads) = spawn_workers(&rt, vec![w0, w1, w2]);
    let ep = EpClient::new(
        clients.clone(),
        rt.handle().clone(),
        hs,
        m.num_experts,
        m.n_shared_experts,
    );
    assert_eq!(ep.n_workers(), 3);

    let x: Vec<f32> = (0..hs).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
    let a = *owned0
        .iter()
        .find(|&&id| id % 3 == 0)
        .expect("an id worker 0 owns");
    let b = *owned1
        .iter()
        .find(|&&id| id % 3 == 1)
        .expect("an id worker 1 owns");
    let bad = (0..m.num_experts)
        .find(|id| id % 3 == 2 && id % 6 == 5)
        .expect("an id homing to worker 2 that shard-2-of-6 does not own");
    assert!(!owned2.contains(&bad));

    // Workers 0 and 1 succeed, worker 2 fails: the dispatch errors naming worker 2.
    let err = ep
        .dispatch(li as u32, &x, &[vec![(a, 0.5), (b, 0.25), (bad, 0.125)]])
        .expect_err("worker 2 cannot serve the id");
    assert!(
        err.contains("expert worker 2") && err.contains(&format!("does not own expert {bad}")),
        "error must name worker 2 and carry its message: {err}"
    );

    // Both survivors stayed frame aligned: a fully-owned dispatch across all
    // three workers is bit-identical to local accumulation. If either worker 0
    // or worker 1 had left its earlier reply in the socket, this would read it
    // as the new result and mismatch.
    let c = *owned2
        .iter()
        .find(|&&id| id % 3 == 2)
        .expect("an id worker 2 owns");
    let per_row: Vec<(usize, f32)> = vec![(a, 0.5), (b, 0.25), (c, 0.125)];
    let got = ep
        .dispatch(li as u32, &x, std::slice::from_ref(&per_row))
        .expect("all three workers serve the second dispatch");
    let b0 = load_expert_bank(&dir, 0, 3, ExpertsMode::Eager).unwrap();
    let b1 = load_expert_bank(&dir, 1, 3, ExpertsMode::Eager).unwrap();
    let b2 = load_expert_bank(&dir, 2, 6, ExpertsMode::Eager).unwrap();
    let mut want = vec![0.0f32; hs];
    for &(id, w) in &per_row {
        let bank = match id % 3 {
            0 => &b0,
            1 => &b1,
            _ => &b2,
        };
        let y = bank
            .expert(li, id)
            .unwrap()
            .forward(&x, hs, m.moe_intermediate);
        for (o, &yi) in want.iter_mut().zip(&y) {
            *o += w * yi;
        }
    }
    assert_eq!(
        got, want,
        "post-failure dispatch must be bit-identical to local accumulation"
    );
    assert!(want.iter().any(|&v| v != 0.0));

    drop(ep);
    close_all(&rt, &clients);
    for (wi, t) in threads.into_iter().enumerate() {
        let frames = t.join().expect("worker thread");
        assert!(frames > 0, "worker {wi} served no frames");
    }
}

/// A worker that receives a non-dispatch frame drains its fixed body, replies
/// ExpertResult{status 1}, and keeps serving. Once the reply is read the
/// connection stays frame aligned, so the next real dispatch succeeds. If the
/// one-way body were not drained (or the reject reply left unread), the stream
/// would desync and the next dispatch would read a stale frame.
#[test]
fn worker_drains_and_rejects_a_foreign_frame_then_keeps_serving() {
    let dir = export_dir();
    let m = read_manifest(&dir).expect("checked-in inkling_export manifest");
    let rt = runtime();
    let hs = m.hidden_size;
    let li = (0..m.num_layers)
        .find(|li| !m.dense_layers.contains(li))
        .expect("a MoE layer");
    // Shard 0 of 1 owns every id, so any routed expert dispatches locally.
    let bank = load_expert_bank(&dir, 0, 1, ExpertsMode::Eager).unwrap();
    let id = bank.owned_ids(li)[0];
    let (clients, threads) = spawn_workers(&rt, vec![bank]);
    let client = clients[0].clone();

    // RestorePrefix is a one-way frame foreign to a stateless expert worker.
    rt.block_on(async {
        send_restore_prefix(&client, 0xABCD_u64).await.unwrap();
        // The worker drained the key body and replied a status-1 rejection.
        assert_eq!(
            recv_kind_client(&client).await.unwrap(),
            Some(FrameKind::ExpertResult)
        );
        let rejected = recv_expert_result_body_client(&client)
            .await
            .unwrap()
            .expect_err("a foreign frame must be rejected");
        assert!(
            rejected.contains("serves expert dispatch frames only"),
            "unexpected rejection message: {rejected}"
        );
    });

    // The connection stayed aligned: a real dispatch over the SAME client works
    // and equals the local expert accumulation.
    let ep = EpClient::new(
        clients.clone(),
        rt.handle().clone(),
        hs,
        m.num_experts,
        m.n_shared_experts,
    );
    let x: Vec<f32> = (0..hs).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let got = ep
        .dispatch(li as u32, &x, &[vec![(id, 0.5)]])
        .expect("dispatch after a rejected foreign frame");
    let y = load_expert_bank(&dir, 0, 1, ExpertsMode::Eager)
        .unwrap()
        .expert(li, id)
        .unwrap()
        .forward(&x, hs, m.moe_intermediate);
    let want: Vec<f32> = y.iter().map(|&yi| 0.5 * yi).collect();
    assert_eq!(got, want);

    drop(ep);
    close_all(&rt, &clients);
    for t in threads {
        assert!(t.join().unwrap() >= 1);
    }
}

#[test]
fn fused_wire_preserves_weights_compacts_rows_and_sums_partials() {
    fused_wire_case(3);
    fused_wire_case(12);
}

fn fused_wire_case(count: usize) {
    use cascadia_engine_sparse_moe::dist::recv_fused_expert_dispatch_body_server;
    let rt = runtime();
    let mut clients = Vec::new();
    let mut tasks = Vec::new();
    for wi in 0..count {
        let (server, client) = rt.block_on(loopback());
        clients.push(client);
        tasks.push(rt.spawn(async move {
            assert_eq!(
                recv_kind_server(&server).await.unwrap(),
                Some(FrameKind::FusedExpertDispatch)
            );
            let (b, weights, shape) = recv_fused_expert_dispatch_body_server(&server)
                .await
                .unwrap();
            assert_eq!(shape, b.ids_shape);
            let mut out = vec![0.; b.rows as usize * 2];
            for (slot, (&id, &weight)) in b.ids.iter().zip(&weights).enumerate() {
                if id == EXPERT_PAD {
                    assert_eq!(weight, 0.);
                    continue;
                }
                assert_eq!(id as usize % count, wi);
                let r = slot / b.k as usize;
                for j in 0..2 {
                    out[r * 2 + j] += weight * (b.hidden[r * 2 + j] + id as f32 * 10.);
                }
            }
            send_expert_result_ok(&server, b.rows, 1, 2, &out)
                .await
                .unwrap();
        }));
    }
    let ep = EpClient::new(clients.clone(), rt.handle().clone(), 2, count * 2, 2).with_fused(true);
    let rows: Vec<_> = (0..count)
        .map(|wi| {
            vec![
                (wi, 0.5),
                (wi + count, -0.25),
                (count * 2, 0.75),
                (count * 2 + 1, 0.125),
            ]
        })
        .chain([vec![]])
        .collect();
    let xs: Vec<f32> = (1..=rows.len() * 2).map(|x| x as f32).collect();
    let out = ep.dispatch(1, &xs, &rows).unwrap();
    let xs_ref = &xs;
    let expected: Vec<f32> = rows
        .iter()
        .enumerate()
        .flat_map(|(r, ids)| {
            (0..2).map(move |j| {
                ids.iter()
                    .map(|&(id, w)| w * (xs_ref[r * 2 + j] + id as f32 * 10.))
                    .sum::<f32>()
            })
        })
        .collect();
    assert_eq!(out, expected);
    close_all(&rt, &clients);
    for t in tasks {
        rt.block_on(t).unwrap();
    }
}

#[test]
fn unsupported_fused_request_is_drained_without_cpu_fallback() {
    let rt = runtime();
    let bank = load_expert_bank(&export_dir(), 0, 1, ExpertsMode::Mmap).unwrap();
    let hidden = bank.hidden();
    let (clients, threads) = spawn_workers(&rt, vec![bank]);
    let ep = EpClient::new(clients.clone(), rt.handle().clone(), hidden, 8, 2).with_fused(true);
    assert!(ep
        .dispatch(1, &vec![0.; hidden], &[vec![(0, 0.25)]])
        .unwrap_err()
        .contains("does not support fused"));
    // The weighted tensor was consumed before rejection; a subsequent raw
    // request on this exact socket succeeds, proving framing was preserved.
    let ep = ep.with_fused(false);
    assert!(ep
        .dispatch(1, &vec![0.; hidden], &[vec![(0, 0.25)]])
        .is_ok());
    close_all(&rt, &clients);
    for t in threads {
        assert_eq!(t.join().unwrap(), 1);
    }
}

#[test]
fn invalid_fused_header_still_drains_the_weights_tensor() {
    use cascadia_engine_sparse_moe::dist::recv_fused_expert_dispatch_body_server;
    use cascadia_transport::{DType, Tensor};
    let rt = runtime();
    rt.block_on(async {
        let (server, client) = loopback().await;
        {
            let mut c = client.lock().await;
            let header: Vec<u8> = [
                FrameKind::FusedExpertDispatch as u32,
                1,
                MAX_BATCH_COUNT + 1,
                1,
            ]
            .into_iter()
            .flat_map(u32::to_be_bytes)
            .collect();
            c.send_raw(&header).await.unwrap();
            c.send(&Tensor::new(DType::F32, [1, 2, 1], vec![0; 8]))
                .await
                .unwrap();
            c.send(&Tensor::new(DType::I32, [1, 1, 1], vec![0; 4]))
                .await
                .unwrap();
            c.send(&Tensor::new(DType::F32, [1, 1, 1], vec![0; 4]))
                .await
                .unwrap();
        }
        send_expert_dispatch(&client, 1, 1, 1, 2, &[3., 4.], &[0])
            .await
            .unwrap();
        assert_eq!(
            recv_kind_server(&server).await.unwrap(),
            Some(FrameKind::FusedExpertDispatch)
        );
        assert!(recv_fused_expert_dispatch_body_server(&server)
            .await
            .is_err());
        assert_eq!(
            recv_kind_server(&server).await.unwrap(),
            Some(FrameKind::ExpertDispatch)
        );
        assert_eq!(
            recv_expert_dispatch_body_server(&server)
                .await
                .unwrap()
                .hidden,
            vec![3., 4.]
        );
        client.lock().await.close().await;
    });
}

#[test]
fn ordered_expert_replies_preserve_cancellation_across_three_and_twelve_workers() {
    // Partial sums on three workers produce 4; original gate order produces 3.
    for count in [3, 12] {
        let rt = runtime();
        let mut clients = Vec::new();
        let mut tasks = Vec::new();
        for wi in 0..count {
            let (server, client) = rt.block_on(loopback());
            clients.push(client);
            tasks.push(rt.spawn(async move {
                assert_eq!(
                    recv_kind_server(&server).await.unwrap(),
                    Some(FrameKind::ExpertDispatch)
                );
                let b = recv_expert_dispatch_body_server(&server).await.unwrap();
                let values: Vec<f32> = b
                    .ids
                    .iter()
                    .map(|&id| {
                        if id == EXPERT_PAD {
                            return 0.;
                        }
                        assert_eq!(id as usize % count, wi);
                        match id {
                            0 => 1e20,
                            1 => 1.,
                            3 => -1e20,
                            2 => 3.,
                            _ => 0.,
                        }
                    })
                    .collect();
                send_expert_result_ok(&server, b.rows, b.k, 1, &values)
                    .await
                    .unwrap();
            }));
        }
        let ep = EpClient::new(clients.clone(), rt.handle().clone(), 1, 12, 0).with_fused(false);
        let routes = [0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11]
            .map(|id| (id, 1.))
            .to_vec();
        assert_eq!(ep.dispatch(2, &[1.], &[routes]).unwrap(), [3.]);
        close_all(&rt, &clients);
        for task in tasks {
            rt.block_on(task).unwrap();
        }
    }
}

#[tokio::test]
async fn lossless_half_reply_preserves_all_finite_half_bits_and_falls_back_exactly() {
    use cascadia_engine_sparse_moe::dist::send_expert_result_lossless;
    let (server, client) = loopback().await;
    let values: Vec<f32> = (0..=u16::MAX)
        .map(half::f16::from_bits)
        .filter(|h| h.is_finite())
        .map(|h| h.to_f32() * 16.)
        .collect();
    assert!(
        send_expert_result_lossless(&server, 1, 1, values.len() as u32, &values, 4)
            .await
            .unwrap()
    );
    assert_eq!(
        recv_kind_client(&client).await.unwrap(),
        Some(FrameKind::ExpertResult)
    );
    let (back, shape) = recv_expert_result_body_client(&client)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(shape, [1, 1, values.len() as u32]);
    assert_eq!(
        back.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    // Arbitrary F32 precision is never silently quantized, nor is its exponent enlarged.
    let precise = [1.0000001f32, -0., 1e30];
    assert!(!send_expert_result_lossless(&server, 1, 1, 3, &precise, 4)
        .await
        .unwrap());
    assert_eq!(
        recv_kind_client(&client).await.unwrap(),
        Some(FrameKind::ExpertResult)
    );
    let (back, _) = recv_expert_result_body_client(&client)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        back.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        precise.map(f32::to_bits)
    );
    assert!(send_expert_result_lossless(&server, 1, 1, 3, &precise, 9)
        .await
        .is_err());
    assert!(
        send_expert_result_lossless(&server, 1, 1, 1, &[f32::NAN], 4)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn malformed_half_reply_is_rejected_and_drained_before_next_frame() {
    use cascadia_transport::{DType, Tensor};
    let (server, client) = loopback().await;
    for (exponent, dtype, data) in [
        (9, DType::F16, vec![0, 0]),
        (4, DType::F32, vec![0, 0, 0, 0]),
        (4, DType::F16, vec![0, 0x7c]),
    ] {
        let mut header = (FrameKind::ExpertResult as u32).to_be_bytes().to_vec();
        header.extend([2, exponent]);
        {
            let mut srv = server.lock().await;
            srv.send_raw(&header).await.unwrap();
            srv.send(&Tensor::new(dtype, [1, 1, 1], data))
                .await
                .unwrap();
        }
        assert_eq!(
            recv_kind_client(&client).await.unwrap(),
            Some(FrameKind::ExpertResult)
        );
        assert!(recv_expert_result_body_client(&client).await.is_err());
        send_expert_result_ok(&server, 1, 1, 1, &[3.])
            .await
            .unwrap();
        assert_eq!(
            recv_kind_client(&client).await.unwrap(),
            Some(FrameKind::ExpertResult)
        );
        assert_eq!(
            recv_expert_result_body_client(&client)
                .await
                .unwrap()
                .unwrap()
                .0,
            [3.]
        );
    }
}

/// Future pipeline drivers need independent sockets into one resident expert
/// bank. Exercise simultaneous callers, a peer-local error/close and a new
/// session without reloading weights. No CLI serving mode is enabled here.
#[test]
fn shared_bank_driver_sessions_are_independent_and_reusable() {
    let dir = export_dir();
    assert!(dir.join("manifest.json").is_file(), "tiny fixture required");
    let manifest = read_manifest(&dir).unwrap();
    let bank = Arc::new(load_expert_bank(&dir, 0, 1, ExpertsMode::Mmap).unwrap());
    let layer = (0..bank.num_layers())
        .find(|&l| bank.is_moe_layer(l))
        .unwrap() as u32;
    let rt = runtime();
    let session = |bank: Arc<ExpertBank>| {
        let (server, client) = rt.block_on(loopback());
        let mut engine = ExpertWorkerEngine::new_shared(bank, server, rt.handle().clone());
        let worker = std::thread::spawn(move || {
            loop {
                match engine.step() {
                    Ok(_) => {}
                    Err(e) if e.is_connection_fatal() => break engine.frames_served(),
                    Err(e) => panic!("shared worker session: {e}"),
                }
            }
        });
        let ep = EpClient::new(
            vec![client.clone()],
            rt.handle().clone(),
            manifest.hidden_size,
            manifest.num_experts,
            manifest.n_shared_experts,
        );
        (ep, client, worker)
    };
    let (first, c1, w1) = session(bank.clone());
    let (second, c2, w2) = session(bank.clone());
    let input: Vec<f32> = (0..manifest.hidden_size)
        .map(|i| (i as f32 - 3.0) * 0.01)
        .collect();
    let body = cascadia_engine_sparse_moe::dist::ExpertDispatchBody {
        layer,
        rows: 1,
        k: 1,
        hidden: input.clone(),
        hidden_shape: [1, manifest.hidden_size as u32, 1],
        ids: vec![0],
        ids_shape: [1, 1, 1],
    };
    let expected = bank.serve(&body).unwrap();
    std::thread::scope(|scope| {
        let a = scope.spawn(|| first.dispatch(layer, &input, &[vec![(0, 1.0)]]).unwrap());
        let b = scope.spawn(|| second.dispatch(layer, &input, &[vec![(0, 1.0)]]).unwrap());
        assert_eq!(a.join().unwrap(), expected);
        assert_eq!(b.join().unwrap(), expected);
    });
    // A complete bad request gets an error without corrupting either socket.
    assert!(
        first
            .dispatch(manifest.num_layers as u32 + 1, &input, &[vec![(0, 1.0)]])
            .is_err()
    );
    assert_eq!(
        second.dispatch(layer, &input, &[vec![(0, 1.0)]]).unwrap(),
        expected
    );
    close_all(&rt, &[c1]);
    assert_eq!(w1.join().unwrap(), 1);
    assert_eq!(
        second.dispatch(layer, &input, &[vec![(0, 1.0)]]).unwrap(),
        expected
    );
    let (third, c3, w3) = session(bank.clone());
    assert_eq!(
        third.dispatch(layer, &input, &[vec![(0, 1.0)]]).unwrap(),
        expected
    );
    close_all(&rt, &[c2, c3]);
    assert_eq!(w2.join().unwrap(), 3);
    assert_eq!(w3.join().unwrap(), 1);
    assert_eq!(
        Arc::strong_count(&bank),
        1,
        "sessions release the shared bank"
    );
}
