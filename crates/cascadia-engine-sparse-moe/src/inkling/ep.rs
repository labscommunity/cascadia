//! Expert-parallel (EP) dispatch for the Inkling family — the star topology
//! sized in `docs/perf/INKLING_SCALING.md` §4.
//!
//! One **driver** (rank 0 of a `total = 1` pipeline: it runs every layer's
//! attention, norms, convs, router, dense MLPs, embed and head) plus `W`
//! **expert workers**. A worker holds, for every MoE layer, the experts homed
//! on it — nothing else, and no sequence state. Per MoE layer the driver
//! routes locally, sends each involved worker the rows' hidden states plus
//! the expert ids it must serve ([`crate::dist::FrameKind::ExpertDispatch`]),
//! and receives each expert's RAW output back
//! ([`crate::dist::FrameKind::ExpertResult`]). The driver applies the gate
//! weights and sums **in gate order** (routed, then the shared experts with
//! their gammas — the shared experts are dispatched like routed ones, so the
//! driver reads no expert weights at all), which makes the result
//! bit-identical to the single-process [`MoeLayer`](super::moe::MoeLayer).
//!
//! Placement is deterministic and manifest-free: [`expert_home`]`(id, W) =
//! id % W`, with ids `0..n_routed` for routed experts and `n_routed + s` for
//! the shared ones. The layer index is ignored on purpose, so worker `k` owns
//! the same ids in every layer and `--ep-worker-index k` needs no table.
//!
//! Wire: every involved worker gets every row of the frame (a row is 24 KB;
//! the simplicity is worth it), padded with [`EXPERT_PAD`] to the max slots
//! any row needs from THAT worker. Frames carry at most
//! [`MAX_BATCH_COUNT`] rows; longer prefills are chunked by the driver. All
//! involved workers of a layer are dispatched and awaited together (one
//! `join_all` over per-worker futures, each locking only its own connection),
//! never serially. A worker that receives any other frame kind replies
//! `ExpertResult{status 1}` and keeps serving.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cascadia_engine::{Engine, EngineError, EngineResult};
use cascadia_transport::{
    frame_idle_ceiling, recv_timeout, ActivationClient, ActivationServer, MAX_TENSOR_BYTES,
    PREFILL_REPLY_TIMEOUT_FACTOR,
};
use cascadia_types::{Chunk, GenerationTask, TaskId};
use rayon::prelude::*;
use tokio::sync::Mutex as TokioMutex;
use tracing::{info, warn};

use super::ffn::AnyExpert;
use super::loader::{load_moe_experts, read_manifest, ExpertSet};
use super::moe::seq_reads;
use crate::dist::{
    recv_expert_dispatch_body_server, recv_expert_result_body_client, recv_key_body_server,
    recv_kind_client, recv_kind_server, send_expert_dispatch, send_expert_result_err,
    send_expert_result_ok, ExpertDispatchBody, FrameKind, EXPERT_PAD, MAX_BATCH_COUNT,
};
use crate::dsv4::loader::{ExpertsMode, LoadError};

/// The worker that serves expert `expert` (routed `0..n_routed`, shared
/// `n_routed + s`) in every MoE layer: `expert % n_workers`.
pub fn expert_home(expert: usize, n_workers: usize) -> usize {
    assert!(n_workers > 0, "expert_home: n_workers must be >= 1");
    expert % n_workers
}

/// Margin the widened prefill reply deadline stays under the transport's
/// frame-idle ceiling by, so our deadline (which keeps the socket usable for
/// an error report) fires before the transport's own idle drop.
const PREFILL_DEADLINE_MARGIN: Duration = Duration::from_secs(5);

/// Bound on one worker's reply for a frame of `rows` rows — the design rule
/// of [`crate::dist::recv_token_reply`]: an owed reply gets a strict deadline,
/// never the idle ceiling. Decode (one row) uses the per-hop activation
/// timeout; a batched frame runs up to [`MAX_BATCH_COUNT`] rows through the
/// worker's experts first, so it gets the same widening the pipeline's
/// batched prefill uses, clamped under the frame-idle ceiling.
fn reply_deadline(rows: u32) -> Duration {
    let base = recv_timeout();
    if rows <= 1 {
        return base;
    }
    let widened = base.saturating_mul(PREFILL_REPLY_TIMEOUT_FACTOR);
    match frame_idle_ceiling() {
        Some(ceiling) => widened.min(ceiling.saturating_sub(PREFILL_DEADLINE_MARGIN).max(base)),
        None => widened,
    }
}

// ───────────────────────────────── driver side ─────────────────────────────────

/// Driver side: one connection per expert worker; all of a layer's dispatches
/// in flight together. Shared (`Arc`) by every MoE layer of the driver's
/// [`InklingRunner`](super::stage::InklingRunner).
pub struct EpClient {
    workers: Vec<Arc<TokioMutex<ActivationClient>>>,
    handle: tokio::runtime::Handle,
    hidden: usize,
    n_routed: usize,
    n_shared: usize,
}

/// One worker's share of a frame: the ids it serves per row, padded to `k`.
struct WorkerPlan {
    worker: usize,
    k: usize,
    ids: Vec<i32>,
}

impl EpClient {
    /// `workers[i]` is the connection to worker `i` of `workers.len()` (the
    /// index every [`expert_home`] refers to). `hidden` / `n_routed` /
    /// `n_shared` are the manifest's; the runner checks them against the
    /// layers it attaches this client to.
    pub fn new(
        workers: Vec<Arc<TokioMutex<ActivationClient>>>,
        handle: tokio::runtime::Handle,
        hidden: usize,
        n_routed: usize,
        n_shared: usize,
    ) -> Self {
        assert!(hidden > 0, "EpClient: hidden must be > 0");
        assert!(n_routed > 0, "EpClient: n_routed must be > 0");
        Self {
            workers,
            handle,
            hidden,
            n_routed,
            n_shared,
        }
    }

    pub fn n_workers(&self) -> usize {
        self.workers.len()
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn n_routed(&self) -> usize {
        self.n_routed
    }

    pub fn n_shared(&self) -> usize {
        self.n_shared
    }

    /// Evaluate one MoE layer's experts on the workers. `rows` is `[T, hidden]`;
    /// `per_row[t]` lists `(expert id, weight)` in GATE ORDER — the routed
    /// selection first, then the shared experts as `n_routed + s` with their
    /// gammas. Returns `[T, hidden]`: per row `Σ w · E(h)` accumulated in
    /// exactly that order from a zero row (the op sequence of
    /// `MoeLayer::forward`, so the bytes match it). `Err` names the worker
    /// index and the layer on any worker / transport failure; every involved
    /// worker has been awaited by then, so no reply is left unread on any
    /// connection. Rows are chunked so a frame never exceeds
    /// [`MAX_BATCH_COUNT`] rows or the transport's tensor cap.
    pub fn dispatch(
        &self,
        layer: u32,
        rows: &[f32],
        per_row: &[Vec<(usize, f32)>],
    ) -> Result<Vec<f32>, String> {
        let (h, t) = (self.hidden, per_row.len());
        if self.workers.is_empty() {
            return Err(format!("layer {layer}: expert client has no workers"));
        }
        if rows.len() != t * h {
            return Err(format!(
                "layer {layer}: rows.len() {} != {t} rows × hidden {h}",
                rows.len()
            ));
        }
        let n_ids = self.n_routed + self.n_shared;
        let mut k_max = 0usize;
        for (r, list) in per_row.iter().enumerate() {
            k_max = k_max.max(list.len());
            if let Some(&(id, _)) = list.iter().find(|&&(id, _)| id >= n_ids) {
                return Err(format!(
                    "layer {layer}: row {r} routes to expert {id} >= {n_ids} (routed {} + shared {})",
                    self.n_routed, self.n_shared
                ));
            }
        }
        let mut out = vec![0.0f32; t * h];
        if t == 0 || k_max == 0 {
            return Ok(out);
        }
        // Rows per frame: the frame cap, and the reply tensor [rows, k, hidden]
        // must fit the transport's per-tensor cap.
        let by_bytes = (MAX_TENSOR_BYTES / (k_max * h * 4)).max(1);
        let chunk = (MAX_BATCH_COUNT as usize).min(by_bytes);
        let mut lo = 0;
        while lo < t {
            let hi = (lo + chunk).min(t);
            self.dispatch_chunk(
                layer,
                &rows[lo * h..hi * h],
                &per_row[lo..hi],
                &mut out[lo * h..hi * h],
            )?;
            lo = hi;
        }
        Ok(out)
    }

    /// One frame (≤ `MAX_BATCH_COUNT` rows) to every involved worker, awaited
    /// together, then the gate-order accumulation into `out` (zeroed here).
    fn dispatch_chunk(
        &self,
        layer: u32,
        rows: &[f32],
        per_row: &[Vec<(usize, f32)>],
        out: &mut [f32],
    ) -> Result<(), String> {
        let (h, n, w) = (self.hidden, per_row.len(), self.workers.len());
        // Per worker, per row: the ids it serves, in gate order.
        let mut slots: Vec<Vec<Vec<i32>>> = vec![vec![Vec::new(); n]; w];
        for (r, list) in per_row.iter().enumerate() {
            for &(id, _) in list {
                slots[expert_home(id, w)][r].push(id as i32);
            }
        }
        let plans: Vec<WorkerPlan> = slots
            .iter()
            .enumerate()
            .filter_map(|(wi, rows_ids)| {
                let k = rows_ids.iter().map(Vec::len).max().unwrap_or(0);
                if k == 0 {
                    return None; // uninvolved: no frame at all
                }
                let mut ids = vec![EXPERT_PAD; n * k];
                for (r, s) in rows_ids.iter().enumerate() {
                    ids[r * k..r * k + s.len()].copy_from_slice(s);
                }
                Some(WorkerPlan { worker: wi, k, ids })
            })
            .collect();
        let (rows_u32, h_u32) = (n as u32, h as u32);
        // All involved workers in flight together; each future locks only its
        // own connection. join_all (not try_join_all) so every reply is read
        // even when one worker fails — the other links stay frame-aligned.
        let results: Vec<Result<Vec<f32>, String>> =
            cascadia_runner::run_async(&self.handle, async {
                let futs =
                    plans.iter().map(|p| {
                        let cli = Arc::clone(&self.workers[p.worker]);
                        let (wi, k, ids) = (p.worker, p.k as u32, &p.ids);
                        async move {
                            Self::round_trip(&cli, wi, layer, rows_u32, k, h_u32, rows, ids).await
                        }
                    });
                futures::future::join_all(futs).await
            });
        let mut data: Vec<Option<(usize, Vec<f32>)>> = (0..w).map(|_| None).collect();
        let mut errs = Vec::new();
        for (p, r) in plans.iter().zip(results) {
            match r {
                Ok(v) => data[p.worker] = Some((p.k, v)),
                Err(e) => errs.push(e),
            }
        }
        if !errs.is_empty() {
            return Err(errs.join("; "));
        }
        // Accumulate exactly like MoeLayer::forward: a zero row, then
        // `out += w · E(h)` per (id, w) in gate order. A worker's slots for a
        // row were filled in that same order, so slot j of worker wi is the
        // j-th id of this row that homes on wi.
        let mut cursor = vec![0usize; w];
        for (r, list) in per_row.iter().enumerate() {
            cursor.iter_mut().for_each(|c| *c = 0);
            let o = &mut out[r * h..(r + 1) * h];
            o.fill(0.0);
            for &(id, wj) in list {
                let wi = expert_home(id, w);
                let (k, d) = data[wi]
                    .as_ref()
                    .expect("an id's home worker is always involved");
                let j = cursor[wi];
                cursor[wi] += 1;
                let y = &d[(r * k + j) * h..(r * k + j + 1) * h];
                for (oo, &yi) in o.iter_mut().zip(y) {
                    *oo += wj * yi;
                }
            }
        }
        Ok(())
    }

    /// Send one worker its frame and await its result, bounded by
    /// [`reply_deadline`]. On timeout the connection is dropped (a late reply
    /// carries no sequence number and would be read as the next layer's).
    #[allow(clippy::too_many_arguments)]
    async fn round_trip(
        cli: &TokioMutex<ActivationClient>,
        wi: usize,
        layer: u32,
        rows: u32,
        k: u32,
        hidden: u32,
        hidden_rows: &[f32],
        ids: &[i32],
    ) -> Result<Vec<f32>, String> {
        let tag = format!("expert worker {wi}, layer {layer}");
        send_expert_dispatch(cli, layer, rows, k, hidden, hidden_rows, ids)
            .await
            .map_err(|e| format!("{tag}: send dispatch: {e}"))?;
        let deadline = reply_deadline(rows);
        let reply = tokio::time::timeout(deadline, async {
            match recv_kind_client(cli).await {
                Ok(Some(FrameKind::ExpertResult)) => recv_expert_result_body_client(cli)
                    .await
                    .map_err(|e| format!("{tag}: recv result: {e}")),
                Ok(Some(other)) => Err(format!("{tag}: expected ExpertResult, got {other:?}")),
                Ok(None) => Err(format!("{tag}: connection closed before the result")),
                Err(e) => Err(format!("{tag}: recv result kind: {e}")),
            }
        })
        .await;
        match reply {
            Ok(Ok(Ok((data, shape)))) => {
                let want = rows as usize * k as usize * hidden as usize;
                if shape != [rows, k, hidden] || data.len() != want {
                    return Err(format!(
                        "{tag}: result shape {shape:?} ({} values) != [{rows}, {k}, {hidden}]",
                        data.len()
                    ));
                }
                Ok(data)
            }
            Ok(Ok(Err(msg))) => Err(format!("{tag}: {msg}")),
            Ok(Err(e)) => Err(e),
            Err(_) => {
                // The inner future — and any guard it held mid-read — is gone,
                // so re-locking cannot deadlock.
                cli.lock().await.close().await;
                Err(format!(
                    "{tag}: no result within {deadline:?} (dead worker?); connection dropped so a \
                     late reply cannot be read as the next layer's"
                ))
            }
        }
    }
}

// ───────────────────────────────── worker side ─────────────────────────────────

/// Worker side: the expert bank for one shard of every MoE layer — per
/// absolute layer, the experts (routed and shared, by id) homed on this
/// worker. Dense layers hold nothing.
pub struct ExpertBank {
    layers: Vec<HashMap<usize, AnyExpert>>,
    moe: Vec<bool>,
    hidden: usize,
    inter: usize,
    n_routed: usize,
    n_shared: usize,
    index: u32,
    count: u32,
}

impl ExpertBank {
    /// This worker's index of `count()`.
    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    /// The model's layer count (the bank is indexed by absolute layer).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn inter(&self) -> usize {
        self.inter
    }

    pub fn n_routed(&self) -> usize {
        self.n_routed
    }

    pub fn n_shared(&self) -> usize {
        self.n_shared
    }

    pub fn is_moe_layer(&self, layer: usize) -> bool {
        self.moe.get(layer).copied().unwrap_or(false)
    }

    /// The expert ids this bank holds for `layer`, ascending (empty for a
    /// dense layer).
    pub fn owned_ids(&self, layer: usize) -> Vec<usize> {
        let mut ids: Vec<usize> = self
            .layers
            .get(layer)
            .map(|t| t.keys().copied().collect())
            .unwrap_or_default();
        ids.sort_unstable();
        ids
    }

    pub fn expert(&self, layer: usize, id: usize) -> Option<&AnyExpert> {
        self.layers.get(layer).and_then(|t| t.get(&id))
    }

    /// Experts held across all layers.
    pub fn n_experts(&self) -> usize {
        self.layers.iter().map(HashMap::len).sum()
    }

    /// Serve one dispatch: validate it, prefetch every requested expert, then
    /// `E_id(h_row)` for every non-pad `(row, slot)` — rayon over the slots,
    /// each with the exact per-expert kernel the local `MoeLayer` uses
    /// (`AnyExpert::forward`, or the overlapped whole-bin read +
    /// `MmapExpert::swiglu_from` for a one-row decode frame, mirroring
    /// `MoeLayer::forward`; both are bit-identical to the mmap kernel).
    /// Returns `[rows · k · hidden]` with zeros in pad slots. `Err` is the
    /// status-1 reply text, naming this worker and the layer.
    pub fn serve(&self, b: &ExpertDispatchBody) -> Result<Vec<f32>, String> {
        let tag = format!(
            "expert worker {}/{}: layer {}",
            self.index, self.count, b.layer
        );
        let (rows, k, h) = (b.rows as usize, b.k as usize, self.hidden);
        let layer = b.layer as usize;
        let table = match self.layers.get(layer) {
            Some(t) if self.moe[layer] => t,
            Some(_) => return Err(format!("{tag}: dense layer has no experts")),
            None => {
                return Err(format!(
                    "{tag}: layer out of range (model has {} layers)",
                    self.layers.len()
                ))
            }
        };
        if rows == 0 || rows > MAX_BATCH_COUNT as usize {
            return Err(format!(
                "{tag}: rows {rows} out of range 1..={MAX_BATCH_COUNT}"
            ));
        }
        if k == 0 {
            return Err(format!("{tag}: k must be >= 1"));
        }
        if b.hidden_shape != [rows as u32, h as u32, 1] || b.hidden.len() != rows * h {
            return Err(format!(
                "{tag}: hidden tensor shape {:?} ({} values) != [{rows}, {h}, 1]",
                b.hidden_shape,
                b.hidden.len()
            ));
        }
        if b.ids_shape != [rows as u32, k as u32, 1] || b.ids.len() != rows * k {
            return Err(format!(
                "{tag}: ids tensor shape {:?} ({} values) != [{rows}, {k}, 1]",
                b.ids_shape,
                b.ids.len()
            ));
        }
        // Resolve every requested expert once: ownership check + prefetch.
        let mut unique: Vec<usize> = Vec::new();
        let mut seen = HashSet::new();
        for &id in &b.ids {
            if id == EXPERT_PAD {
                continue;
            }
            if id < 0 {
                return Err(format!("{tag}: invalid expert id {id}"));
            }
            let id = id as usize;
            if seen.insert(id) {
                if !table.contains_key(&id) {
                    return Err(format!(
                        "{tag}: does not own expert {id} (home is worker {} of {}; this bank holds {:?})",
                        expert_home(id, self.count as usize),
                        self.count,
                        self.owned_ids(layer)
                    ));
                }
                unique.push(id);
            }
        }
        for &id in &unique {
            table[&id].prefetch();
        }
        // Decode (one row): `MoeLayer::forward`'s reads — a paged-out expert
        // is streamed whole, concurrently; a resident one is computed straight
        // off the mmap. A batch computes off the mmap like `forward_batch`
        // (each expert's pages are touched once per frame).
        let bufs: HashMap<usize, Vec<u8>> =
            if rows == 1 && !seq_reads() && unique.iter().any(|id| table[id].as_mmap().is_some()) {
                unique
                    .par_iter()
                    .filter_map(|&id| {
                        table[&id]
                            .as_mmap()
                            .filter(|m| !m.mostly_resident())
                            .and_then(|m| m.read_bytes().ok())
                            .map(|buf| (id, buf))
                    })
                    .collect()
            } else {
                HashMap::new()
            };
        let inter = self.inter;
        let mut out = vec![0.0f32; rows * k * h];
        out.par_chunks_mut(h).enumerate().for_each(|(s, o)| {
            let id = b.ids[s];
            if id == EXPERT_PAD {
                return; // pad slot: zeros, no work
            }
            let id = id as usize;
            let x = &b.hidden[(s / k) * h..(s / k + 1) * h];
            let e = &table[&id];
            let y = match (bufs.get(&id), e.as_mmap()) {
                (Some(buf), Some(m)) => m.swiglu_from(buf, x),
                _ => e.forward(x, h, inter),
            };
            o.copy_from_slice(&y);
        });
        Ok(out)
    }
}

/// Open worker `index` of `count`'s expert shard for every MoE layer of the
/// export at `dir`: the ids with [`expert_home`]`(id, count) == index`, shared
/// ids included, in `mode`. No shells, edge tables or sequence state.
pub fn load_expert_bank(
    dir: &Path,
    index: u32,
    count: u32,
    mode: ExpertsMode,
) -> Result<ExpertBank, LoadError> {
    if count == 0 || index >= count {
        return Err(LoadError::Manifest(format!(
            "expert worker index {index} of {count} is out of range"
        )));
    }
    let m = read_manifest(dir)?;
    let t0 = Instant::now();
    let mut layers = Vec::with_capacity(m.num_layers);
    let mut moe = Vec::with_capacity(m.num_layers);
    for li in 0..m.num_layers {
        if m.dense_layers.contains(&li) {
            layers.push(HashMap::new());
            moe.push(false);
            continue;
        }
        let set = load_moe_experts(dir, &m, li, mode, ExpertSet::Shard { index, count })?;
        layers.push(set.into_iter().collect());
        moe.push(true);
    }
    let bank = ExpertBank {
        layers,
        moe,
        hidden: m.hidden_size,
        inter: m.moe_intermediate,
        n_routed: m.num_experts,
        n_shared: m.n_shared_experts,
        index,
        count,
    };
    info!(
        index,
        count,
        experts = bank.n_experts(),
        moe_layers = bank.moe.iter().filter(|&&b| b).count(),
        mode = ?mode,
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "inkling expert bank loaded"
    );
    Ok(bank)
}

/// Cool-off after a failed frame so a misbehaving peer cannot make the relay
/// loop hot-spin (the pipeline worker's value).
const WORKER_BACKOFF: Duration = Duration::from_millis(200);

/// The engine an expert worker rank runs: `step()` serves one
/// `ExpertDispatch` frame (recv → compute → reply), like
/// `PipelineEngine::step_worker`; driven by `Runner::run_relay_loop`. Takes
/// no tasks. A clean close by the driver latches `peer_disconnected` and
/// `step()` surfaces a connection-fatal `Err` exactly once, so the relay loop
/// exits for a supervisor rebuild (the driver's connection is accepted at
/// `connect()` only).
pub struct ExpertWorkerEngine {
    bank: ExpertBank,
    server: Arc<TokioMutex<ActivationServer>>,
    handle: tokio::runtime::Handle,
    peer_disconnected: bool,
    disconnect_reported: bool,
    frames: u64,
}

impl ExpertWorkerEngine {
    pub fn new(
        bank: ExpertBank,
        server: Arc<TokioMutex<ActivationServer>>,
        handle: tokio::runtime::Handle,
    ) -> Self {
        Self {
            bank,
            server,
            handle,
            peer_disconnected: false,
            disconnect_reported: false,
            frames: 0,
        }
    }

    pub fn bank(&self) -> &ExpertBank {
        &self.bank
    }

    /// Frames served so far.
    pub fn frames_served(&self) -> u64 {
        self.frames
    }

    fn block_on<F: std::future::Future>(&self, fut: F) -> F::Output {
        cascadia_runner::run_async(&self.handle, fut)
    }

    /// Serve one frame. `Ok(())` after a clean close too (the flag is
    /// latched); `Err` is a transport failure or an unrecoverable framing
    /// problem — `step()` classifies it.
    fn serve_one(&mut self) -> Result<(), String> {
        let server = Arc::clone(&self.server);
        let kind = self
            .block_on(recv_kind_server(&server))
            .map_err(|e| format!("recv_kind: {e}"))?;
        let Some(kind) = kind else {
            info!(
                index = self.bank.index,
                count = self.bank.count,
                frames = self.frames,
                "driver closed the connection; expert worker idling"
            );
            self.peer_disconnected = true;
            return Ok(());
        };
        match kind {
            FrameKind::ExpertDispatch => {
                let body = self
                    .block_on(recv_expert_dispatch_body_server(&server))
                    .map_err(|e| format!("recv dispatch body: {e}"))?;
                let t0 = Instant::now();
                let served = self.bank.serve(&body);
                let compute = t0.elapsed();
                match served {
                    Ok(out) => {
                        self.block_on(send_expert_result_ok(
                            &server,
                            body.rows,
                            body.k,
                            body.hidden_size() as u32,
                            &out,
                        ))
                        .map_err(|e| format!("send result: {e}"))?;
                        self.frames += 1;
                        tracing::trace!(
                            layer = body.layer,
                            rows = body.rows,
                            k = body.k,
                            compute_us = compute.as_micros() as u64,
                            "expert dispatch served"
                        );
                    }
                    Err(msg) => {
                        warn!(layer = body.layer, rows = body.rows, k = body.k, "{msg}");
                        self.block_on(send_expert_result_err(&server, &msg))
                            .map_err(|e| format!("send error result: {e}"))?;
                    }
                }
                Ok(())
            }
            // Known one-way frames with a fixed body: consume it so the stream
            // stays aligned, then say no.
            k @ (FrameKind::RestorePrefix | FrameKind::CachePrefix) => {
                let _ = self
                    .block_on(recv_key_body_server(&server))
                    .map_err(|e| format!("recv {k:?} body: {e}"))?;
                self.reject(&server, k)
            }
            other => self.reject(&server, other),
        }
    }

    fn reject(
        &mut self,
        server: &TokioMutex<ActivationServer>,
        kind: FrameKind,
    ) -> Result<(), String> {
        let msg = format!(
            "expert worker {}/{} is stateless and serves ExpertDispatch only; got {kind:?}",
            self.bank.index, self.bank.count
        );
        warn!("{msg}");
        self.block_on(send_expert_result_err(server, &msg))
            .map_err(|e| format!("send error result: {e}"))
    }
}

impl Engine for ExpertWorkerEngine {
    fn warmup(&mut self) {
        // Nothing to compile; the bank is already open. Log what we serve.
        info!(
            index = self.bank.index,
            count = self.bank.count,
            experts = self.bank.n_experts(),
            "inkling expert worker ready"
        );
    }

    fn submit(&mut self, task: GenerationTask) -> EngineResult<()> {
        Err(EngineError::Backend(format!(
            "expert worker takes no tasks (got {}); submit to the driver",
            task.task_id
        )))
    }

    fn step(&mut self) -> EngineResult<Vec<(TaskId, Chunk)>> {
        if self.peer_disconnected {
            if !self.disconnect_reported {
                self.disconnect_reported = true;
                return Err(EngineError::NotConnected);
            }
            std::thread::sleep(WORKER_BACKOFF);
            return Ok(Vec::new());
        }
        if let Err(e) = self.serve_one() {
            let e = format!("expert worker {}/{}: {e}", self.bank.index, self.bank.count);
            warn!("{e}");
            let err = EngineError::Backend(e);
            if err.is_connection_fatal() {
                // The driver's socket is gone; only a rebuild re-accepts one.
                self.peer_disconnected = true;
                self.disconnect_reported = true;
                return Err(err);
            }
            std::thread::sleep(WORKER_BACKOFF);
            return Ok(Vec::new());
        }
        // Mirror PipelineEngine::step: a clean close surfaces as a
        // connection-fatal Err exactly once, so the relay loop exits.
        if self.peer_disconnected && !self.disconnect_reported {
            self.disconnect_reported = true;
            return Err(EngineError::NotConnected);
        }
        Ok(Vec::new())
    }

    fn cancel(&mut self, _task_id: &TaskId) {}

    fn close(&mut self) {
        let server = Arc::clone(&self.server);
        self.block_on(async move {
            server.lock().await.close().await;
        });
        self.peer_disconnected = true;
        self.disconnect_reported = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_home_is_modulo_and_covers_every_worker() {
        for w in 1..=7usize {
            let mut seen = vec![0usize; w];
            for id in 0..258usize {
                let h = expert_home(id, w);
                assert_eq!(h, id % w);
                seen[h] += 1;
            }
            assert!(seen.iter().all(|&n| n > 0));
        }
    }

    #[test]
    fn reply_deadline_widens_for_batches_but_stays_under_the_ceiling() {
        let one = reply_deadline(1);
        let batch = reply_deadline(64);
        assert!(batch >= one);
        if let Some(c) = frame_idle_ceiling() {
            assert!(batch < c);
        }
    }
}
