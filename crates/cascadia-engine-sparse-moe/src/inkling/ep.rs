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
//! driver reads no expert weights at all). CPU workers reproduce the
//! single-process [`MoeLayer`](super::moe::MoeLayer) bits. Optional per-expert
//! OpenVINO GPU workers preserve routing and accumulation order, but their
//! kernel numerics differ from CPU; compare them to a matching GPU reference.
//! `CASCADIA_INKLING_EP_REQUIRE_GPU=1` rejects missing GPU support/IRs and
//! forbids CPU fallback. Compact `CASCADIA_INKLING_EP_FUSED=1` workers return
//! individual GPU expert outputs for placement-independent gate-order sums.
//! `CASCADIA_INKLING_EP_FUSED_PARTIAL_SUMS=1` opts into weighted partial
//! sums and strict fused GPU shards; see [`super::ep_fused`].
//!
//! Default placement is deterministic and manifest-free: [`expert_home`]`(id, W) =
//! id % W`, with ids `0..n_routed` for routed experts and `n_routed + s` for
//! the shared ones. The layer index is ignored on purpose, so worker `k` owns
//! the same ids in every layer and `--ep-worker-index k` needs no table.
//!
//! An optional [`EpPlacement`] supplies capacity-checked replicas and calibrated
//! worker costs. Replica selection keeps each unique expert's rows together.
//! Wire: every involved worker gets only its active rows, padded with [`EXPERT_PAD`] to the max slots
//! any row needs from THAT worker. Frames carry at most
//! [`MAX_BATCH_COUNT`] rows; longer prefills are chunked by the driver. All
//! involved workers of a layer are dispatched and awaited together (one
//! `join_all` over per-worker futures, each locking only its own connection),
//! never serially. A worker that receives any other frame kind replies
//! `ExpertResult{status 1}` and keeps serving.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
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

use super::ep_placement::EpPlacement;
use super::ffn::AnyExpert;
use super::loader::{
    load_moe_experts, load_moe_experts_filtered, read_manifest, ExpertSet, InklingManifest,
};
use super::moe::seq_reads;
use super::ov_expert::OvExperts;
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
    placement: Option<Arc<EpPlacement>>,
    fused: bool,
}

/// One worker's share of a frame: the ids it serves per row, padded to `k`.
struct WorkerPlan {
    worker: usize,
    k: usize,
    ids: Vec<i32>,
    hidden_rows: Vec<f32>,
    weights: Vec<f32>,
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
            placement: None,
            // Compact GPU shards return individual expert outputs by default.
            // Weighted partial sums change f32 accumulation order with placement.
            fused: super::env_flag("CASCADIA_INKLING_EP_FUSED")
                && super::env_flag("CASCADIA_INKLING_EP_FUSED_PARTIAL_SUMS"),
        }
    }

    /// Opt in to explicit placement. Validate before loading/dispatching any
    /// weights, including when constructed outside the CLI.
    pub fn with_placement(
        mut self,
        placement: Arc<EpPlacement>,
        model: &InklingManifest,
    ) -> Result<Self, String> {
        placement.validate(model, self.workers.len())?;
        if self.hidden != model.hidden_size
            || self.n_routed != model.num_experts
            || self.n_shared != model.n_shared_experts
        {
            return Err("EP client dimensions do not match placement model".into());
        }
        self.placement = Some(placement);
        Ok(self)
    }

    /// Opt into the weighted partial-sum fused protocol, whose cross-worker sum
    /// runs on the driver in worker order (placement-dependent f32 order, hence
    /// off by default). It is NOT negotiated over the wire: every selected
    /// worker must itself run with fused GPU shards. A driver-on / worker-off
    /// mismatch is not silent — that worker rejects the fused frame and the
    /// dispatch fails naming it. Production derives this flag from the
    /// environment in `EpClient::new`; this setter is for tests and direct
    /// callers.
    pub fn with_fused(mut self, fused: bool) -> Self {
        self.fused = fused;
        self
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
    /// exactly that order from a zero row in raw mode (the op sequence of
    /// `MoeLayer::forward`, so the bytes match it). Fused mode sums weighted
    /// worker partials; its FP16/reduction numerics require tolerance checks.
    /// `Err` names the worker
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
            if list.iter().any(|&(_, w)| !w.is_finite()) {
                return Err(format!(
                    "layer {layer}: nonfinite routing weight in row {r}"
                ));
            }
            if self.fused
                && list
                    .iter()
                    .enumerate()
                    .any(|(i, (id, _))| list[..i].iter().any(|(prev, _)| prev == id))
            {
                return Err(format!("layer {layer}: duplicate expert in fused row {r}"));
            }
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
        let started = Instant::now();
        let homes = match &self.placement {
            Some(p) => p.assign(layer as usize, per_row)?,
            None => (0..self.n_routed + self.n_shared)
                .map(|id| expert_home(id, w))
                .collect(),
        };
        // Per worker, per row: the ids it serves, in gate order.
        let mut slots: Vec<Vec<Vec<(i32, f32)>>> = vec![vec![Vec::new(); n]; w];
        for (r, list) in per_row.iter().enumerate() {
            for &(id, weight) in list {
                slots[homes[id]][r].push((id as i32, weight));
            }
        }
        // Map original row -> compact row on each worker. Empty rows never
        // cross the network, including padded result vectors for those rows.
        let mut row_map = vec![vec![usize::MAX; n]; w];
        let plans: Vec<WorkerPlan> = slots
            .iter()
            .enumerate()
            .filter_map(|(wi, rows_ids)| {
                let k = rows_ids.iter().map(Vec::len).max().unwrap_or(0);
                if k == 0 {
                    return None; // uninvolved: no frame at all
                }
                let active = rows_ids.iter().filter(|s| !s.is_empty()).count();
                let mut ids = vec![EXPERT_PAD; active * k];
                let mut weights = vec![0.0; active * k];
                let mut hidden_rows = Vec::with_capacity(active * h);
                let mut compact = 0;
                for (r, s) in rows_ids.iter().enumerate() {
                    if s.is_empty() {
                        continue;
                    }
                    row_map[wi][r] = compact;
                    for (j, &(id, weight)) in s.iter().enumerate() {
                        ids[compact * k + j] = id;
                        weights[compact * k + j] = weight;
                    }
                    hidden_rows.extend_from_slice(&rows[r * h..(r + 1) * h]);
                    compact += 1;
                }
                Some(WorkerPlan {
                    worker: wi,
                    k,
                    ids,
                    hidden_rows,
                    weights,
                })
            })
            .collect();
        // All involved workers in flight together; each future locks only its
        // own connection. join_all (not try_join_all) so every reply is read
        // even when one worker fails — the other links stay frame-aligned.
        let results: Vec<Result<Vec<f32>, String>> =
            cascadia_runner::run_async(&self.handle, async {
                let futs = plans.iter().map(|p| {
                    let cli = Arc::clone(&self.workers[p.worker]);
                    let (wi, k, ids) = (p.worker, p.k as u32, &p.ids);
                    async move {
                        Self::round_trip(
                            &cli,
                            wi,
                            layer,
                            (p.hidden_rows.len() / h) as u32,
                            k,
                            h as u32,
                            &p.hidden_rows,
                            ids,
                            self.fused.then_some(p.weights.as_slice()),
                        )
                        .await
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
        if self.fused {
            // Deterministic worker-order reduction. Weights have already been
            // applied on the worker, without per-shard renormalization.
            out.fill(0.0);
            for (wi, result) in data.iter().enumerate() {
                let Some((_, values)) = result else { continue };
                for (r, &compact) in row_map[wi].iter().enumerate() {
                    if compact == usize::MAX {
                        continue;
                    }
                    for (o, &v) in out[r * h..(r + 1) * h]
                        .iter_mut()
                        .zip(&values[compact * h..(compact + 1) * h])
                    {
                        *o += v;
                    }
                }
            }
            return Ok(());
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
                let wi = homes[id];
                let (k, d) = data[wi]
                    .as_ref()
                    .expect("an id's home worker is always involved");
                let j = cursor[wi];
                cursor[wi] += 1;
                let compact = row_map[wi][r];
                let y = &d[(compact * k + j) * h..(compact * k + j + 1) * h];
                for (oo, &yi) in o.iter_mut().zip(y) {
                    *oo += wj * yi;
                }
            }
        }
        tracing::debug!(
            layer,
            rows = n,
            workers = plans.len(),
            sent_hidden_bytes = plans.iter().map(|p| p.hidden_rows.len() * 4).sum::<usize>(),
            result_bytes = plans.iter().map(|p| p.ids.len() * h * 4).sum::<usize>(),
            elapsed_us = started.elapsed().as_micros() as u64,
            "inkling expert dispatch complete"
        );
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
        weights: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let tag = format!("expert worker {wi}, layer {layer}");
        let sent = if let Some(weights) = weights {
            crate::dist::send_fused_expert_dispatch(
                cli,
                layer,
                rows,
                k,
                hidden,
                hidden_rows,
                ids,
                weights,
            )
            .await
        } else {
            send_expert_dispatch(cli, layer, rows, k, hidden, hidden_rows, ids).await
        };
        sent.map_err(|e| format!("{tag}: send dispatch: {e}"))?;
        let k = if weights.is_some() { 1 } else { k };
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
    ov: Option<OvExperts>,
    fused: Option<super::ep_fused::FusedExpertBank>,
    require_gpu: bool,
    gpu_name: Option<String>,
    cpu_calls: AtomicU64,
    cpu_f16_reference: bool,
    wire_f16_replies: AtomicU64,
    wire_f32_replies: AtomicU64,
    wire_tensor_bytes: AtomicU64,
    wire_f32_equivalent_bytes: AtomicU64,
}

type ExpertSlotOutputs = Vec<(usize, Vec<f32>)>;

impl ExpertBank {
    /// Qualification mode: fail instead of ever computing an expert on CPU.
    /// For a non-fused OpenVINO bank this checks up front that the device is a
    /// concrete GPU and that every owned expert has an IR. A fused bank already
    /// verified its GPU device and shard IRs in `FusedExpertBank::load`, so here
    /// it only latches the strict flag — its serve path never falls back to CPU
    /// regardless.
    pub fn require_gpu(mut self) -> Result<Self, String> {
        if self.fused.is_some() {
            self.require_gpu = true;
            return Ok(self);
        }
        let ov = self
            .ov
            .as_ref()
            .ok_or("GPU required but OpenVINO experts are not enabled/present")?;
        if ov.device() != "GPU" && !ov.device().starts_with("GPU.") {
            return Err(format!("GPU required, got device {}", ov.device()));
        }
        self.gpu_name = Some(
            cascadia_ov_genai_shim::device_full_name(ov.device())
                .map_err(|e| format!("GPU required but device query failed: {e}"))?,
        );
        for (layer, experts) in self.layers.iter().enumerate() {
            for &id in experts.keys() {
                if !ov.has_expert(layer as u32, id as u32, self.n_routed as u32) {
                    return Err(format!(
                        "GPU required but layer {layer} expert {id} has no IR"
                    ));
                }
            }
        }
        self.require_gpu = true;
        Ok(self)
    }

    pub fn backend_stats(&self) -> serde_json::Value {
        let stats = self.ov.as_ref().map(OvExperts::stats).unwrap_or_default();
        let (uncached_bytes, uncached_fallbacks) = super::read_buffers::uncached_read_statistics();
        serde_json::json!({"gpu_required":self.require_gpu,"gpu_name":self.gpu_name,
            "device":self.ov.as_ref().map(OvExperts::device),
            "cpu_calls":self.cpu_calls.load(Ordering::Relaxed),
            "cpu_f16_reference":self.cpu_f16_reference,
            "wire_f16_replies":self.wire_f16_replies.load(Ordering::Relaxed),
            "wire_f32_replies":self.wire_f32_replies.load(Ordering::Relaxed),
            "wire_tensor_bytes":self.wire_tensor_bytes.load(Ordering::Relaxed),
            "wire_f32_equivalent_bytes":self.wire_f32_equivalent_bytes.load(Ordering::Relaxed),
            "uncached_read_bytes":uncached_bytes,"uncached_read_fallbacks":uncached_fallbacks,
            "ov_successful_calls":stats.hits+stats.misses,"ov_cache_hits":stats.hits,
            "ov_cache_misses":stats.misses,"ov_fallbacks":stats.fallbacks,
            "fused":self.fused.as_ref().map(|f| f.stats())})
    }

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

    /// Serve one fused GPU dispatch: delegate to this worker's fused shard bank,
    /// which runs the compressed expert GEMMs on the GPU and returns one
    /// weighted partial per row (`[rows · hidden]`, k=1) already summed over the
    /// worker's owned experts in the request's gate order. `Err` (naming this
    /// worker) if the bank is not fused; the plain [`Self::serve`] path serves a
    /// non-fused dispatch.
    pub fn serve_fused(&self, b: &ExpertDispatchBody, weights: &[f32]) -> Result<Vec<f32>, String> {
        self.fused
            .as_ref()
            .ok_or("worker does not support fused GPU dispatch")?
            .serve(b, weights)
    }

    pub fn is_fused(&self) -> bool {
        self.fused.is_some()
    }

    /// Serve one dispatch: validate it, prefetch every requested expert, then
    /// `E_id(h_row)` for every non-pad `(row, slot)` — rayon over the slots,
    /// each with the exact per-expert kernel the local `MoeLayer` uses
    /// (`AnyExpert::forward`, or the overlapped whole-bin read +
    /// `MmapExpert::swiglu_from` for a paged-out expert, mirroring
    /// `MoeLayer::forward`; both CPU paths match the mmap kernel). Optional
    /// per-expert OpenVINO calls run first and have their own numerics.
    /// Returns `[rows · k · hidden]` with zeros in pad slots. `Err` is the
    /// status-1 reply text, naming this worker and the layer.
    ///
    /// A fused bank has no per-expert CPU/OpenVINO path here: it early-returns
    /// its raw K=1 expert outputs via `FusedExpertBank::serve_raw`, leaving the
    /// driver to apply routing weights ([`Self::serve_fused`] is the
    /// weight-applied fused reply).
    pub fn serve(&self, b: &ExpertDispatchBody) -> Result<Vec<f32>, String> {
        if let Some(fused) = &self.fused {
            return fused.serve_raw(b);
        }
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
        // Group slots by expert, so a prefill reads each expert once and uses
        // those bytes for all its rows before releasing the temporary buffer.
        let mut occurrences: HashMap<usize, Vec<usize>> = HashMap::new();
        for (slot, &id) in b.ids.iter().enumerate() {
            if id == EXPERT_PAD {
                continue;
            }
            if id < 0 {
                return Err(format!("{tag}: invalid expert id {id}"));
            }
            let id = id as usize;
            if !occurrences.contains_key(&id) && !table.contains_key(&id) {
                return Err(format!(
                    "{tag}: does not own expert {id} (this bank holds {:?})",
                    self.owned_ids(layer)
                ));
            }
            occurrences.entry(id).or_default().push(slot);
        }
        let inter = self.inter;
        let ov = self.ov.as_ref().filter(|ov| ov.has_layer(b.layer));
        let streamed_cpu = super::env_flag("CASCADIA_INKLING_EP_STREAM_CPU");
        let compute = |(&id, slots): (&usize, &Vec<usize>)| {
            let e = &table[&id];
            let mut cpu_ready = false;
            let mut buf = None;
            let mut stream_buf = None;
            let mut f16_reference = None;
            slots
                .iter()
                .map(|&slot| {
                    let x = &b.hidden[(slot / k) * h..(slot / k + 1) * h];
                    let gpu =
                        ov.and_then(|ov| ov.expert(b.layer, id as u32, self.n_routed as u32, x));
                    let y = match gpu {
                        Some(y) => y,
                        None if self.require_gpu => {
                            return Err(format!(
                                "{tag}: GPU expert {id} failed; CPU fallback forbidden"
                            ))
                        }
                        None => {
                            // A configured GPU expert that returns no output is
                            // a silent-numerics hazard: without REQUIRE_GPU we
                            // recompute it on CPU, which differs from the device
                            // bit-for-bit. Surface the first such fallback (once
                            // per bank) on both a subscriber-less bench
                            // (eprintln) and production logs (warn); the running
                            // total lives in backend_stats().cpu_calls.
                            if self.cpu_calls.fetch_add(1, Ordering::Relaxed) == 0 && ov.is_some() {
                                let note = format!(
                                    "{tag}: GPU expert {id} produced no output; \
                                     recomputing on CPU (device numerics differ). \
                                     Set CASCADIA_INKLING_EP_REQUIRE_GPU=1 to fail instead."
                                );
                                warn!("{note}");
                                eprintln!("{note}");
                            }
                            if !cpu_ready {
                                if streamed_cpu {
                                    if let Some(m) = e.as_mmap() {
                                        let mut lease =
                                            super::read_buffers::ReadBuffers::acquire(1);
                                        lease.buffers[0].read(m.bin_path(), m.bin_len()).map_err(
                                            |e| {
                                                format!(
                                                    "{tag}: expert {id} streamed read failed: {e}"
                                                )
                                            },
                                        )?;
                                        if self.cpu_f16_reference {
                                            f16_reference =
                                                Some(super::f16_reference::Prepared::new(
                                                    lease.buffers[0].as_slice(),
                                                    h,
                                                    inter,
                                                )?);
                                        }
                                        stream_buf = Some(lease);
                                    }
                                } else {
                                    e.prefetch();
                                    buf = e
                                        .as_mmap()
                                        .filter(|m| !seq_reads() && !m.mostly_resident())
                                        .and_then(|m| m.read_bytes().ok());
                                }
                                cpu_ready = true;
                            }
                            match (stream_buf.as_ref(), buf.as_ref(), e.as_mmap()) {
                                (Some(lease), _, Some(m)) => match &f16_reference {
                                    Some(reference) => {
                                        reference.forward(lease.buffers[0].as_slice(), x)
                                    }
                                    None => m.swiglu_from(lease.buffers[0].as_slice(), x),
                                },
                                (_, Some(buf), Some(m)) => m.swiglu_from(buf, x),
                                _ => e.forward(x, h, inter),
                            }
                        }
                    };
                    Ok((slot, y))
                })
                .collect()
        };
        // Nested Rayon GEMVs may suspend outer tasks while their read buffers —
        // streamed leases, or the whole-bin `read_bytes` buffer taken on a
        // paged-out expert — remain live. Fixed cohorts cap retained buffers at
        // eight experts on both the streamed and direct paths, even when a long
        // prefill selects hundreds of experts. Slot outputs are written by
        // index below, so cohorting does not change the result.
        let entries: Vec<_> = occurrences.iter().collect();
        let mut outputs: Vec<ExpertSlotOutputs> = Vec::with_capacity(entries.len());
        for cohort in entries.chunks(8) {
            let part: Result<Vec<_>, String> = cohort.par_iter().copied().map(compute).collect();
            outputs.extend(part?);
        }
        let computed: Result<Vec<ExpertSlotOutputs>, String> = Ok(outputs);
        let mut out = vec![0.0f32; rows * k * h];
        for (slot, y) in computed?.into_iter().flatten() {
            out[slot * h..(slot + 1) * h].copy_from_slice(&y);
        }
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
    load_expert_bank_with_placement(dir, index, count, mode, None)
}

pub fn load_expert_bank_with_placement(
    dir: &Path,
    index: u32,
    count: u32,
    mode: ExpertsMode,
    placement: Option<&EpPlacement>,
) -> Result<ExpertBank, LoadError> {
    if count == 0 || index >= count {
        return Err(LoadError::Manifest(format!(
            "expert worker index {index} of {count} is out of range"
        )));
    }
    let m = read_manifest(dir)?;
    let own = super::env_flag("CASCADIA_INKLING_EP_OWN_EXPERTS");
    let cpu_f16_reference = super::env_flag("CASCADIA_INKLING_EP_CPU_F16_REFERENCE");
    if cpu_f16_reference
        && (own
            || !super::env_flag("CASCADIA_INKLING_EP_STREAM_CPU")
            || super::env_flag("CASCADIA_INKLING_EP_FUSED")
            || super::env_flag("CASCADIA_INKLING_OV_EXPERTS"))
    {
        return Err(LoadError::Manifest(
            "diagnostic FP16 reference requires streamed CPU experts only".into(),
        ));
    }
    if own && placement.is_none() {
        return Err(LoadError::Manifest(
            "CASCADIA_INKLING_EP_OWN_EXPERTS requires a capacity-checked EP placement".into(),
        ));
    }
    if let Some(p) = placement {
        p.validate(&m, count as usize)
            .map_err(LoadError::Manifest)?;
    }
    let t0 = Instant::now();
    let mut layers = Vec::with_capacity(m.num_layers);
    let mut moe = Vec::with_capacity(m.num_layers);
    for li in 0..m.num_layers {
        if m.dense_layers.contains(&li) {
            layers.push(HashMap::new());
            moe.push(false);
            continue;
        }
        let set = match placement {
            Some(p) => {
                load_moe_experts_filtered(dir, &m, li, mode, |id| p.owns(li, id, index as usize))?
            }
            None => load_moe_experts(dir, &m, li, mode, ExpertSet::Shard { index, count })?,
        };
        // The plan must not understate actual packed bytes to pass its capacity
        // check. Eager fixtures have no mapping; production uses mmap int4.
        if let Some(p) = placement {
            if let Some((id, _)) = set.iter().find(|(_, e)| {
                e.as_mmap()
                    .is_some_and(|m| m.bin_len() as u64 > p.expert_bytes)
                    || e.owned_int4_bytes() as u64 > p.expert_bytes
            }) {
                return Err(LoadError::Manifest(format!(
                    "EP layer {li} expert {id}: file exceeds placement expert_bytes"
                )));
            }
        }
        let set = if own {
            set.into_iter()
                .map(|(id, e)| e.into_owned_int4().map(|e| (id, e)))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            set
        };
        if cpu_f16_reference && set.iter().any(|(_, e)| e.as_mmap().is_none()) {
            return Err(LoadError::Manifest(
                "diagnostic FP16 reference requires packed int4 bins".into(),
            ));
        }
        layers.push(set.into_iter().collect());
        moe.push(true);
    }
    let fused = if super::env_flag("CASCADIA_INKLING_EP_FUSED") {
        if own {
            return Err(LoadError::Manifest(
                "fused EP must not retain owned CPU weights".into(),
            ));
        }
        let fused_dir = std::env::var_os("CASCADIA_INKLING_EP_FUSED_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| dir.join("moe_ep").join(format!("worker_{index:02}")));
        let budget = std::env::var("CASCADIA_INKLING_EP_FUSED_CACHE_MB")
            .unwrap_or_else(|_| "4096".into())
            .parse::<u64>()
            .ok()
            .and_then(|n| n.checked_mul(1 << 20))
            .filter(|&n| n > 0)
            .ok_or_else(|| LoadError::Manifest("invalid fused IR cache budget".into()))?;
        let ownership: Vec<Vec<usize>> = layers
            .iter()
            .map(|l: &HashMap<usize, AnyExpert>| {
                let mut ids: Vec<usize> = l.keys().copied().collect();
                ids.sort_unstable();
                ids
            })
            .collect();
        Some(
            super::ep_fused::FusedExpertBank::load(&fused_dir, &m, &ownership, budget)
                .map_err(LoadError::Manifest)?,
        )
    } else {
        None
    };
    let bank = ExpertBank {
        layers,
        moe,
        hidden: m.hidden_size,
        inter: m.moe_intermediate,
        n_routed: m.num_experts,
        n_shared: m.n_shared_experts,
        index,
        count,
        ov: if fused.is_none() {
            OvExperts::from_env(dir, m.hidden_size)
        } else {
            None
        },
        fused,
        require_gpu: false,
        gpu_name: None,
        cpu_calls: AtomicU64::new(0),
        cpu_f16_reference,
        wire_f16_replies: AtomicU64::new(0),
        wire_f32_replies: AtomicU64::new(0),
        wire_tensor_bytes: AtomicU64::new(0),
        wire_f32_equivalent_bytes: AtomicU64::new(0),
    };
    let bank = if super::env_flag("CASCADIA_INKLING_EP_REQUIRE_GPU") {
        bank.require_gpu().map_err(LoadError::Manifest)?
    } else {
        bank
    };
    info!(
        index,
        count,
        experts = bank.n_experts(),
        moe_layers = bank.moe.iter().filter(|&&b| b).count(),
        mode = ?mode,
        owned_packed_weights = own,
        backend = %bank.backend_stats(),
        elapsed_ms = t0.elapsed().as_millis() as u64,
        "inkling expert bank loaded"
    );
    Ok(bank)
}

/// Cool-off after a failed frame so a misbehaving peer cannot make the relay
/// loop hot-spin (the pipeline worker's value).
const WORKER_BACKOFF: Duration = Duration::from_millis(200);

#[cfg(test)]
mod gpu_requirement_tests {
    use super::*;

    fn fixture() -> ExpertBank {
        load_expert_bank(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling_export"),
            0,
            1,
            ExpertsMode::Mmap,
        )
        .unwrap()
    }

    #[test]
    fn requiring_gpu_rejects_a_cpu_bank() {
        let mut bank = fixture();
        bank.ov = None;
        assert!(bank.require_gpu().is_err());
    }

    #[test]
    fn strict_dispatch_never_falls_back_to_cpu() {
        let mut bank = fixture();
        bank.require_gpu = true;
        bank.ov = None; // Emulate a missing/disabled backend at dispatch time.
        let body = ExpertDispatchBody {
            layer: 1,
            rows: 1,
            k: 1,
            hidden: vec![0.; bank.hidden],
            hidden_shape: [1, bank.hidden as u32, 1],
            ids: vec![0],
            ids_shape: [1, 1, 1],
        };
        assert!(bank
            .serve(&body)
            .unwrap_err()
            .contains("CPU fallback forbidden"));
        assert_eq!(bank.cpu_calls.load(Ordering::Relaxed), 0);
    }
}

/// The engine an expert worker rank runs: `step()` serves one
/// `ExpertDispatch` frame (recv → compute → reply), like
/// `PipelineEngine::step_worker`; driven by `Runner::run_relay_loop`. Takes
/// no tasks. A clean close by the driver latches `peer_disconnected` and
/// `step()` surfaces a connection-fatal `Err` exactly once, so the relay loop
/// exits for a supervisor rebuild (the driver's connection is accepted at
/// `connect()` only).
pub struct ExpertWorkerEngine {
    bank: Arc<ExpertBank>,
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
        Self::new_shared(Arc::new(bank), server, handle)
    }

    /// Serve an independent driver connection using an already loaded bank.
    /// Each connection owns its framing/disconnect state; weights and backend
    /// caches are shared. This does not enable a listener or change the CLI's
    /// single-driver topology. A serving-mode controller must bound accepted
    /// sessions and own their lifetime before using this constructor.
    pub fn new_shared(
        bank: Arc<ExpertBank>,
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
            FrameKind::FusedExpertDispatch => {
                let (body, weights, shape) = self
                    .block_on(crate::dist::recv_fused_expert_dispatch_body_server(&server))
                    .map_err(|e| format!("recv fused body: {e}"))?;
                let served = if shape != [body.rows, body.k, 1] {
                    Err("invalid fused routing weights shape".into())
                } else {
                    self.bank.serve_fused(&body, &weights)
                };
                match served {
                    Ok(out) => {
                        self.block_on(send_expert_result_ok(
                            &server,
                            body.rows,
                            1,
                            self.bank.hidden as u32,
                            &out,
                        ))
                        .map_err(|e| format!("send fused result: {e}"))?;
                        self.frames += 1;
                    }
                    Err(msg) => {
                        warn!(layer = body.layer, "{msg}");
                        self.block_on(send_expert_result_err(&server, &msg))
                            .map_err(|e| e.to_string())?;
                    }
                }
                Ok(())
            }
            FrameKind::ExpertDispatch => {
                let body = self
                    .block_on(recv_expert_dispatch_body_server(&server))
                    .map_err(|e| format!("recv dispatch body: {e}"))?;
                let t0 = Instant::now();
                let served = self.bank.serve(&body);
                let compute = t0.elapsed();
                match served {
                    Ok(out) => {
                        let exponent = self
                            .bank
                            .fused
                            .as_ref()
                            .and_then(|f| f.output_exponent(body.layer));
                        let compact = if let Some(exponent) = exponent
                            .filter(|_| super::env_flag("CASCADIA_INKLING_EP_FUSED_F16_WIRE"))
                        {
                            self.block_on(crate::dist::send_expert_result_lossless(
                                &server,
                                body.rows,
                                body.k,
                                body.hidden_size() as u32,
                                &out,
                                exponent,
                            ))
                            .map_err(|e| format!("send lossless result: {e}"))?
                        } else {
                            self.block_on(send_expert_result_ok(
                                &server,
                                body.rows,
                                body.k,
                                body.hidden_size() as u32,
                                &out,
                            ))
                            .map_err(|e| format!("send result: {e}"))?;
                            false
                        };
                        if compact {
                            &self.bank.wire_f16_replies
                        } else {
                            &self.bank.wire_f32_replies
                        }
                        .fetch_add(1, Ordering::Relaxed);
                        self.bank.wire_tensor_bytes.fetch_add(
                            (out.len() * if compact { 2 } else { 4 }) as u64,
                            Ordering::Relaxed,
                        );
                        self.bank
                            .wire_f32_equivalent_bytes
                            .fetch_add((out.len() * 4) as u64, Ordering::Relaxed);
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
            "expert worker {}/{} is stateless and serves expert dispatch frames only; got {kind:?}",
            self.bank.index, self.bank.count
        );
        warn!("{msg}");
        self.block_on(send_expert_result_err(server, &msg))
            .map_err(|e| format!("send error result: {e}"))
    }
}

impl Engine for ExpertWorkerEngine {
    fn warmup(&mut self) {
        // Banks are open; fused layers compile lazily on their first dispatch.
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
