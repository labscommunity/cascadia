//! Continuous-batching skeleton for the sparse-MoE engine.
//!
//! **Status:** iter 059 shipped the skeleton (slots + `plan_step`);
//! iter 078 (this iter) wires per-request sampling state (blocker 3).
//! The remaining blockers (1: per-request KV slabs; 2: shell forward
//! `[N,1,H]` signature; 4: API admission rewrite) are still out of
//! scope — they each need their own design review + bench.
//!
//! This module defines the request-queue + batch-assembly + per-request
//! KV-slot accounting that a real continuous batcher needs, with a
//! `BatchPlan` data structure produced per decode step and a
//! `ContinuousBatcher` that owns the active slots. **No batched tensor
//! work happens here.** The runner-level batched forward (which would
//! actually call `forward_layer0_step` / `forward_shells` /
//! `forward_head_last` over N requests in lock-step) is intentionally
//! not implemented — that depends on the batched-shells primitives
//! (iter 048 + `forward_shells_multi`; iter 051 +
//! `forward_shells_multi_batched_experts`) which are still on the
//! `autolab/k26-perf` research branch.
//!
//! ## Why most of this is still a skeleton
//!
//! Implementing the full thing requires four parallel changes that each
//! need their own design review + benchmark:
//!
//! 1. **Per-request KV slabs.** Today `LayerState` owns one
//!    `[NUM_HEADS, capacity, HEAD_DIM]` buffer and one `past_seq_len`
//!    counter that the whole engine shares. Batching N requests means N
//!    independent slabs that the same shell forward must attend over
//!    (or N separate shell calls with different K/V — see "blockers"
//!    below). The choice between "padded slab" (`[N, NUM_HEADS,
//!    max_seq, HEAD_DIM]` with a mask) and "ragged paged-attention"
//!    (vLLM-style block table) is load-bearing and deserves its own PR.
//!
//! 2. **Shell forward signature change.** `forward_shells` takes one
//!    `[1, 1, HIDDEN]` tile and one `past_seq_len`. Batching needs
//!    `[N, 1, HIDDEN]` plus a per-request `past_seq_len[N]`. The
//!    underlying int4 kernel (`shell_forward_decode_int4_with_capacity`)
//!    is hard-coded `seq=1`. The autolab branch's `forward_shells_multi`
//!    adds a `seq>1` path but treats those tokens as belonging to ONE
//!    request (spec-decode), not N. We need a real multi-request
//!    variant.
//!
//! 3. **Per-request sampling state.** [DONE — iter 078.] The batcher
//!    now owns a parallel `samplers: Vec<Option<SamplerState>>` keyed
//!    by `slot_idx`, and `sample_for_plan(planned, logits_per_slot)`
//!    runs `sampling::sample` once per planned slot with that slot's
//!    `(rng, history=slot.generated)` state. The rep-penalty history
//!    is read directly from `RequestSlot.generated` (already
//!    maintained by `commit_step`); the RNG is lazy-seeded from
//!    `sampling.seed` on first sample to match the single-request
//!    engine path's bit-for-bit determinism. Vectorizing the
//!    rep-penalty / softmax / categorical-pick across N rows is still
//!    a TODO — `sample_for_plan` calls the scalar sampler N times.
//!    That's fine for now: sampling is ~µs vs ~ms shells, so the loop
//!    is not on the critical path.
//!
//! 4. **API server admission control.** `tahoma-api` today serializes
//!    requests behind a `Semaphore(MAX_CONCURRENT)` and the runner
//!    grabs the engine mutex per `step()`. The continuous-batching
//!    contract is "submit any time, get tokens out as soon as a slot
//!    opens" — that needs the API to hand requests directly to the
//!    batcher and the batcher to demux chunks to the right SSE stream.
//!    Touches `tahoma-api/src/lib.rs`, `tahoma-runner/src/lib.rs`, and
//!    the `Engine` trait (would gain `submit_streaming` or similar).
//!
//! What this module DOES give you:
//!
//! - A `BatchPlan` value the engine could produce per step describing
//!   exactly which slots run, what their past_seq_len is, and what the
//!   sampling/eos config is for each.
//! - Slot allocation + freeing logic that respects a `max_concurrent`
//!   cap (the "N" in the design).
//! - The seam where prefill (variable prompt length per slot) would
//!   join with decode (1 token per slot per step) — same `BatchPlan`
//!   shape, just `tokens_this_step > 1` for slots in prefill.
//! - A unit-tested `plan_step` that proves the assembly logic at N=2
//!   without any tensor backend at all.
//! - A `sample_for_plan` that runs the per-slot sampler with
//!   independent RNG + history state, so two concurrent slots produce
//!   two independent token streams from identical logits — even with
//!   different seeds, temperatures, or rep-penalty histories.

use std::collections::HashMap;

use tahoma_types::{GenerationTask, TaskId};

use crate::sampling::{self, SamplingConfig};

/// Hard cap on concurrent slots. Picked at construction time so the
/// per-slot KV slab footprint is bounded.
///
/// At Kimi K2.6 sizes — 60 MoE layers × 2 (K+V) × `NUM_HEADS=128` ×
/// `(QK_HEAD_DIM=192 + V_HEAD_DIM=128) / 2` × 4 B (f32) × max_seq, plus
/// layer-0 — one slot at max_seq=4096 is ~30 GB of KV before
/// quantization. The MVP `MAX_BATCH_SLOTS_DEFAULT=4` exists to keep the
/// per-slot accounting honest without committing the engine to a slab
/// layout yet.
pub const MAX_BATCH_SLOTS_DEFAULT: usize = 4;

/// Per-request state held across decode steps. Owned by the batcher;
/// reads-only from the engine's `step_batch`.
#[derive(Clone, Debug)]
pub struct RequestSlot {
    /// Stable id used to demux output chunks back to the API stream.
    pub task_id: TaskId,
    /// Tokenized prompt. Owned by the slot so we can drop the original
    /// `GenerationTask` (and its `String` prompt) after admission.
    pub prompt_ids: Vec<i64>,
    /// Token ids generated so far (excludes prompt, excludes EOS).
    pub generated: Vec<i64>,
    /// Past key/value sequence length already in the per-slot KV slab.
    /// Counts prefill + decode tokens that have been written.
    pub past_seq_len: usize,
    /// Maximum number of new tokens this request should produce.
    pub max_new: usize,
    /// Sampling config snapshotted at admission. Per-request so two
    /// concurrent slots can use different temperature / seed / etc.
    pub sampling: SamplingConfig,
    /// Current phase. Drives whether `plan_step` schedules the slot for
    /// prefill (variable `tokens_this_step`) or decode (fixed 1).
    pub phase: SlotPhase,
}

/// What state of the request lifecycle a slot is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotPhase {
    /// Still chewing through prompt tokens. `past_seq_len <
    /// prompt_ids.len()` — the next step would advance prefill, not
    /// emit a generated token.
    Prefill,
    /// All prompt tokens are in KV; each step samples one new token.
    Decode,
    /// Slot is done (EOS or hit `max_new`). The batcher will free it
    /// at the next `gc` call. Keeps the slot in `slots` for one extra
    /// step so the caller can pick up the final chunk.
    Done {
        /// `true` if termination was triggered by EOS, `false` if we
        /// hit `max_new`. Surfaced in the final-chunk `finish_reason`.
        eos: bool,
    },
}

impl RequestSlot {
    /// Construct a freshly-admitted slot from a task + its tokenized
    /// prompt + the engine-side sampling config.
    pub fn new(task: &GenerationTask, prompt_ids: Vec<i64>, sampling: SamplingConfig) -> Self {
        Self {
            task_id: task.task_id.clone(),
            prompt_ids,
            generated: Vec::new(),
            past_seq_len: 0,
            max_new: task.max_tokens.max(1) as usize,
            sampling,
            phase: SlotPhase::Prefill,
        }
    }

    /// Have we placed every prompt token into the KV cache?
    pub fn prefill_done(&self) -> bool {
        self.past_seq_len >= self.prompt_ids.len()
    }

    /// Have we generated as many tokens as the caller asked for?
    pub fn max_new_reached(&self) -> bool {
        self.generated.len() >= self.max_new
    }
}

/// A single slot's contribution to one batched decode step.
///
/// `BatchPlan::slots` is a `Vec<PlannedSlot>` — the engine's
/// `step_batch` walks this list once to gather inputs, runs the shared
/// shell + per-request sampling, then commits the per-slot KV writes +
/// generated tokens.
#[derive(Clone, Debug)]
pub struct PlannedSlot {
    /// Index into `ContinuousBatcher::slots` — stable across one step,
    /// invalidated by `gc()`.
    pub slot_idx: usize,
    /// The token ids this slot consumes on this step. Always length 1
    /// in decode; up to `prefill_chunk` in prefill.
    pub input_ids: Vec<i64>,
    /// The slot's current `past_seq_len` BEFORE this step. The engine
    /// must advance it by `input_ids.len()` after writing the present
    /// K/V into the slab.
    pub past_seq_len: usize,
    /// Sampling config to apply when picking the next token. Cloned
    /// per-step so the engine doesn't need a borrow back into the
    /// batcher.
    pub sampling: SamplingConfig,
    /// `true` if the engine should sample + emit a token on this step.
    /// `false` for all-but-the-last prefill step on this slot
    /// (intermediate prefill states are discarded).
    pub sample_this_step: bool,
}

/// Output of `ContinuousBatcher::plan_step`. Either an empty plan
/// (nothing to do) or a non-empty batch the engine should run.
#[derive(Clone, Debug, Default)]
pub struct BatchPlan {
    /// Per-slot work for this step. `slots.len()` is the effective `N`
    /// the engine should run; it is bounded above by
    /// `ContinuousBatcher::max_slots` and may be smaller (some slots
    /// finished, queue is short, etc.).
    pub slots: Vec<PlannedSlot>,
}

impl BatchPlan {
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// Per-slot sampler state. One of these per active `RequestSlot`.
///
/// **What this owns** (iter 078, blocker 3):
///
/// - `rng`: the xorshift64* state used by `sampling::sample` for the
///   categorical pick. Seeded lazily on first sample so deterministic
///   `seed=Some(_)` reproduces across runs. Zero is treated as
///   "unseeded" — `sampling::init_rng` will replace it on first use.
/// - `rng_seeded`: tracks whether the lazy init has happened.
///
/// **What it does NOT own** — by design:
///
/// - The repetition-penalty history. That lives on `RequestSlot.generated`
///   and is read directly by `sample_for_plan`. Two reasons: (a) it's
///   already populated by `commit_step`, no need to duplicate; (b) the
///   token-stream is the source of truth for what "this slot has emitted".
/// - The sampling config. That's snapshotted on the slot at admission
///   (`RequestSlot.sampling`) and copied into each `PlannedSlot` so the
///   sampler can read it without borrowing back into the batcher.
///
/// Lifecycle: created (via `Default`) when `submit()` (or
/// `fill_from_pending`) places a task into a free slot; dropped (its
/// index set to `None`) when `gc()` frees the slot.
#[derive(Clone, Debug, Default)]
struct SamplerState {
    rng: u64,
    rng_seeded: bool,
}

/// Continuous batcher. Owns the active slot pool + pending queue.
///
/// **Not** the engine itself — the engine would hold this and call
/// `plan_step()` before each batched forward, then `commit_step()`
/// after the forward returns sampled tokens. Errors are not strongly
/// typed yet because this is the skeleton — the wiring iter would
/// thread a real `BatchError` enum.
pub struct ContinuousBatcher {
    /// Hard cap on simultaneously-active slots. KV slab pre-allocation
    /// happens at this width per layer.
    pub max_slots: usize,
    /// Active slots indexed by position; `slots[i]` is what
    /// `PlannedSlot { slot_idx: i, .. }` refers to. Sparse over time
    /// (slots are freed in place by `gc()` but the vec is not
    /// compacted within a step so `slot_idx`s stay stable).
    slots: Vec<Option<RequestSlot>>,
    /// Per-slot sampler state, index-parallel with `slots`. `samplers[i]`
    /// is `Some` exactly when `slots[i]` is `Some`. Owned by the batcher
    /// (not the slot) so the sampler can be reset independently — e.g. a
    /// future iter that supports prompt-cache-reuse across slots would
    /// keep the slot but wipe the sampler.
    samplers: Vec<Option<SamplerState>>,
    /// Reverse lookup so `submit()` can no-op on duplicate `task_id`.
    by_task: HashMap<TaskId, usize>,
    /// Pending slot-less tasks. Drained into `slots` at `plan_step`
    /// time, FIFO, up to the free-slot count.
    pending: Vec<(GenerationTask, Vec<i64>, SamplingConfig)>,
    /// How many prefill tokens to advance per slot per step. `1`
    /// preserves the existing rainier-quality decode pattern (no
    /// shape-specialization surprises). Higher values would batch
    /// prefill at the cost of (a) needing multi-token shell forward
    /// and (b) potentially blocking decode-phase slots in the same
    /// batch on the heavier prefill compute.
    pub prefill_chunk: usize,
}

impl ContinuousBatcher {
    pub fn new(max_slots: usize) -> Self {
        let max_slots = max_slots.max(1);
        Self {
            max_slots,
            slots: (0..max_slots).map(|_| None).collect(),
            samplers: (0..max_slots).map(|_| None).collect(),
            by_task: HashMap::new(),
            pending: Vec::new(),
            prefill_chunk: 1,
        }
    }

    /// Admit a tokenized task. Either lands in a free slot immediately
    /// or queues in `pending` until `plan_step` drains it.
    ///
    /// Returns `Err` if the task_id is already known (duplicate
    /// submission) so the caller can decide whether to no-op or fail.
    pub fn submit(
        &mut self,
        task: GenerationTask,
        prompt_ids: Vec<i64>,
        sampling: SamplingConfig,
    ) -> Result<(), &'static str> {
        if self.by_task.contains_key(&task.task_id) {
            return Err("duplicate task_id");
        }
        if let Some(free) = self.first_free_slot() {
            let slot = RequestSlot::new(&task, prompt_ids, sampling);
            self.by_task.insert(slot.task_id.clone(), free);
            self.slots[free] = Some(slot);
            // Parallel sampler-state init. `rng` stays 0 until the first
            // `sample_for_plan` call seeds it from `sampling.seed` — see
            // `SamplerState` rationale for the lazy seeding contract.
            self.samplers[free] = Some(SamplerState::default());
        } else {
            self.pending.push((task, prompt_ids, sampling));
        }
        Ok(())
    }

    /// Walk active slots + drain pending into free slots; return one
    /// `BatchPlan` describing what every slot should do this step.
    ///
    /// Slots in `Done` are skipped (waiting for the next `gc()`); slots
    /// in `Prefill` advance up to `prefill_chunk` prompt tokens; slots
    /// in `Decode` advance exactly one token.
    pub fn plan_step(&mut self) -> BatchPlan {
        // Pull pending tasks into any newly-free slots first so they
        // get included in this step rather than waiting one extra
        // round trip.
        self.fill_from_pending();
        let mut planned = Vec::new();
        for (i, slot) in self.slots.iter().enumerate() {
            let Some(slot) = slot.as_ref() else { continue };
            match slot.phase {
                SlotPhase::Done { .. } => continue,
                SlotPhase::Prefill => {
                    let remaining = slot.prompt_ids.len().saturating_sub(slot.past_seq_len);
                    if remaining == 0 {
                        // Edge case: prompt was empty. Promote
                        // straight to decode with no input.
                        continue;
                    }
                    let take = remaining.min(self.prefill_chunk.max(1));
                    let start = slot.past_seq_len;
                    let input_ids: Vec<i64> = slot.prompt_ids[start..start + take].to_vec();
                    let consumes_last = start + take == slot.prompt_ids.len();
                    planned.push(PlannedSlot {
                        slot_idx: i,
                        input_ids,
                        past_seq_len: slot.past_seq_len,
                        sampling: slot.sampling.clone(),
                        // Only the prefill step that consumes the LAST
                        // prompt token yields the first generated
                        // token — matches the single-request driver in
                        // `drive_generation_first`.
                        sample_this_step: consumes_last,
                    });
                }
                SlotPhase::Decode => {
                    // Decode samples one new token from the last
                    // generated (or last prompt if `generated`
                    // empty). The engine reads `slot.generated.last()`
                    // for the input, or falls back to the slot's
                    // bookkeeping if needed.
                    let input = match slot.generated.last() {
                        Some(&t) => t,
                        None => *slot.prompt_ids.last().unwrap_or(&0),
                    };
                    planned.push(PlannedSlot {
                        slot_idx: i,
                        input_ids: vec![input],
                        past_seq_len: slot.past_seq_len,
                        sampling: slot.sampling.clone(),
                        sample_this_step: true,
                    });
                }
            }
        }
        BatchPlan { slots: planned }
    }

    /// Apply one step's results back into the slots. The engine calls
    /// this with the sampled token (`None` if `sample_this_step` was
    /// false) plus an EOS-hit flag.
    ///
    /// Returns the slot indices whose phase transitioned to `Done` this
    /// step; the caller can flush a final chunk for each.
    pub fn commit_step(
        &mut self,
        planned: &[PlannedSlot],
        sampled: &[StepOutcome],
        eos_ids: &[i64],
    ) -> Vec<usize> {
        assert_eq!(planned.len(), sampled.len(), "outcomes/plan mismatch");
        let mut newly_done = Vec::new();
        for (p, outcome) in planned.iter().zip(sampled.iter()) {
            let Some(slot) = self.slots.get_mut(p.slot_idx).and_then(|s| s.as_mut()) else {
                continue;
            };
            // Advance KV bookkeeping. Engine has already written the
            // present K/V into the slab, so we just bump the counter.
            slot.past_seq_len = p.past_seq_len + p.input_ids.len();

            if let Some(token) = outcome.sampled {
                // Was this slot's first emit? Then it transitions
                // Prefill -> Decode (the final prefill step that
                // consumed the last prompt token).
                if matches!(slot.phase, SlotPhase::Prefill) {
                    slot.phase = SlotPhase::Decode;
                }
                if eos_ids.contains(&token) {
                    slot.phase = SlotPhase::Done { eos: true };
                    newly_done.push(p.slot_idx);
                    continue;
                }
                slot.generated.push(token);
                if slot.max_new_reached() {
                    slot.phase = SlotPhase::Done { eos: false };
                    newly_done.push(p.slot_idx);
                    continue;
                }
            }

            // Slot might have just finished prefill without emitting
            // (if `sample_this_step` was false — multi-token prefill
            // batch). Promote phase if so.
            if matches!(slot.phase, SlotPhase::Prefill) && slot.prefill_done() {
                slot.phase = SlotPhase::Decode;
            }
        }
        newly_done
    }

    /// Sample one token per planned slot, applying per-slot sampler
    /// state (RNG + rep-penalty history) independently. This is the
    /// blocker-3 wiring from `perf/continuous-batching-059`'s docstring:
    /// the existing single-rank sampler in `SparseMoEEngine` holds one
    /// `(history, rng)` shared across every request; this routes it
    /// per-slot so two concurrent requests with different seeds /
    /// histories / temperatures produce two independent token streams.
    ///
    /// Contract:
    ///
    /// - `planned.len() == logits_per_slot.len()`. Caller must pass one
    ///   logits vector per planned slot, in the same order.
    /// - For each planned slot:
    ///   - If `sample_this_step == false`: returns
    ///     `StepOutcome::none()`. The slot's sampler is NOT touched so
    ///     intermediate prefill steps don't advance the RNG (matters for
    ///     deterministic-seed reproducibility — the rainier reference
    ///     only invokes RNG on the steps that actually sample).
    ///   - If `sample_this_step == true`: looks up the slot's
    ///     `SamplerState`, lazy-seeds the RNG from `planned.sampling.seed`
    ///     on first use, then calls `sampling::sample(logits,
    ///     &slot.generated, &planned.sampling, &mut sampler.rng)` and
    ///     returns the result wrapped in `StepOutcome::token`.
    /// - The rep-penalty history is `slot.generated` (already maintained
    ///   by `commit_step`), so callers should run this BEFORE
    ///   `commit_step` for the same plan — otherwise the just-emitted
    ///   token would be in the rep-penalty window for itself, which is a
    ///   one-token off-by-one vs the existing single-request path.
    ///
    /// Panics if a planned slot's `slot_idx` points to a freed slot
    /// (`samplers[i] == None`); the batcher's invariants make this
    /// impossible if you pass back the plan from `plan_step` without
    /// intervening `gc()` calls.
    pub fn sample_for_plan(
        &mut self,
        planned: &[PlannedSlot],
        logits_per_slot: &[Vec<f32>],
    ) -> Vec<StepOutcome> {
        assert_eq!(
            planned.len(),
            logits_per_slot.len(),
            "sample_for_plan: one logits row per planned slot"
        );
        let mut out = Vec::with_capacity(planned.len());
        for (p, logits) in planned.iter().zip(logits_per_slot.iter()) {
            if !p.sample_this_step {
                out.push(StepOutcome::none());
                continue;
            }
            // Pull history out of the slot first (immutable borrow), then
            // mutate the sampler — separate fields so this is safe.
            let history: Vec<i64> = match self.slots.get(p.slot_idx).and_then(|s| s.as_ref()) {
                Some(slot) => slot.generated.clone(),
                None => {
                    // Slot got GC'd between plan and sample. Treat as
                    // "nothing to sample" rather than panicking — caller
                    // can choose to skip the chunk.
                    out.push(StepOutcome::none());
                    continue;
                }
            };
            let sampler = self
                .samplers
                .get_mut(p.slot_idx)
                .and_then(|s| s.as_mut())
                .expect("sample_for_plan: sampler missing for active slot");
            if !sampler.rng_seeded {
                sampler.rng = sampling::init_rng(p.sampling.seed);
                sampler.rng_seeded = true;
            }
            let token = sampling::sample(logits, &history, &p.sampling, &mut sampler.rng);
            out.push(StepOutcome::token(token));
        }
        out
    }

    /// Convenience: borrow a slot's sampler RNG state. Test-only hook
    /// so unit tests can verify independent advancement without
    /// reaching into the batcher's private fields.
    #[cfg(test)]
    fn sampler_rng(&self, idx: usize) -> Option<u64> {
        self.samplers
            .get(idx)
            .and_then(|s| s.as_ref())
            .map(|s| s.rng)
    }

    /// Free slots in `Done` phase, freeing their KV slabs for new
    /// admissions. Returns the freed `task_id`s in undefined order.
    pub fn gc(&mut self) -> Vec<TaskId> {
        let mut freed = Vec::new();
        for i in 0..self.slots.len() {
            let take =
                matches!(&self.slots[i], Some(s) if matches!(s.phase, SlotPhase::Done { .. }));
            if take {
                let s = self.slots[i].take().unwrap();
                // Drop the matching sampler state. Parallel-index
                // invariant: samplers[i].is_some() iff slots[i].is_some().
                self.samplers[i] = None;
                self.by_task.remove(&s.task_id);
                freed.push(s.task_id);
            }
        }
        freed
    }

    /// How many slots are currently occupied (any phase).
    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Length of the pending admission queue.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Borrow an active slot for inspection (tests + the engine when
    /// it needs to gather inputs not in `PlannedSlot`).
    pub fn slot(&self, idx: usize) -> Option<&RequestSlot> {
        self.slots.get(idx).and_then(|s| s.as_ref())
    }

    fn first_free_slot(&self) -> Option<usize> {
        self.slots.iter().position(|s| s.is_none())
    }

    fn fill_from_pending(&mut self) {
        while !self.pending.is_empty() {
            let Some(free) = self.first_free_slot() else {
                break;
            };
            let (task, prompt_ids, sampling) = self.pending.remove(0);
            let slot = RequestSlot::new(&task, prompt_ids, sampling);
            self.by_task.insert(slot.task_id.clone(), free);
            self.slots[free] = Some(slot);
            // Same init as `submit`'s direct-fill path. See
            // `SamplerState::default` for why rng=0 is OK pre-seed.
            self.samplers[free] = Some(SamplerState::default());
        }
    }
}

/// What the engine's batched forward gives back per planned slot.
///
/// `sampled = None` is allowed: a multi-token prefill step that hasn't
/// yet consumed the last prompt token doesn't emit a sample.
#[derive(Clone, Debug)]
pub struct StepOutcome {
    pub sampled: Option<i64>,
}

impl StepOutcome {
    pub fn token(t: i64) -> Self {
        Self { sampled: Some(t) }
    }
    pub fn none() -> Self {
        Self { sampled: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: &str, prompt: &str, max_tokens: u32) -> GenerationTask {
        GenerationTask::new(id, prompt).with_max_tokens(max_tokens)
    }

    fn cfg() -> SamplingConfig {
        SamplingConfig::default()
    }

    /// Smallest end-to-end: one slot, two-token prompt, one decode step.
    /// Verifies prefill-then-decode phase machine + KV bookkeeping.
    #[test]
    fn single_slot_prefill_then_decode() {
        let mut b = ContinuousBatcher::new(2);
        b.submit(task("a", "hi there", 4), vec![1, 2], cfg())
            .unwrap();

        // Step 1: prefill token id=1, no sample.
        let plan = b.plan_step();
        assert_eq!(plan.slots.len(), 1);
        assert_eq!(plan.slots[0].input_ids, vec![1]);
        assert_eq!(plan.slots[0].past_seq_len, 0);
        assert!(!plan.slots[0].sample_this_step);
        b.commit_step(&plan.slots, &[StepOutcome::none()], &[]);
        assert_eq!(b.slot(0).unwrap().past_seq_len, 1);
        assert!(matches!(b.slot(0).unwrap().phase, SlotPhase::Prefill));

        // Step 2: prefill token id=2, this consumes the last prompt
        // token so sample_this_step=true. Pretend the engine sampled
        // token 99.
        let plan = b.plan_step();
        assert_eq!(plan.slots[0].input_ids, vec![2]);
        assert_eq!(plan.slots[0].past_seq_len, 1);
        assert!(plan.slots[0].sample_this_step);
        b.commit_step(&plan.slots, &[StepOutcome::token(99)], &[]);
        assert_eq!(b.slot(0).unwrap().past_seq_len, 2);
        assert!(matches!(b.slot(0).unwrap().phase, SlotPhase::Decode));
        assert_eq!(b.slot(0).unwrap().generated, vec![99]);

        // Step 3: decode. Input is the last generated token (99).
        let plan = b.plan_step();
        assert_eq!(plan.slots[0].input_ids, vec![99]);
        assert_eq!(plan.slots[0].past_seq_len, 2);
        assert!(plan.slots[0].sample_this_step);
    }

    /// THE test the task asks for: batch assembly at N=2 with mocks.
    /// Two slots admitted with different prompt lengths; verify both
    /// get planned each step and KV counts advance independently.
    #[test]
    fn batch_assembly_n2_independent_kv_advance() {
        let mut b = ContinuousBatcher::new(4);
        b.submit(task("alpha", "p", 8), vec![10], cfg()).unwrap();
        b.submit(task("bravo", "p p p", 8), vec![20, 21, 22], cfg())
            .unwrap();

        // Step 1: both slots admitted; alpha is in its only prefill
        // step (consumes last token, samples), bravo is in step 1/3.
        let plan = b.plan_step();
        assert_eq!(plan.slots.len(), 2, "both slots should be in batch");
        let a = plan.slots.iter().find(|p| p.slot_idx == 0).unwrap();
        let bp = plan.slots.iter().find(|p| p.slot_idx == 1).unwrap();
        assert_eq!(a.input_ids, vec![10]);
        assert!(a.sample_this_step, "alpha consumes its last prompt token");
        assert_eq!(bp.input_ids, vec![20]);
        assert!(!bp.sample_this_step, "bravo still has prompt tokens left");

        b.commit_step(
            &plan.slots,
            &[
                if plan.slots[0].slot_idx == 0 {
                    StepOutcome::token(100)
                } else {
                    StepOutcome::none()
                },
                if plan.slots[1].slot_idx == 0 {
                    StepOutcome::token(100)
                } else {
                    StepOutcome::none()
                },
            ],
            &[],
        );
        assert_eq!(b.slot(0).unwrap().past_seq_len, 1);
        assert_eq!(b.slot(1).unwrap().past_seq_len, 1);
        assert!(matches!(b.slot(0).unwrap().phase, SlotPhase::Decode));
        assert!(matches!(b.slot(1).unwrap().phase, SlotPhase::Prefill));
        assert_eq!(b.slot(0).unwrap().generated, vec![100]);

        // Step 2: alpha decodes, bravo prefills. Verify per-slot
        // past_seq_len divergence — this is the load-bearing property
        // a real continuous batcher needs.
        let plan = b.plan_step();
        assert_eq!(plan.slots.len(), 2);
        let a = plan.slots.iter().find(|p| p.slot_idx == 0).unwrap();
        let bp = plan.slots.iter().find(|p| p.slot_idx == 1).unwrap();
        assert_eq!(a.past_seq_len, 1, "alpha advanced past prompt+1");
        assert_eq!(bp.past_seq_len, 1, "bravo prefilled 1 of 3");
        assert_eq!(a.input_ids, vec![100]);
        assert_eq!(bp.input_ids, vec![21]);
    }

    /// N=2 with EOS on slot 0 mid-decode: slot transitions to Done, gc
    /// frees it, a queued third request takes its place.
    #[test]
    fn eos_frees_slot_and_promotes_pending() {
        let mut b = ContinuousBatcher::new(2);
        let mut s = cfg();
        s.seed = Some(1);
        b.submit(task("alpha", "p", 4), vec![10], s.clone())
            .unwrap();
        b.submit(task("bravo", "p", 4), vec![20], s.clone())
            .unwrap();
        // Third is queued (no free slot at submission time).
        b.submit(task("gamma", "p", 4), vec![30], s.clone())
            .unwrap();
        assert_eq!(b.active_count(), 2);
        assert_eq!(b.pending_count(), 1);

        // Step 1: both prefill+sample. Sample token 42 (EOS) for alpha.
        let plan = b.plan_step();
        assert_eq!(plan.slots.len(), 2);
        let outcomes: Vec<StepOutcome> = plan
            .slots
            .iter()
            .map(|p| {
                if p.slot_idx == 0 {
                    StepOutcome::token(42)
                } else {
                    StepOutcome::token(50)
                }
            })
            .collect();
        let done = b.commit_step(&plan.slots, &outcomes, &[42]);
        assert_eq!(done, vec![0]);
        assert!(matches!(
            b.slot(0).unwrap().phase,
            SlotPhase::Done { eos: true }
        ));

        // GC: alpha freed; gamma drains in on the next plan_step.
        let freed = b.gc();
        assert_eq!(freed, vec!["alpha"]);
        assert_eq!(b.active_count(), 1);

        let plan = b.plan_step();
        assert_eq!(plan.slots.len(), 2, "gamma now in batch with bravo");
        assert!(plan.slots.iter().any(|p| p.slot_idx == 0));
        assert!(plan.slots.iter().any(|p| p.slot_idx == 1));
        assert!(b.slots[0]
            .as_ref()
            .map(|s| s.task_id == "gamma")
            .unwrap_or(false));
        assert_eq!(b.pending_count(), 0);
    }

    /// Hitting `max_new` (no EOS) terminates the slot with `eos:
    /// false` in `Done`. Bounds-check the loop so a request can't
    /// stream forever.
    #[test]
    fn max_new_terminates_slot() {
        let mut b = ContinuousBatcher::new(1);
        b.submit(task("a", "p", 2), vec![10], cfg()).unwrap();
        // Prefill (also samples first token).
        let plan = b.plan_step();
        b.commit_step(&plan.slots, &[StepOutcome::token(1)], &[]);
        // Decode 1 more (max_new=2 → 2 tokens total).
        let plan = b.plan_step();
        b.commit_step(&plan.slots, &[StepOutcome::token(2)], &[]);
        assert!(matches!(
            b.slot(0).unwrap().phase,
            SlotPhase::Done { eos: false }
        ));
        assert_eq!(b.slot(0).unwrap().generated, vec![1, 2]);
    }

    /// Duplicate submit is rejected; current contract is "fail" rather
    /// than no-op so the API layer can surface the error to the client.
    #[test]
    fn duplicate_submit_errors() {
        let mut b = ContinuousBatcher::new(2);
        b.submit(task("a", "p", 4), vec![10], cfg()).unwrap();
        let res = b.submit(task("a", "p", 4), vec![10], cfg());
        assert!(res.is_err());
    }

    /// Empty plan when nothing is admitted. Engine's main loop should
    /// treat this as "spin and wait for a submission".
    #[test]
    fn empty_plan_when_idle() {
        let mut b = ContinuousBatcher::new(4);
        let plan = b.plan_step();
        assert!(plan.is_empty());
    }

    /// `MAX_BATCH_SLOTS_DEFAULT` is a real cap, not advisory: the 5th
    /// submission goes to `pending` even with everything queued at
    /// once.
    #[test]
    fn max_slots_is_hard_cap() {
        let mut b = ContinuousBatcher::new(MAX_BATCH_SLOTS_DEFAULT);
        for i in 0..MAX_BATCH_SLOTS_DEFAULT + 2 {
            b.submit(task(&format!("t{i}"), "p", 4), vec![i as i64], cfg())
                .unwrap();
        }
        assert_eq!(b.active_count(), MAX_BATCH_SLOTS_DEFAULT);
        assert_eq!(b.pending_count(), 2);
    }

    // ============================================================
    // Per-request sampling state tests (iter 078, blocker 3).
    //
    // These tests prove the "blocker 3" wiring from iter 059:
    // ContinuousBatcher now owns N independent (rng, rng_seeded)
    // pairs keyed by slot_idx, so two concurrent slots produce two
    // independent token streams from the same logits.
    // ============================================================

    /// Build a flat-ish logits vector with a mild peak at one index, so
    /// temperature + RNG meaningfully decide the outcome (vs argmax
    /// dominating). Vocab=32 keeps test output small but still gives
    /// the sampler enough surface area to diverge.
    fn flatish_logits(peak_idx: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; 32];
        // Peak is only +1.5; with temperature=1.0 this leaves ~25% prob
        // on the peak and ~3% on each tail token, so different RNG
        // states will pick different tokens. Argmax would always be
        // `peak_idx`.
        v[peak_idx] = 1.5;
        v
    }

    fn temp_cfg(seed: u64) -> SamplingConfig {
        SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            seed: Some(seed),
            ..SamplingConfig::default()
        }
    }

    /// The headline test: two slots, same prompt, same logits, but
    /// DIFFERENT seeds. With temperature>0, sample_for_plan must
    /// produce different tokens for the two slots — proving the per-slot
    /// RNG state is actually independent.
    #[test]
    fn two_slots_different_seeds_independent_streams() {
        let mut b = ContinuousBatcher::new(2);
        b.submit(task("a", "p", 8), vec![10], temp_cfg(1)).unwrap();
        b.submit(task("b", "p", 8), vec![10], temp_cfg(999_999))
            .unwrap();

        let plan = b.plan_step();
        assert_eq!(plan.slots.len(), 2);
        // Both slots get the EXACT same logits, so any divergence in the
        // sampled tokens has to come from the per-slot RNG.
        let logits = vec![flatish_logits(7), flatish_logits(7)];
        let outcomes = b.sample_for_plan(&plan.slots, &logits);
        assert_eq!(outcomes.len(), 2);
        let t0 = outcomes[0].sampled.expect("slot 0 should sample");
        let t1 = outcomes[1].sampled.expect("slot 1 should sample");
        assert_ne!(
            t0, t1,
            "different seeds should yield different tokens (got {t0} == {t1})"
        );

        // And the RNG state for each slot should have advanced
        // independently — neither slot was touched by the other's
        // sampling call.
        let rng0 = b.sampler_rng(0).unwrap();
        let rng1 = b.sampler_rng(1).unwrap();
        assert_ne!(rng0, rng1, "per-slot RNGs should diverge after sample");
        assert_ne!(rng0, 0, "slot 0 RNG should be seeded");
        assert_ne!(rng1, 0, "slot 1 RNG should be seeded");
    }

    /// Sampling-state independence is also about history (rep-penalty
    /// window). Two slots with the same seed and same logits but
    /// different `generated` histories should diverge as soon as the
    /// rep-penalty kicks in on one but not the other.
    #[test]
    fn two_slots_different_history_independent_streams() {
        // Greedy + strong repetition penalty so the test is
        // deterministic. The rep-penalty divides any token in history
        // by alpha (when its logit is positive), so we need a second
        // candidate close to the peak to win after the peak gets
        // penalized. Logit layout: peak at 5 (value 2.0), runner-up at
        // 8 (value 1.0). With alpha=10, after penalizing token 5 its
        // logit drops to 0.2, so 8 (1.0) wins.
        let c = SamplingConfig {
            temperature: 0.0,
            repetition_penalty: 10.0,
            repetition_window: 0,
            seed: Some(7),
            ..SamplingConfig::default()
        };

        let mut b = ContinuousBatcher::new(2);
        b.submit(task("a", "p", 8), vec![10], c.clone()).unwrap();
        b.submit(task("b", "p", 8), vec![10], c.clone()).unwrap();

        let two_peak_logits = || {
            let mut v = vec![0.0_f32; 32];
            v[5] = 2.0; // primary peak
            v[8] = 1.0; // runner-up — wins when 5 is penalized
            v
        };

        // Step 1: prefill+sample on the one-token prompt. Both slots
        // empty-history → argmax = 5 (the primary peak).
        let plan1 = b.plan_step();
        let logits1 = vec![two_peak_logits(), two_peak_logits()];
        let out1 = b.sample_for_plan(&plan1.slots, &logits1);
        assert_eq!(out1[0].sampled, Some(5));
        assert_eq!(out1[1].sampled, Some(5));
        // Force-divergence: commit DIFFERENT tokens. Slot 0 gets the
        // real argmax (5); slot 1 gets a synthetic alternative (8).
        // This sets up histories that point at DIFFERENT indices, so
        // the rep penalty will hit different positions next step.
        b.commit_step(
            &plan1.slots,
            &[StepOutcome::token(5), StepOutcome::token(8)],
            &[],
        );
        assert_eq!(b.slot(0).unwrap().generated, vec![5]);
        assert_eq!(b.slot(1).unwrap().generated, vec![8]);

        // Step 2: both decode. Same logits AGAIN — but slot 0's
        // rep-penalty knocks index 5 down (logit 2.0 → 0.2) so the
        // runner-up at 8 (logit 1.0) wins. Slot 1's history has 8
        // instead, so 8 → 0.1 and 5 (unpenalized 2.0) wins. The two
        // slots SWAP their preferred token.
        let plan2 = b.plan_step();
        let logits2 = vec![two_peak_logits(), two_peak_logits()];
        let out2 = b.sample_for_plan(&plan2.slots, &logits2);
        let t0 = out2[0].sampled.expect("slot 0 sampled");
        let t1 = out2[1].sampled.expect("slot 1 sampled");
        assert_eq!(t0, 8, "slot 0 penalized 5 → runner-up 8 wins");
        assert_eq!(t1, 5, "slot 1 penalized 8 → primary peak 5 wins");
        assert_ne!(t0, t1, "per-slot histories must diverge token streams");
    }

    /// `sample_this_step == false` (intermediate prefill steps) must NOT
    /// advance the slot's RNG. This matches the existing single-request
    /// engine path which only calls the sampler on the last prefill
    /// step + decode — deterministic seeds reproduce bit-for-bit only
    /// if the batched path obeys the same invariant.
    #[test]
    fn unsampled_step_does_not_advance_rng() {
        let mut b = ContinuousBatcher::new(1);
        // 3-token prompt → 2 unsampled prefill steps then 1 sampled.
        b.submit(task("a", "abc", 4), vec![10, 11, 12], temp_cfg(1))
            .unwrap();

        // Step 1: prefill token 10. sample_this_step=false.
        let plan = b.plan_step();
        assert!(!plan.slots[0].sample_this_step);
        let out = b.sample_for_plan(&plan.slots, &[flatish_logits(7)]);
        assert!(out[0].sampled.is_none(), "intermediate prefill yields none");
        // RNG should NOT be seeded yet — no sample happened.
        assert_eq!(b.sampler_rng(0), Some(0), "rng untouched on unsampled step");
        // Commit to advance phase.
        b.commit_step(&plan.slots, &out, &[]);

        // Step 2: prefill token 11. Still sample_this_step=false.
        let plan = b.plan_step();
        assert!(!plan.slots[0].sample_this_step);
        b.sample_for_plan(&plan.slots, &[flatish_logits(7)]);
        assert_eq!(b.sampler_rng(0), Some(0), "rng still untouched");
        b.commit_step(&plan.slots, &[StepOutcome::none()], &[]);

        // Step 3: prefill token 12, LAST prompt token → sample_this_step=true.
        let plan = b.plan_step();
        assert!(plan.slots[0].sample_this_step);
        let _ = b.sample_for_plan(&plan.slots, &[flatish_logits(7)]);
        let rng_now = b.sampler_rng(0).unwrap();
        assert_ne!(rng_now, 0, "rng must be seeded + advanced after a sample");
    }

    /// GC frees the sampler in lockstep with the slot. Re-admitting a
    /// task into the same slot index gets a FRESH sampler (not the
    /// stale one from the previous occupant). The parallel-index
    /// invariant (samplers[i].is_some() iff slots[i].is_some()) is the
    /// load-bearing thing here.
    #[test]
    fn gc_clears_sampler_state() {
        let mut b = ContinuousBatcher::new(1);
        b.submit(task("a", "p", 1), vec![10], temp_cfg(42)).unwrap();
        let plan = b.plan_step();
        b.sample_for_plan(&plan.slots, &[flatish_logits(3)]);
        let stale_rng = b.sampler_rng(0).unwrap();
        assert_ne!(stale_rng, 0);

        // Terminate slot 0 by max_new=1 → Done(eos: false) → gc.
        b.commit_step(&plan.slots, &[StepOutcome::token(99)], &[]);
        assert!(matches!(
            b.slot(0).unwrap().phase,
            SlotPhase::Done { eos: false }
        ));
        b.gc();
        assert!(b.slot(0).is_none(), "slot 0 freed");
        assert!(b.sampler_rng(0).is_none(), "sampler 0 freed in lockstep");

        // Re-admit. Slot 0 is reused; the sampler must NOT carry over
        // the previous occupant's RNG state.
        b.submit(task("b", "p", 1), vec![20], temp_cfg(42)).unwrap();
        assert_eq!(
            b.sampler_rng(0),
            Some(0),
            "fresh slot gets fresh (unseeded) sampler, not stale RNG state"
        );
    }

    /// End-to-end round trip on N=2: plan → sample (per-slot RNGs
    /// independent) → commit → plan again → sample again, verifying
    /// per-slot state survives across steps. This is the "2 slots
    /// produce 2 independent token streams" test the task asks for, in
    /// integration form rather than the unit-level seed-divergence test.
    #[test]
    fn n2_round_trip_independent_token_streams() {
        let mut b = ContinuousBatcher::new(2);
        b.submit(task("a", "p", 4), vec![10], temp_cfg(1)).unwrap();
        b.submit(task("b", "p", 4), vec![20], temp_cfg(2)).unwrap();

        let mut stream_a: Vec<i64> = Vec::new();
        let mut stream_b: Vec<i64> = Vec::new();

        for _ in 0..3 {
            let plan = b.plan_step();
            assert_eq!(plan.slots.len(), 2);
            // Same logits to both — the only source of divergence is
            // the per-slot sampler state.
            let logits: Vec<Vec<f32>> = plan.slots.iter().map(|_| flatish_logits(13)).collect();
            let outcomes = b.sample_for_plan(&plan.slots, &logits);
            // Push the sampled token into whichever stream it belongs to.
            for (p, o) in plan.slots.iter().zip(outcomes.iter()) {
                if let Some(t) = o.sampled {
                    if p.slot_idx == 0 {
                        stream_a.push(t);
                    } else {
                        stream_b.push(t);
                    }
                }
            }
            b.commit_step(&plan.slots, &outcomes, &[]);
        }

        assert!(!stream_a.is_empty());
        assert!(!stream_b.is_empty());
        assert_ne!(
            stream_a, stream_b,
            "independent per-slot samplers must produce different streams"
        );
    }
}
