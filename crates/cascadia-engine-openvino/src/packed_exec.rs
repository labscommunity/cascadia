//! Engine-side execution for packed multi-slot decode (see [`crate::packed`]).
//!
//! Holds the compiled packed variant, the per-slot KV, and the slot table, and
//! turns one `step()` into exactly one inference that either prefills a chunk of
//! one admitted request or decodes one token for every ready request.
//!
//! Scheduling is prefill-first: a newly admitted request consumes its prompt in
//! `packed_seq`-wide chunks before joining the decode batch, so a long prompt
//! cannot stall other slots for more than one inference at a time.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use cascadia_engine::{EngineError, EngineResult};
use cascadia_ov_genai_shim::{DType as ShimDType, Runtime as OvRuntime};
use cascadia_types::GenerationTask;

use crate::packed::{PackedKv, PackedPlan};

/// Primary input of a packed inference: prompt/decode token ids on the embed
/// stage, upstream hidden rows on relay + head stages.
pub(crate) enum PackedPrimary<'a> {
    Ids(&'a [i64]),
    /// `rows * hidden_size` f32 in row-major order, converted to the IR's f16.
    Hidden(&'a [f32], usize),
}

/// One request occupying a packed slot.
pub(crate) struct PackedSlot {
    pub(crate) task: GenerationTask,
    pub(crate) prompt_ids: Vec<i64>,
    /// Prompt tokens already consumed; `== prompt_ids.len()` once prefilled.
    pub(crate) prompt_fed: usize,
    pub(crate) generated: Vec<i32>,
    pub(crate) last_text: String,
    pub(crate) last_token: i32,
    pub(crate) started: Instant,
    pub(crate) t_prefill: Duration,
}

impl PackedSlot {
    fn prefilled(&self) -> bool {
        self.prompt_fed >= self.prompt_ids.len()
    }
}

/// Per-layer KV port wiring for the packed variant (mirrors `StaticKvLayer`,
/// duplicated here so the packed path owns its own resolved ports).
pub(crate) struct PackedLayer {
    pub(crate) key_in: String,
    pub(crate) val_in: String,
    pub(crate) key_out: usize,
    pub(crate) val_out: usize,
}

/// What one packed inference did, so the caller can sample the right rows.
pub(crate) enum PackedStepKind {
    /// Prefill chunk for `slot`; `finished_prompt` marks the chunk that
    /// consumed the last prompt token, whose final real row carries the first
    /// generated token's logits at `last_row`.
    Prefill {
        slot: usize,
        last_row: usize,
        finished_prompt: bool,
    },
    /// One decode row per listed `(row, slot)`.
    Decode { rows: Vec<(usize, usize)> },
}

pub(crate) struct PackedState {
    pub(crate) runtime: OvRuntime,
    pub(crate) kv: PackedKv,
    pub(crate) slots: Vec<Option<PackedSlot>>,
    pub(crate) ids_in: String,
    pub(crate) hidden_in: String,
    pub(crate) mask_in: String,
    pub(crate) pos_in: String,
    pub(crate) mask_dtype: ShimDType,
    pub(crate) layers: Vec<PackedLayer>,
    /// Token ids whose K/V now sit in the shared prefix region. Only the
    /// driver stage matches against these; relay stages are told the reuse
    /// length on the wire.
    pub(crate) prefix_ids: Vec<i64>,
    /// Slot currently populating the shared prefix, if any.
    populating: Option<usize>,
    last_plan_rows: Vec<Option<(usize, i64, usize)>>,
    mask_bytes: Vec<u8>,
    ids_bytes: Vec<u8>,
    pos_bytes: Vec<u8>,
}

impl PackedState {
    pub(crate) fn new(
        runtime: OvRuntime,
        kv: PackedKv,
        layers: Vec<PackedLayer>,
        ids_in: String,
        hidden_in: String,
        mask_in: String,
        pos_in: String,
        mask_dtype: ShimDType,
    ) -> Self {
        let slots = kv.slots;
        Self {
            runtime,
            kv,
            slots: (0..slots).map(|_| None).collect(),
            ids_in,
            hidden_in,
            mask_in,
            pos_in,
            mask_dtype,
            layers,
            prefix_ids: Vec::new(),
            populating: None,
            last_plan_rows: Vec::new(),
            mask_bytes: Vec::new(),
            ids_bytes: Vec::new(),
            pos_bytes: Vec::new(),
        }
    }

    pub(crate) fn free_slot(&self) -> Option<usize> {
        self.slots.iter().position(|s| s.is_none())
    }

    /// Per-row `(slot, position)` of the most recent inference — what a
    /// pipeline stage ships downstream so every stage masks identically.
    pub(crate) fn last_plan_rows(&self) -> Vec<Option<(usize, i64, usize)>> {
        self.last_plan_rows.clone()
    }

    pub(crate) fn occupied(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Longest prefix of `prompt_ids` whose K/V is already cached in the shared
    /// region. Reusing it means those tokens are never prefilled again.
    pub(crate) fn prefix_reuse(&self, prompt_ids: &[i64]) -> usize {
        let cap = self.kv.prefix_valid.min(self.prefix_ids.len());
        let mut n = 0;
        while n < cap && n < prompt_ids.len() && self.prefix_ids[n] == prompt_ids[n] {
            n += 1;
        }
        // Never reuse the WHOLE prompt: the slot needs at least one token to
        // run through the model to produce logits for its first output.
        n.min(prompt_ids.len().saturating_sub(1))
    }

    /// Place a task into `slot`, reusing `shared` cached prefix tokens and
    /// clearing only that slot's own KV region.
    pub(crate) fn admit(&mut self, slot: usize, task: GenerationTask, prompt_ids: Vec<i64>) {
        let shared = self.prefix_reuse(&prompt_ids);
        self.kv.reset_slot_reusing(slot, shared);
        // Nothing cached yet? This request populates the shared prefix.
        if self.kv.prefix_valid == 0 && self.populating.is_none() && self.kv.prefix_capacity > 0 {
            self.populating = Some(slot);
            self.prefix_ids = prompt_ids
                .iter()
                .copied()
                .take(self.kv.prefix_capacity)
                .collect();
        }
        self.slots[slot] = Some(PackedSlot {
            task,
            prompt_ids,
            prompt_fed: shared,
            generated: Vec::new(),
            last_text: String::new(),
            last_token: 0,
            started: Instant::now(),
            t_prefill: Duration::ZERO,
        });
    }

    pub(crate) fn retire(&mut self, slot: usize) -> Option<PackedSlot> {
        let taken = self.slots[slot].take();
        // A slot that was going to populate the shared prefix but left first
        // must release the claim, or nothing would ever populate it.
        if self.populating == Some(slot) && self.kv.prefix_valid == 0 {
            self.populating = None;
            self.prefix_ids.clear();
        }
        self.kv.reset_slot(slot);
        taken
    }

    /// Slot needing prefill, lowest index first (admission order).
    fn next_prefill_slot(&self) -> Option<usize> {
        self.slots
            .iter()
            .position(|s| s.as_ref().is_some_and(|t| !t.prefilled()))
    }

    /// Run exactly one packed inference. Returns the primary output
    /// (dtype, shape, bytes) plus what the step did.
    pub(crate) fn step(
        &mut self,
    ) -> EngineResult<Option<(ShimDType, Vec<usize>, Vec<u8>, PackedStepKind)>> {
        let s = self.kv.packed_seq;
        let (plan, ids, positions, kind) = if let Some(slot) = self.next_prefill_slot() {
            // ---- prefill chunk for one slot ----
            let t = self.slots[slot].as_ref().unwrap();
            let start = t.prompt_fed;
            let take = (t.prompt_ids.len() - start).min(s);
            let ids: Vec<i64> = t.prompt_ids[start..start + take].to_vec();
            let base = self.kv.position(slot);
            let positions: Vec<i64> = (0..take).map(|i| (base + i) as i64).collect();
            let finished = start + take >= t.prompt_ids.len();
            (
                PackedPlan::chunk(s, slot, take),
                ids,
                positions,
                PackedStepKind::Prefill {
                    slot,
                    last_row: take - 1,
                    finished_prompt: finished,
                },
            )
        } else {
            // ---- decode one token for every ready slot ----
            let ready: Vec<usize> = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(i, t)| t.as_ref().map(|_| i))
                .collect();
            if ready.is_empty() {
                return Ok(None);
            }
            let ids: Vec<i64> = ready
                .iter()
                .map(|&i| self.slots[i].as_ref().unwrap().last_token as i64)
                .collect();
            let positions: Vec<i64> = ready.iter().map(|&i| self.kv.position(i) as i64).collect();
            let rows: Vec<(usize, usize)> = ready.iter().copied().enumerate().collect();
            (
                PackedPlan::decode(s, &ready),
                ids,
                positions,
                PackedStepKind::Decode { rows },
            )
        };
        if plan.is_empty() {
            return Ok(None);
        }
        let reuse: Vec<usize> = plan
            .rows
            .iter()
            .map(|r| r.map(|pr| self.kv.shared_use(pr.slot)).unwrap_or(0))
            .collect();
        let (odt, oshape, obytes) =
            self.run_plan(&plan, PackedPrimary::Ids(&ids), &positions, &reuse)?;
        if let PackedStepKind::Prefill { slot, .. } = &kind {
            if let Some(t) = self.slots[*slot].as_mut() {
                t.prompt_fed += plan.active_rows().count();
            }
        }
        Ok(Some((odt, oshape, obytes, kind)))
    }

    /// Run one packed inference for a caller-supplied plan. Stage 0 builds the
    /// plan from its slot table; relay and head stages decode it off the wire,
    /// which is what keeps every stage's per-slot rings in lockstep.
    ///
    /// A row whose absolute position equals its reuse length resets its slot
    /// first (position 0 when nothing is reused) — the same in-band "new
    /// sequence" signal the single-task static path already uses, so a
    /// downstream stage needs no separate admission message.
    pub(crate) fn run_plan(
        &mut self,
        plan: &PackedPlan,
        primary: PackedPrimary<'_>,
        positions: &[i64],
        reuse: &[usize],
    ) -> EngineResult<(ShimDType, Vec<usize>, Vec<u8>)> {
        let s = self.kv.packed_seq;
        // A row whose absolute position equals its reuse length is the FIRST
        // row of that slot's sequence — the in-band new-sequence signal, which
        // generalises the old "position == 0" rule to prefix reuse.
        for (r, pr) in plan.active_rows() {
            let pos = positions.get(r).copied().unwrap_or(0) as usize;
            let shared = reuse.get(r).copied().unwrap_or(0);
            if pos == shared {
                self.kv.reset_slot_reusing(pr.slot, shared);
            }
        }
        self.last_plan_rows = (0..s)
            .map(|r| {
                plan.rows.get(r).copied().flatten().map(|pr| {
                    (
                        pr.slot,
                        positions.get(r).copied().unwrap_or(0),
                        self.kv.shared_use(pr.slot),
                    )
                })
            })
            .collect();
        let ids: Vec<i64> = match primary {
            PackedPrimary::Ids(v) => v.to_vec(),
            PackedPrimary::Hidden(_, _) => Vec::new(),
        };

        // ---- feed ----
        self.kv
            .fill_mask(&plan, &mut self.mask_bytes, self.mask_dtype);
        // Idle rows still need a defined token/position; row 0's values are
        // safe filler because an idle row attends only to itself and its
        // output is never read.
        self.ids_bytes.clear();
        self.pos_bytes.clear();
        for r in 0..s {
            let (id, pos) = match plan.rows[r] {
                Some(_) => (
                    ids.get(r).copied().unwrap_or(0),
                    positions.get(r).copied().unwrap_or(0),
                ),
                None => (0i64, 0i64),
            };
            self.ids_bytes.extend_from_slice(&id.to_le_bytes());
            self.pos_bytes.extend_from_slice(&pos.to_le_bytes());
        }
        match primary {
            PackedPrimary::Ids(_) => {
                self.runtime
                    .set_input(&self.ids_in, ShimDType::I64, &[1, s], &self.ids_bytes)
                    .map_err(crate::runtime::map_ov_err)?;
            }
            PackedPrimary::Hidden(rows, hidden_size) => {
                // Pad idle rows with zeros so the tensor is always [1, S, H];
                // their outputs are masked to self and never read.
                let mut buf = vec![0f32; s * hidden_size];
                let n = (rows.len() / hidden_size).min(s);
                buf[..n * hidden_size].copy_from_slice(&rows[..n * hidden_size]);
                let bytes = crate::runtime::f32_to_f16_bytes(&buf);
                self.runtime
                    .set_input(
                        &self.hidden_in,
                        ShimDType::F16,
                        &[1, s, hidden_size],
                        &bytes,
                    )
                    .map_err(crate::runtime::map_ov_err)?;
            }
        }
        self.runtime
            .set_input(
                &self.mask_in,
                self.mask_dtype,
                &[1, 1, s, self.kv.context],
                &self.mask_bytes,
            )
            .map_err(crate::runtime::map_ov_err)?;
        self.runtime
            .set_input(&self.pos_in, ShimDType::I64, &[1, s], &self.pos_bytes)
            .map_err(crate::runtime::map_ov_err)?;
        let kv_shape = [1, self.kv.kv_heads(), self.kv.past_len, self.kv.head_dim()];
        for (li, layer) in self.layers.iter().enumerate() {
            self.runtime
                .set_input(
                    &layer.key_in,
                    ShimDType::F16,
                    &kv_shape,
                    self.kv.key_bytes(li),
                )
                .map_err(crate::runtime::map_ov_err)?;
            self.runtime
                .set_input(
                    &layer.val_in,
                    ShimDType::F16,
                    &kv_shape,
                    self.kv.val_bytes(li),
                )
                .map_err(crate::runtime::map_ov_err)?;
        }

        self.runtime.infer().map_err(crate::runtime::map_ov_err)?;
        let (odt, oshape, obytes) = self.runtime.output(0).map_err(crate::runtime::map_ov_err)?;

        // ---- scatter present rows back into their owning slot regions ----
        let expect = self.kv.present_layer_bytes();
        let want_shape = [
            1usize,
            self.kv.kv_heads(),
            self.kv.context,
            self.kv.head_dim(),
        ];
        // occupancy per slot BEFORE this step drives append-vs-slide
        let mut at: HashMap<usize, usize> = HashMap::new();
        for (_r, pr) in plan.active_rows() {
            at.entry(pr.slot).or_insert_with(|| self.kv.valid(pr.slot));
        }
        for li in 0..self.layers.len() {
            let (ko, vo) = (self.layers[li].key_out, self.layers[li].val_out);
            let (_, kshape, kpres) = self
                .runtime
                .output(ko)
                .map_err(crate::runtime::map_ov_err)?;
            let (_, vshape, vpres) = self
                .runtime
                .output(vo)
                .map_err(crate::runtime::map_ov_err)?;
            if kpres.len() != expect
                || vpres.len() != expect
                || kshape != want_shape
                || vshape != want_shape
            {
                return Err(EngineError::Backend(format!(
                    "packed present.{li} mismatch: key shape={kshape:?} len={} val shape={vshape:?} \
                     len={}; expected shape {want_shape:?} ({expect} bytes f16)",
                    kpres.len(),
                    vpres.len(),
                )));
            }
            let mut per_slot = at.clone();
            for (r, pr) in plan.active_rows() {
                let slot_at = per_slot.get_mut(&pr.slot).expect("seeded above");
                let capped = (*slot_at).min(self.kv.region);
                self.kv.absorb_row(li, false, &kpres, r, pr.slot, capped);
                self.kv.absorb_row(li, true, &vpres, r, pr.slot, capped);
                *slot_at += 1;
            }
        }
        // advance each slot by the rows it consumed
        let mut consumed: HashMap<usize, usize> = HashMap::new();
        for (_r, pr) in plan.active_rows() {
            *consumed.entry(pr.slot).or_insert(0) += 1;
        }
        for (slot, n) in consumed {
            self.kv.advance(slot, n);
        }
        // Once the populating request has enough tokens in its own region, copy
        // them into the shared prefix so later requests can skip re-prefilling
        // them. Deterministic from `pos` alone, so every pipeline stage does it
        // at the same step without extra signalling.
        if let Some(src) = self.populating {
            let want = self.kv.prefix_capacity.min(self.prefix_ids.len().max(1));
            if self.kv.prefix_valid == 0 && self.kv.valid(src) >= want && want > 0 {
                self.kv.populate_prefix_from(src, want);
                self.prefix_ids.truncate(self.kv.prefix_valid);
                self.populating = None;
            }
        }
        Ok((odt, oshape, obytes))
    }
}
