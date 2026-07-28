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
    pub(crate) mask_in: String,
    pub(crate) pos_in: String,
    pub(crate) mask_dtype: ShimDType,
    pub(crate) layers: Vec<PackedLayer>,
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
            mask_in,
            pos_in,
            mask_dtype,
            layers,
            mask_bytes: Vec::new(),
            ids_bytes: Vec::new(),
            pos_bytes: Vec::new(),
        }
    }

    pub(crate) fn free_slot(&self) -> Option<usize> {
        self.slots.iter().position(|s| s.is_none())
    }

    pub(crate) fn occupied(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Place a task into `slot`, clearing that slot's KV region only.
    pub(crate) fn admit(&mut self, slot: usize, task: GenerationTask, prompt_ids: Vec<i64>) {
        self.kv.reset_slot(slot);
        self.slots[slot] = Some(PackedSlot {
            task,
            prompt_ids,
            prompt_fed: 0,
            generated: Vec::new(),
            last_text: String::new(),
            last_token: 0,
            started: Instant::now(),
            t_prefill: Duration::ZERO,
        });
    }

    pub(crate) fn retire(&mut self, slot: usize) -> Option<PackedSlot> {
        let taken = self.slots[slot].take();
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
        self.runtime
            .set_input(&self.ids_in, ShimDType::I64, &[1, s], &self.ids_bytes)
            .map_err(crate::runtime::map_ov_err)?;
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
            if let PackedStepKind::Prefill { slot: ps, .. } = &kind {
                if *ps == slot {
                    if let Some(t) = self.slots[slot].as_mut() {
                        t.prompt_fed += n;
                    }
                }
            }
        }
        Ok(Some((odt, oshape, obytes, kind)))
    }
}
