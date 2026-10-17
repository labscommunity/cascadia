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

use crate::packed::{image_safe_take, prompt_image_ids, PackedKv, PackedPlan};
use crate::qwen35_vision::rope::MropePositions;

/// Qwen VL packed extras fed alongside a plan. `mrope` is the per-row M-RoPE
/// `position_ids` as `[3, S]` row-major (`mrope[r*S + col]`), empty on a text /
/// non-Qwen tree. `ds_mask` / `ds_embeds` are Qwen3-VL stage 0's DeepStack
/// inputs (`visual_pos_masks [S]`, `deepstack_visual_embeds [layers, S,
/// hidden]`); empty on every other stage, since only stage 0 holds the
/// DeepStack layers. The relay stages carry only `mrope` on the wire.
#[derive(Default, Clone)]
pub(crate) struct PackedExtras {
    pub(crate) mrope: Vec<i64>,
    pub(crate) ds_mask: Vec<i64>,
    pub(crate) ds_embeds: Vec<f32>,
}

/// Primary input of a packed inference: prompt/decode token ids on the embed
/// stage, upstream hidden rows on relay + head stages.
pub(crate) enum PackedPrimary<'a> {
    Ids(&'a [i64]),
    /// `rows * hidden_size` f32 in row-major order, converted to the IR's f16.
    Hidden(&'a [f32], usize),
    /// A vision tree's stage 0: `rows * hidden_size` embedding rows, fed as
    /// f32 into the `inputs_embeds` port (held in `ids_in`).
    Embeds(&'a [f32], usize),
}

/// Rank 0 of a vision tree embeds ids itself: the text embedding table,
/// returning `ids.len() * hidden` row-major f32.
pub(crate) type EmbedFn<'a> = &'a mut dyn FnMut(&[i64]) -> EngineResult<Vec<f32>>;

/// One request occupying a packed slot.
pub(crate) struct PackedSlot {
    pub(crate) task: GenerationTask,
    pub(crate) prompt_ids: Vec<i64>,
    /// Vision trees: the prompt's embedding rows (`prompt_ids.len() *
    /// hidden`, image rows spliced in); empty on id-fed stages.
    pub(crate) prompt_rows: Vec<f32>,
    /// Image id per prompt token when the tree masks image blocks (Gemma 4),
    /// else empty. Prefill chunks never split an image.
    pub(crate) prompt_images: Vec<Option<u32>>,
    /// Qwen VL: the prompt's M-RoPE positions (`None` on a text turn, where
    /// every row is the plain token index). A prefill chunk slices these; a
    /// decode step continues from `rope_delta`.
    pub(crate) mrope: Option<MropePositions>,
    /// Qwen3-VL: this slot's DeepStack rows flattened to `[layers, total,
    /// hidden]` (prompt order); empty otherwise. `ds_total` is the slot's image
    /// tokens, so a prefill chunk slices `[cursor, cursor + k)` for its own.
    pub(crate) ds_global: Vec<f32>,
    pub(crate) ds_total: usize,
    /// Prompt tokens already consumed; `== prompt_ids.len()` once prefilled.
    pub(crate) prompt_fed: usize,
    pub(crate) generated: Vec<i32>,
    /// Bytes already handed to the client (see `ActiveTask::emitted`).
    pub(crate) emitted: Vec<u8>,
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

/// One attention-mask input of the packed IR. Most families have one,
/// `attn_mask_4d` (full attention); Gemma 4 adds `attn_mask_sliding_4d`, fed
/// the same mask with its sliding window applied.
pub(crate) struct PackedMask {
    pub(crate) port: String,
    pub(crate) dtype: ShimDType,
    /// `Some(window)` for a sliding-window mask.
    pub(crate) window: Option<usize>,
    pub(crate) bytes: Vec<u8>,
}

impl PackedMask {
    pub(crate) fn new(port: String, dtype: ShimDType, window: Option<usize>) -> Self {
        Self {
            port,
            dtype,
            window,
            bytes: Vec::new(),
        }
    }
}

/// One row of a packed plan as it travels between stages: slot, absolute
/// position, shared-prefix reuse length.
pub(crate) type WireRow = (usize, i64, usize);
/// A [`WireRow`], or `None` for an idle row.
pub(crate) type WireRowOpt = Option<WireRow>;

/// One planned packed inference (see [`PackedState::plan_step`]).
pub(crate) struct Planned {
    pub(crate) plan: PackedPlan,
    pub(crate) ids: Vec<i64>,
    pub(crate) positions: Vec<i64>,
    pub(crate) kind: PackedStepKind,
    /// A prefill chunk's embedding rows on a vision stage 0; empty otherwise.
    pub(crate) rows: Vec<f32>,
    /// Qwen VL M-RoPE positions + Qwen3-VL DeepStack feed for this inference.
    pub(crate) extras: PackedExtras,
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
    /// `input_ids`, or `inputs_embeds` when `embed_hidden > 0`.
    pub(crate) ids_in: String,
    pub(crate) hidden_in: String,
    /// Row width of a vision tree's stage 0 (fed embedding rows); 0 when the
    /// stage takes ids or hidden states.
    pub(crate) embed_hidden: usize,
    /// Every mask port the packed IR declares (one, or Gemma 4's two).
    pub(crate) masks: Vec<PackedMask>,
    pub(crate) pos_in: String,
    /// `token_type_ids`, when the packed IR still reads it (something other
    /// than the replaced mask producers consumed it); fed 1 on image rows.
    pub(crate) types_in: Option<String>,
    /// The image soft-token id on a tree whose masks open image blocks
    /// (Gemma 4, rank 0): admission derives each prompt's image ids from it.
    pub(crate) image_token: Option<i64>,
    /// Qwen VL: this stage feeds `[3, 1, S]` M-RoPE `position_ids`.
    pub(crate) mrope: bool,
    /// Qwen3-VL stage 0: the DeepStack ports and what feeds them.
    pub(crate) visual_mask_in: Option<String>,
    pub(crate) deepstack_in: Option<String>,
    pub(crate) deepstack_layers: usize,
    /// Row width of the DeepStack embeds (the model hidden size).
    pub(crate) deepstack_hidden: usize,
    /// The image soft-token id for the DeepStack mask (Qwen3-VL).
    pub(crate) deepstack_image_pad: i64,
    pub(crate) layers: Vec<PackedLayer>,
    /// Token ids whose K/V now sit in the shared prefix region. Only the
    /// driver stage matches against these; relay stages are told the reuse
    /// length on the wire.
    pub(crate) prefix_ids: Vec<i64>,
    /// Slot currently populating the shared prefix, if any.
    populating: Option<usize>,
    /// Tasks retired by planning, awaiting their error chunk.
    failed: Vec<(GenerationTask, String)>,
    last_plan_rows: Vec<WireRowOpt>,
    /// Per-row image ids of the most recent inference (see `last_plan_rows`).
    last_plan_images: Vec<Option<u32>>,
    /// Per-row M-RoPE positions of the most recent inference (`[3, S]`
    /// row-major), shipped to the relay stages on the wire.
    last_plan_mrope: Vec<i64>,
    ids_bytes: Vec<u8>,
    types_bytes: Vec<u8>,
    pos_bytes: Vec<u8>,
}

impl PackedState {
    /// `types_in` and `image_token` start unset; a Gemma 4 tree sets them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        runtime: OvRuntime,
        kv: PackedKv,
        layers: Vec<PackedLayer>,
        ids_in: String,
        hidden_in: String,
        embed_hidden: usize,
        masks: Vec<PackedMask>,
        pos_in: String,
    ) -> Self {
        let slots = kv.slots;
        Self {
            runtime,
            kv,
            slots: (0..slots).map(|_| None).collect(),
            ids_in,
            hidden_in,
            embed_hidden,
            masks,
            pos_in,
            types_in: None,
            image_token: None,
            mrope: false,
            visual_mask_in: None,
            deepstack_in: None,
            deepstack_layers: 0,
            deepstack_hidden: 0,
            deepstack_image_pad: 0,
            layers,
            prefix_ids: Vec::new(),
            populating: None,
            failed: Vec::new(),
            last_plan_rows: Vec::new(),
            last_plan_images: Vec::new(),
            last_plan_mrope: Vec::new(),
            ids_bytes: Vec::new(),
            types_bytes: Vec::new(),
            pos_bytes: Vec::new(),
        }
    }

    pub(crate) fn free_slot(&self) -> Option<usize> {
        self.slots.iter().position(|s| s.is_none())
    }

    /// Per-row `(slot, position)` of the most recent inference — what a
    /// pipeline stage ships downstream so every stage masks identically.
    pub(crate) fn last_plan_rows(&self) -> Vec<WireRowOpt> {
        self.last_plan_rows.clone()
    }

    /// Per-row image ids of the most recent inference, for the wire of a
    /// tree whose masks open image blocks.
    pub(crate) fn last_plan_images(&self) -> Vec<Option<u32>> {
        self.last_plan_images.clone()
    }

    /// Per-row M-RoPE positions (`[3, S]` row-major) of the most recent
    /// inference, shipped to the relay stages of a Qwen VL tree.
    pub(crate) fn last_plan_mrope(&self) -> Vec<i64> {
        self.last_plan_mrope.clone()
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
    ///
    /// `shareable` caps how much of the prompt may be matched against, or
    /// populate, the shared prefix: a vision prompt passes the index of its
    /// first image token, since every image of one size has the same ids and
    /// matching past it would reuse another image's K/V.
    ///
    /// Returns how many prompt tokens come from the shared prefix.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn admit(
        &mut self,
        slot: usize,
        task: GenerationTask,
        prompt_ids: Vec<i64>,
        prompt_rows: Vec<f32>,
        shareable: usize,
        mrope: Option<MropePositions>,
        ds_global: Vec<f32>,
        ds_total: usize,
    ) -> usize {
        let shared = self
            .prefix_reuse(&prompt_ids[..shareable.min(prompt_ids.len())])
            .min(prompt_ids.len().saturating_sub(1));
        self.kv.reset_slot_reusing(slot, shared);
        // Nothing cached yet? This request populates the shared prefix.
        // A prompt that opens with an image (`shareable == 0`) has nothing
        // shareable; claiming would pin one image row as the prefix for good.
        if self.kv.prefix_valid == 0
            && self.populating.is_none()
            && self.kv.prefix_capacity > 0
            && shareable > 0
        {
            self.populating = Some(slot);
            self.prefix_ids = prompt_ids
                .iter()
                .copied()
                .take(self.kv.prefix_capacity.min(shareable))
                .collect();
        }
        let prompt_images = match self.image_token {
            Some(image) => prompt_image_ids(&prompt_ids, image),
            None => Vec::new(),
        };
        self.slots[slot] = Some(PackedSlot {
            task,
            prompt_ids,
            prompt_rows,
            prompt_images,
            mrope,
            ds_global,
            ds_total,
            prompt_fed: shared,
            generated: Vec::new(),
            emitted: Vec::new(),
            last_token: 0,
            started: Instant::now(),
            t_prefill: Duration::ZERO,
        });
        shared
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
    ///
    /// A vision tree's stage 0 (`embed_hidden > 0`) feeds embedding rows: a
    /// prefill chunk slices the slot's spliced prompt rows, a decode step
    /// embeds the sampled tokens through `embed`.
    pub(crate) fn step(
        &mut self,
        embed: Option<EmbedFn<'_>>,
    ) -> EngineResult<Option<(ShimDType, Vec<usize>, Vec<u8>, PackedStepKind)>> {
        let h = self.embed_hidden;
        let Some(Planned {
            plan,
            ids,
            positions,
            kind,
            mut rows,
            extras,
        }) = self.plan_step()
        else {
            return Ok(None);
        };
        if h > 0 && matches!(kind, PackedStepKind::Decode { .. }) {
            let embed = embed.ok_or_else(|| {
                EngineError::Backend("packed vision stage stepped without its embedder".into())
            })?;
            rows = embed(&ids)?;
            if rows.len() != ids.len() * h {
                return Err(EngineError::Backend(format!(
                    "embedder returned {} floats for {} tokens of {h}",
                    rows.len(),
                    ids.len()
                )));
            }
        }
        let reuse: Vec<usize> = plan
            .rows
            .iter()
            .map(|r| r.map(|pr| self.kv.shared_use(pr.slot)).unwrap_or(0))
            .collect();
        let primary = if h > 0 {
            PackedPrimary::Embeds(&rows, h)
        } else {
            PackedPrimary::Ids(&ids)
        };
        let (odt, oshape, obytes) = self.run_plan(&plan, primary, &positions, &reuse, &extras)?;
        if let PackedStepKind::Prefill { slot, .. } = &kind {
            if let Some(t) = self.slots[*slot].as_mut() {
                t.prompt_fed += plan.active_rows().count();
            }
        }
        Ok(Some((odt, oshape, obytes, kind)))
    }

    /// Plan the next inference: a prefill chunk for the first slot still
    /// prefilling, else one decode row per occupied slot (decode rows are
    /// embedded by `step`). A slot whose own chunk cannot be planned (an
    /// image wider than `packed_seq`, embedding rows that do not match its
    /// prompt) is retired with its own error, collected by
    /// [`PackedState::take_failed`]; the other slots are planned as if it had
    /// never been admitted. Nothing here can fail the batch.
    pub(crate) fn plan_step(&mut self) -> Option<Planned> {
        while let Some(slot) = self.next_prefill_slot() {
            match self.plan_prefill(slot) {
                Ok(p) => return Some(p),
                Err(e) => {
                    if let Some(t) = self.retire(slot) {
                        self.failed.push((t.task, e));
                    }
                }
            }
        }
        let s = self.kv.packed_seq;
        let ready: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, t)| t.as_ref().map(|_| i))
            .collect();
        if ready.is_empty() {
            return None;
        }
        let ids: Vec<i64> = ready
            .iter()
            .map(|&i| self.slots[i].as_ref().unwrap().last_token as i64)
            .collect();
        let positions: Vec<i64> = ready.iter().map(|&i| self.kv.position(i) as i64).collect();
        let rows: Vec<(usize, usize)> = ready.iter().copied().enumerate().collect();
        // Qwen VL: each decode row continues its slot from `rope_delta`. The
        // DeepStack feed is all zero — a decode token is text (prompt image
        // tokens are all fed during prefill), matching `deepstack_inputs` on an
        // empty image list.
        let extras = PackedExtras {
            mrope: self.decode_mrope(&ready),
            ds_mask: self.zero_ds_mask(),
            ds_embeds: self.zero_ds_embeds(),
        };
        let plan = PackedPlan::decode(s, &ready);
        (!plan.is_empty()).then_some(Planned {
            plan,
            ids,
            positions,
            kind: PackedStepKind::Decode { rows },
            rows: Vec::new(),
            extras,
        })
    }

    /// `[3, S]` row-major M-RoPE positions for a decode batch: each ready
    /// slot's next token at its own `rope_delta`. Empty on a non-Qwen tree.
    fn decode_mrope(&self, ready: &[usize]) -> Vec<i64> {
        if !self.mrope {
            return Vec::new();
        }
        let (s, r) = (self.kv.packed_seq, crate::qwen_vl_vision::MROPE_ROWS);
        let mut out = vec![0i64; r * s];
        for (col, &slot) in ready.iter().enumerate() {
            let t = self.slots[slot].as_ref().unwrap();
            let pos = self.kv.position(slot);
            let p = crate::qwen_vl_vision::positions(t.mrope.as_ref(), pos, pos + 1);
            for row in 0..r {
                out[row * s + col] = p[row];
            }
        }
        out
    }

    /// An all-zero DeepStack mask / embeds of the packed shape (Qwen3-VL stage
    /// 0), empty on every other stage — fed on a decode step and on idle rows.
    fn zero_ds_mask(&self) -> Vec<i64> {
        if self.deepstack_layers == 0 {
            return Vec::new();
        }
        vec![0i64; self.kv.packed_seq]
    }
    fn zero_ds_embeds(&self) -> Vec<f32> {
        if self.deepstack_layers == 0 {
            return Vec::new();
        }
        vec![0f32; self.deepstack_layers * self.kv.packed_seq * self.deepstack_hidden]
    }

    /// The next prefill chunk of `slot`, or why that slot cannot continue.
    fn plan_prefill(&self, slot: usize) -> Result<Planned, String> {
        let s = self.kv.packed_seq;
        let h = self.embed_hidden;
        let t = self.slots[slot].as_ref().expect("a prefilling slot");
        let start = t.prompt_fed;
        // An image is attended bidirectionally only inside one inference:
        // end the chunk before it rather than split it. Submit refuses any
        // image wider than `packed_seq`; this guards the slot regardless.
        let take = image_safe_take(&t.prompt_images, start, (t.prompt_ids.len() - start).min(s))?;
        if take == 0 {
            return Err(format!("packed slot {slot}: empty prefill chunk"));
        }
        let ids: Vec<i64> = t.prompt_ids[start..start + take].to_vec();
        let images = t
            .prompt_images
            .get(start..start + take)
            .unwrap_or_default()
            .to_vec();
        let mut rows = Vec::new();
        if h > 0 {
            if t.prompt_rows.len() != t.prompt_ids.len() * h {
                return Err(format!(
                    "packed slot {slot}: {} embedding floats for {} prompt tokens of {h}",
                    t.prompt_rows.len(),
                    t.prompt_ids.len()
                ));
            }
            rows = t.prompt_rows[start * h..(start + take) * h].to_vec();
        }
        let base = self.kv.position(slot);
        let positions: Vec<i64> = (0..take).map(|i| (base + i) as i64).collect();
        let finished = start + take >= t.prompt_ids.len();
        // Qwen VL: this chunk's M-RoPE positions are the slot's prompt
        // positions over `[base, base + take)`, laid into the plan's first
        // `take` columns.
        let mut extras = PackedExtras::default();
        if self.mrope {
            let r = crate::qwen_vl_vision::MROPE_ROWS;
            let p = crate::qwen_vl_vision::positions(t.mrope.as_ref(), base, base + take);
            let mut out = vec![0i64; r * s];
            for row in 0..r {
                for c in 0..take {
                    out[row * s + c] = p[row * take + c];
                }
            }
            extras.mrope = out;
        }
        // Qwen3-VL stage 0: the chunk's DeepStack mask + embeds, image tokens
        // sliced from this slot's buffer at its own image-token cursor (the
        // image tokens fed before `start`). Once the slot's image tokens are
        // all fed (a text turn: `ds_total == 0`) every row is text, a pad id
        // in the text included, so the feed is all zero — as on the static
        // path (`take_static_deepstack`) and in decode.
        if self.deepstack_layers > 0 {
            let cursor = t.prompt_ids[..start]
                .iter()
                .filter(|&&id| id == self.deepstack_image_pad)
                .count();
            if cursor >= t.ds_total {
                extras.ds_mask = self.zero_ds_mask();
                extras.ds_embeds = self.zero_ds_embeds();
            } else {
                let (mask, embeds, _k) = crate::qwen_vl_vision::deepstack_window(
                    &ids,
                    self.deepstack_image_pad,
                    &t.ds_global,
                    t.ds_total,
                    cursor,
                    self.deepstack_layers,
                    self.deepstack_hidden,
                    s,
                )?;
                extras.ds_mask = mask;
                extras.ds_embeds = embeds;
            }
        }
        Ok(Planned {
            plan: if images.is_empty() {
                PackedPlan::chunk(s, slot, take)
            } else {
                PackedPlan::chunk_with_images(s, slot, &images)
            },
            ids,
            positions,
            kind: PackedStepKind::Prefill {
                slot,
                last_row: take - 1,
                finished_prompt: finished,
            },
            rows,
            extras,
        })
    }

    /// Tasks whose slot was retired by planning since the last call, each
    /// with its own error. The caller answers each one; the batch goes on.
    pub(crate) fn take_failed(&mut self) -> Vec<(GenerationTask, String)> {
        std::mem::take(&mut self.failed)
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
        extras: &PackedExtras,
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
        self.last_plan_images = (0..s)
            .map(|r| plan.rows.get(r).copied().flatten().and_then(|pr| pr.image))
            .collect();
        // M-RoPE positions ship to the relay stages on the wire.
        self.last_plan_mrope = extras.mrope.clone();
        if self.mrope && extras.mrope.len() != crate::qwen_vl_vision::MROPE_ROWS * s {
            return Err(EngineError::Backend(format!(
                "packed M-RoPE tree: {} positions for {} rows (need {})",
                extras.mrope.len(),
                s,
                crate::qwen_vl_vision::MROPE_ROWS * s
            )));
        }
        let ids: Vec<i64> = match primary {
            PackedPrimary::Ids(v) => v.to_vec(),
            PackedPrimary::Hidden(_, _) | PackedPrimary::Embeds(_, _) => Vec::new(),
        };

        // ---- feed ----
        for m in self.masks.iter_mut() {
            self.kv
                .fill_mask_window(plan, &mut m.bytes, m.dtype, m.window);
        }
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
            PackedPrimary::Embeds(rows, hidden_size) => {
                // Same layout as Hidden, kept f32 like every other
                // `inputs_embeds` feed.
                let mut bytes = vec![0u8; s * hidden_size * 4];
                let n = (rows.len() / hidden_size).min(s) * hidden_size;
                for (dst, v) in bytes.chunks_exact_mut(4).zip(&rows[..n]) {
                    dst.copy_from_slice(&v.to_le_bytes());
                }
                self.runtime
                    .set_input(&self.ids_in, ShimDType::F32, &[1, s, hidden_size], &bytes)
                    .map_err(crate::runtime::map_ov_err)?;
            }
        }
        for m in &self.masks {
            self.runtime
                .set_input(&m.port, m.dtype, &[1, 1, s, self.kv.context], &m.bytes)
                .map_err(crate::runtime::map_ov_err)?;
        }
        if let Some(types_in) = &self.types_in {
            self.types_bytes.clear();
            for r in 0..s {
                let t: i64 = plan
                    .rows
                    .get(r)
                    .copied()
                    .flatten()
                    .map_or(0, |pr| pr.image.is_some() as i64);
                self.types_bytes.extend_from_slice(&t.to_le_bytes());
            }
            self.runtime
                .set_input(types_in, ShimDType::I64, &[1, s], &self.types_bytes)
                .map_err(crate::runtime::map_ov_err)?;
        }
        // Positions: 1-D `[1, S]` on a text/other tree, M-RoPE `[3, 1, S]` on a
        // Qwen VL tree (per-row t/h/w; the sequential `positions` still drives
        // the ring reset and mask above).
        if self.mrope {
            let mut buf = Vec::with_capacity(extras.mrope.len() * 8);
            for v in &extras.mrope {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            self.runtime
                .set_input(
                    &self.pos_in,
                    ShimDType::I64,
                    &[crate::qwen_vl_vision::MROPE_ROWS, 1, s],
                    &buf,
                )
                .map_err(crate::runtime::map_ov_err)?;
        } else {
            self.runtime
                .set_input(&self.pos_in, ShimDType::I64, &[1, s], &self.pos_bytes)
                .map_err(crate::runtime::map_ov_err)?;
        }
        // Qwen3-VL stage 0 DeepStack: `visual_pos_masks [1, S]` +
        // `deepstack_visual_embeds [layers, S, hidden]`. Only stage 0 has these
        // ports (the relay stages carry no DeepStack rows on the wire).
        if let (Some(mask_in), Some(ds_in)) =
            (self.visual_mask_in.as_deref(), self.deepstack_in.as_deref())
        {
            let mut mbuf = Vec::with_capacity(extras.ds_mask.len() * 8);
            for v in &extras.ds_mask {
                mbuf.extend_from_slice(&v.to_le_bytes());
            }
            let mut ebuf = Vec::with_capacity(extras.ds_embeds.len() * 4);
            for v in &extras.ds_embeds {
                ebuf.extend_from_slice(&v.to_le_bytes());
            }
            self.runtime
                .set_input(mask_in, ShimDType::I64, &[1, s], &mbuf)
                .map_err(crate::runtime::map_ov_err)?;
            self.runtime
                .set_input(
                    ds_in,
                    ShimDType::F32,
                    &[self.deepstack_layers, s, self.deepstack_hidden],
                    &ebuf,
                )
                .map_err(crate::runtime::map_ov_err)?;
        }
        for (li, layer) in self.layers.iter().enumerate() {
            let kv_shape = [
                1,
                self.kv.kv_heads(li),
                self.kv.past_len,
                self.kv.head_dim(li),
            ];
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
        // occupancy per slot BEFORE this step drives append-vs-slide
        let mut at: HashMap<usize, usize> = HashMap::new();
        for (_r, pr) in plan.active_rows() {
            at.entry(pr.slot).or_insert_with(|| self.kv.valid(pr.slot));
        }
        for li in 0..self.layers.len() {
            let expect = self.kv.present_layer_bytes(li);
            let want_shape = [
                1usize,
                self.kv.kv_heads(li),
                self.kv.context,
                self.kv.head_dim(li),
            ];
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

#[cfg(all(test, not(feature = "openvino")))]
mod tests {
    use super::*;

    fn state(slots: usize, packed_seq: usize) -> PackedState {
        let kv = PackedKv::new(slots, 32, packed_seq, 1, 1, 2, ShimDType::F16, 0);
        let masks = vec![PackedMask::new("attn_mask_4d".into(), ShimDType::F16, None)];
        PackedState::new(
            OvRuntime::stub(),
            kv,
            Vec::new(),
            "input_ids".into(),
            String::new(),
            0,
            masks,
            "position_ids".into(),
        )
    }

    /// A slot whose chunk cannot be planned (here an image wider than one
    /// packed inference, which submit would have refused) is retired with its
    /// own error; the other slot still decodes in the same step.
    #[test]
    fn a_slot_that_cannot_be_planned_fails_alone() {
        let mut st = state(2, 4);
        st.image_token = Some(9);
        st.admit(
            0,
            GenerationTask::new("ok", "hi"),
            vec![1, 2],
            Vec::new(),
            2,
            None,
            Vec::new(),
            0,
        );
        {
            let t = st.slots[0].as_mut().unwrap();
            t.prompt_fed = 2; // prefilled, decoding
            t.last_token = 7;
        }
        st.kv.advance(0, 2);
        st.admit(
            1,
            GenerationTask::new("bad", "img"),
            vec![9, 9, 9, 9, 9, 2],
            Vec::new(),
            0,
            None,
            Vec::new(),
            0,
        );
        let planned = st.plan_step().expect("slot 0 still decodes");
        match planned.kind {
            PackedStepKind::Decode { rows } => assert_eq!(rows, vec![(0, 0)]),
            _ => panic!("expected a decode step for slot 0"),
        }
        assert_eq!(planned.ids, vec![7]);
        let failed = st.take_failed();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0.task_id, TaskIdOf::from("bad"));
        assert!(failed[0].1.contains("does not fit"), "{}", failed[0].1);
        assert!(st.slots[1].is_none(), "the failed slot is free again");
        assert!(st.slots[0].is_some());
        assert!(st.take_failed().is_empty());
    }

    /// A Qwen VL slot's prefill chunk carries the prompt's M-RoPE positions
    /// over its rows, and a later decode step continues from `rope_delta` —
    /// each slot with its own delta.
    #[test]
    fn packed_mrope_prefill_and_decode_per_slot() {
        use crate::qwen35_vision::rope::mrope_positions;
        let mut st = state(2, 4);
        st.mrope = true;
        // [text, image 2x2, text]: the image at index 1 shifts later positions.
        let ids = vec![1i64, 9, 9, 9, 9, 2];
        let m = mrope_positions(&ids, 9, 8, &[[1, 4, 4]], &[], 2).unwrap();
        assert_eq!(m.rope_delta, -2);
        st.admit(
            0,
            GenerationTask::new("q", "p"),
            ids.clone(),
            Vec::new(),
            ids.len(),
            Some(m.clone()),
            Vec::new(),
            0,
        );
        let planned = st.plan_step().expect("prefill chunk");
        let s = st.kv.packed_seq;
        let take = planned.plan.active_rows().count();
        assert_eq!(take, 4); // packed_seq-wide chunk of the 6-token prompt
        let want = crate::qwen_vl_vision::positions(Some(&m), 0, take);
        for r in 0..3 {
            for c in 0..take {
                assert_eq!(planned.extras.mrope[r * s + c], want[r * take + c]);
            }
            // Columns past `take` are zero padding.
            for c in take..s {
                assert_eq!(planned.extras.mrope[r * s + c], 0);
            }
        }
        // Decode after the whole prompt (position 6): index + rope_delta = 4.
        st.kv.advance(0, 6);
        let dec = st.decode_mrope(&[0]);
        assert_eq!((dec[0], dec[s], dec[2 * s]), (4, 4, 4));
    }

    /// Mismatched embedding rows on a vision stage 0 fail that slot only.
    #[test]
    fn bad_embedding_rows_fail_only_their_slot() {
        let mut st = state(2, 4);
        st.embed_hidden = 3;
        st.admit(
            0,
            GenerationTask::new("a", "x"),
            vec![1, 2],
            vec![0.0; 6],
            2,
            None,
            Vec::new(),
            0,
        );
        st.admit(
            1,
            GenerationTask::new("b", "y"),
            vec![1, 2],
            vec![0.0; 5],
            2,
            None,
            Vec::new(),
            0,
        );
        let planned = st.plan_step().expect("slot 0 prefills");
        assert!(matches!(
            planned.kind,
            PackedStepKind::Prefill { slot: 0, .. }
        ));
        // slot 0's chunk is planned first; slot 1 fails when its turn comes
        st.slots[0].as_mut().unwrap().prompt_fed = 2;
        let _ = st.plan_step();
        let failed = st.take_failed();
        assert_eq!(failed.len(), 1);
        assert!(failed[0].1.contains("embedding floats"), "{}", failed[0].1);
        assert!(st.slots[1].is_none() && st.slots[0].is_some());
    }

    /// A Qwen3-VL text turn (`ds_total == 0`) whose prompt carries the image
    /// pad id still prefills, with the all-zero DeepStack feed decode uses,
    /// instead of retiring the slot for running past its image tokens.
    #[test]
    fn deepstack_text_turn_with_a_pad_id_gets_the_zero_feed() {
        let mut st = state(1, 4);
        st.deepstack_layers = 2;
        st.deepstack_hidden = 3;
        st.deepstack_image_pad = 9;
        st.admit(
            0,
            GenerationTask::new("t", "x"),
            vec![1, 9, 2],
            Vec::new(),
            3,
            None,
            Vec::new(),
            0,
        );
        let planned = st.plan_step().expect("prefill chunk");
        assert!(st.take_failed().is_empty());
        assert!(matches!(
            planned.kind,
            PackedStepKind::Prefill { slot: 0, .. }
        ));
        assert_eq!(planned.extras.ds_mask, vec![0i64; 4]);
        assert_eq!(planned.extras.ds_embeds, vec![0f32; 2 * 4 * 3]);
    }

    type TaskIdOf = cascadia_types::TaskId;
}
