//! Issue-34 plane warm-resume: the one-slot mailbox every engine parks a pulled slice in.
//!
//! Lives here, not in an engine crate, because the `kv_handoff_*` events it emits are what the
//! certification greps. A per-engine copy would let those names — and the conditions that emit them —
//! drift apart silently, which is how the 2026-08-02 cert lost a cycle to three engines that looked
//! identical and were not.
//!
//! What stays engine-local is the DECISION: validating a slice needs the engine's own layout constant,
//! engine-rev and decoder (OV's opaque single-payload blob vs sparse-MoE's structured per-layer
//! snapshot), so no shared helper can check both.

use std::sync::Mutex;

use cascadia_kv_wire::Manifest;

/// A plane-pulled slice parked for the engine to apply itself.
pub struct KvHandoffSlot {
    pub epoch: u64,
    pub manifest: Manifest,
    pub payloads: Vec<(Vec<u8>, Vec<u8>)>,
}

/// One-slot mailbox handing a pulled warm-resume slice from the KV plane to the engine's recv loop.
///
/// Its mutex is INDEPENDENT of the engine mutex, and the producer side never touches the engine at
/// all. The confirm/commit path runs on the node's control task while the tail is typically parked
/// inside `step()` holding the engine mutex — reaching the engine from there deadlocks (proven
/// on-rig). So the producer only parks bytes here; the engine drains the mailbox from inside its own
/// recv loop, where it already owns its lock and can still apply ahead of the turn's forward.
pub struct KvHandoffMailbox {
    inner: Mutex<MailboxInner>,
}

#[derive(Default)]
struct MailboxInner {
    /// Slices dropped because the RESTORE asked for a different epoch (see `take`).
    epoch_mismatches: u64,
    /// Last epoch DROPPED as foreign by `take`. Distinct from `drained` on purpose: a dropped slice
    /// never reached the engine, so a later `clear` for it must not read as "the engine already took
    /// it, this rank MAY be warm under a cold head" — that rank is provably COLD.
    dropped: Option<u64>,
    slot: Option<KvHandoffSlot>,
    /// Last epoch the engine TOOK, so a `clear` that finds the slot empty can tell "already gone to
    /// the engine" from "this rank never held it" — the two have opposite consequences for the head.
    /// Set by `take` before the drain validates / decodes / depth-guards / applies, so it does NOT
    /// mean applied: a taken-then-rejected slice records here too.
    drained: Option<u64>,
    too_late: u64,
    /// Drains that found the mailbox empty, and whether a slice has ever landed here at all. Together
    /// they separate the ordinary no-plane-slice drain from the engine running ahead of the commit —
    /// the race that was otherwise diagnosable only by inference across two runs.
    empty_drains: u64,
    ever_parked: bool,
}

impl KvHandoffMailbox {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(MailboxInner::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MailboxInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Overwrites any unconsumed slice: only the newest pull can still be ahead of the engine's
    /// position, and an older one would only be rejected by the apply-site depth guard anyway.
    pub fn put(&self, epoch: u64, manifest: Manifest, payloads: Vec<(Vec<u8>, Vec<u8>)>) {
        let n_payloads = payloads.len();
        let mut g = self.lock();
        g.ever_parked = true;
        // A fresh park supersedes any earlier foreign-drop marker for this mailbox.
        g.dropped = None;
        g.slot = Some(KvHandoffSlot {
            epoch,
            manifest,
            payloads,
        });
        drop(g);
        tracing::info!(target: "cascadia::kv", event = "kv_handoff_put", epoch, n_payloads);
    }

    /// Drain the parked slice for `expected_epoch` — the epoch of the RESTORE being served.
    ///
    /// **Epoch-bound on purpose.** This used to take whatever was parked, and no layer below it
    /// looked at the epoch either (`handoff_decision` is not given one), while the `OPCODE_RESTORE`
    /// arm decodes the epoch and drains FIRST — ahead of the head's own carried blob. So a slice
    /// stranded by an earlier turn was applied on the NEXT turn's RESTORE, for a different session
    /// with different tokens, and the rank acked warm: wrong output with nothing in the failure
    /// taxonomy firing. Stranding is not hypothetical — a drain that races ahead of the plane's
    /// first `put` is recorded in `kv_handoff_drain_empty`'s note, and a head that commits and then
    /// dies sends neither RESTORE nor abort while this mailbox has no TTL.
    ///
    /// A mismatch DROPS the slice rather than re-parking it: it belongs to a move whose RESTORE is
    /// not coming, and leaving it parked keeps the landmine armed for the turn after this one. The
    /// cost is a cold reprefill, which is the safety floor this path is supposed to have.
    pub fn take(&self, expected_epoch: u64) -> Option<KvHandoffSlot> {
        let mut g = self.lock();
        if let Some(s) = g.slot.as_ref() {
            if s.epoch != expected_epoch {
                let parked = s.epoch;
                g.slot = None;
                g.dropped = Some(parked);
                g.epoch_mismatches += 1;
                let n = g.epoch_mismatches;
                drop(g);
                tracing::warn!(target: "cascadia::kv", event = "kv_handoff_epoch_mismatch",
                    parked, expected = expected_epoch, count = n);
                return None;
            }
        }
        let slot = g.slot.take();
        match &slot {
            Some(s) => {
                g.drained = Some(s.epoch);
                // Clear the foreign-drop marker: `synth_epoch` is a content hash, so epoch REUSE is
                // guaranteed, and a sticky `dropped` made a later `clear(E)` for a genuinely drained
                // slice read as "dropped as foreign, rank provably cold" — suppressing `too_late`
                // and breaking the upper-bound property `aborts_too_late` documents.
                g.dropped = None;
            }
            None => {
                g.empty_drains += 1;
                // An empty drain on a rank the plane never fed is routine and scales with turns, so
                // it stays DEBUG. An empty drain on a rank that WAS parked into is the
                // slice-stranded-under-a-warm-head pathology and must be visible at the level the rig
                // actually runs (`info`), or it can only be inferred across runs — which is exactly
                // how the 2026-08-02 cert had to diagnose it.
                // `ever_parked == false` is NOT purely routine: it also covers the drain that runs
                // BEFORE the plane's first put, which strands the slice for the rest of the turn and
                // leaves the rank cold under a warm head. At DEBUG that case is invisible at the
                // level the rig runs, so a lost race can only be inferred from an ABSENT
                // `kv_handoff_applied_inline` — which is exactly how the 2026-08-04 cert had to find
                // it. Make the first few loud either way; the steady-state repeats stay DEBUG so the
                // turn-scaling counter cannot flood a long run.
                const LOUD_EMPTY_DRAINS: u64 = 3;
                if g.ever_parked || g.empty_drains <= LOUD_EMPTY_DRAINS {
                    tracing::info!(target: "cascadia::kv", event = "kv_handoff_drain_empty",
                        count = g.empty_drains, ever_parked = g.ever_parked);
                } else {
                    tracing::debug!(target: "cascadia::kv", event = "kv_handoff_drain_empty",
                        count = g.empty_drains, ever_parked = false);
                }
            }
        }
        slot
    }

    /// Retract a parked slice (see [`crate::KvWarmHandoff::clear`]). Epoch-matched so an abort for a
    /// stale epoch cannot drop a newer pull that overwrote it.
    ///
    /// A `false` for an epoch this mailbox already drained is the residual the retraction cannot
    /// close — the abort races the engine's recv-loop drain — so it is counted, not swallowed.
    pub fn clear(&self, epoch: u64) -> bool {
        let mut g = self.lock();
        let retracted = g.slot.as_ref().is_some_and(|s| s.epoch == epoch);
        if retracted {
            g.slot = None;
        }
        // A slice this mailbox DROPPED as foreign never reached the engine, so the abort is clean
        // and the rank is provably cold. Counting it as `too_late` would raise the
        // warm-under-a-cold-head hazard for a rank that cannot be warm, and would inflate
        // `aborts_too_late`, which is documented as the UPPER BOUND on that residual.
        let dropped_foreign = !retracted && g.dropped == Some(epoch);
        let too_late = !retracted && !dropped_foreign && g.drained == Some(epoch);
        if dropped_foreign {
            tracing::info!(target: "cascadia::kv", event = "kv_handoff_cleared",
                epoch, retracted = false, dropped_foreign = true);
        } else if too_late {
            g.too_late += 1;
            tracing::warn!(target: "cascadia::kv", event = "kv_handoff_abort_too_late",
                epoch, count = g.too_late);
        } else {
            tracing::info!(target: "cascadia::kv", event = "kv_handoff_cleared", epoch, retracted);
        }
        retracted
    }

    /// Epoch-blind retraction for the chain `ABORT`, whose frame is a bare opcode byte with no epoch
    /// to match on. `true` if a slice was still parked.
    ///
    /// Deliberately leaves `drained` alone: that field is what makes a later `clear(epoch)` count as a
    /// lost race, and a discarded slice never reached the engine at all.
    pub fn discard_any(&self) -> bool {
        self.lock().slot.take().is_some()
    }

    /// Aborts that arrived after the engine had already taken the slice. An UPPER BOUND on the
    /// warm-rank-under-cold-head residual, not a count of it: `drained` is stamped at `take`, so a
    /// slice the drain then rejected leaves that rank cold and is counted here anyway.
    pub fn aborts_too_late(&self) -> u64 {
        self.lock().too_late
    }

    /// Slices dropped because the RESTORE asked for a different epoch. Nonzero means either a
    /// genuinely stale parked slice was refused (the guard working) or the head's warm-entry
    /// selection drifted off the plane-pulled entry (a warm→cold conversion worth investigating).
    pub fn epoch_mismatches(&self) -> u64 {
        self.lock().epoch_mismatches
    }

    /// Drains that found nothing parked. Read with [`Self::ever_parked`].
    pub fn empty_drains(&self) -> u64 {
        self.lock().empty_drains
    }

    /// Whether any slice has ever been parked here. An empty drain with this `true` is the engine
    /// running ahead of the plane's commit; with it `false` the rank simply has no plane traffic.
    pub fn ever_parked(&self) -> bool {
        self.lock().ever_parked
    }
}

impl Default for KvHandoffMailbox {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::KvWarmHandoff for KvHandoffMailbox {
    fn put(&self, epoch: u64, manifest: Manifest, payloads: Vec<(Vec<u8>, Vec<u8>)>) {
        KvHandoffMailbox::put(self, epoch, manifest, payloads);
    }

    fn clear(&self, epoch: u64) -> bool {
        KvHandoffMailbox::clear(self, epoch)
    }
}
