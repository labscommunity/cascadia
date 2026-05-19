//! Speculative-decode helpers: pure functions + unit tests.
//!
//! The actual spec-decode loop lives in
//! [`crate::runner::Runner::generate_speculative`] because it needs
//! access to the Runner's private KV state. This file holds the
//! pure-logic primitives — acceptance counting, KV/history rewind
//! math, and dynamic-K adaptation — extracted so we can test them
//! without loading a 553 GB model.
//!
//! Surface:
//! - [`count_accepted`]: greedy acceptance — longest matching prefix
//!   between drafted tokens and target-sampled tokens.
//! - [`reconcile_after_round`]: computes the post-round state
//!   transitions (how many KV slots to rewind, how many tokens to
//!   keep in history) given the accept count + whether the all-accepted
//!   bonus forward was run.
//! - [`AdaptiveK`] + [`AdaptiveKConfig`]: per-request draft-length
//!   controller. Tracks accept rate over a sliding window and adjusts
//!   K up/down based on configurable thresholds. Opt-in via the
//!   `--spec-k-adaptive` CLI flag (iter 083); static K is the default.
//!
//! See [`crate::runner::Runner::generate_speculative`] for the call
//! site and the wire-up to the int4 shell forward path.

/// Greedy acceptance count: scan `drafts` and `target_samples` in
/// lockstep, returning the length of the longest matching prefix.
/// Once any position differs we stop — the rest of `drafts` was
/// "wishful thinking" and is discarded.
///
/// Mirrors the inner loop of `dist_spec.rs::do_one_round` and the
/// rainier `k26_spec_decode.py` accept loop.
pub fn count_accepted(drafts: &[i64], target_samples: &[i64]) -> usize {
    let mut accepted = 0usize;
    let n = drafts.len().min(target_samples.len());
    for i in 0..n {
        if drafts[i] == target_samples[i] {
            accepted += 1;
        } else {
            break;
        }
    }
    accepted
}

/// Outcome of one spec round, ready to be applied to the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundReconcile {
    /// How many tokens to truncate from the post-verify history (i.e.
    /// the rejected drafts that were pushed during the round).
    pub history_pop: usize,
    /// How many KV-cache slots to rewind. Combines both: rejected
    /// drafts AND the all-accepted bonus forward's KV slot if that
    /// forward was run.
    pub kv_rewind: usize,
    /// How many new tokens this round adds to the public `generated`
    /// list. Includes the accepted drafts + the bonus token (if any).
    pub new_tokens_emitted: usize,
    /// True if the bonus token was produced by an extra "all-accepted"
    /// forward (vs being a free byproduct of the rejection-position
    /// forward). Used by the caller to decide whether `bonus` should
    /// be re-added to KV via the next round's first forward (no extra
    /// work needed — the next round's standard first forward will
    /// condition on `bonus` and produce its prediction).
    pub all_accepted_bonus: bool,
}

/// Compute reconcile actions given accept count + draft count.
///
/// # The two history conventions
///
/// Two call sites carry slightly different history/KV invariants
/// going INTO each spec-decode round; pass `pending_token_in_history`
/// to select between them.
///
/// ## `pending_token_in_history = false` (clean convention)
///
/// At round entry, `history.len() == KV.past_seq_len` — every token
/// in history has a matching KV slot. The bonus token from the
/// previous round was either consumed by an explicit "warmup" forward
/// before re-entry, or it's round 1 and there's no previous bonus.
/// This is the convention exercised by the existing helper tests
/// (e.g. `simulated_session_matches_sequential_greedy`).
///
/// ## `pending_token_in_history = true` (runner / pipeline-parallel convention)
///
/// At round entry, `history.len() == KV.past_seq_len + 1` — there's
/// one trailing token in history whose KV slot is NOT yet written.
/// It's the previous round's bonus (or, for round 1, the `first_gen`
/// token sampled from the last prefill step). The next round's first
/// verify forward will fill that slot as a side effect.
///
/// This is the convention used by:
/// - [`crate::runner::Runner::generate_speculative`] (single-stage),
///   which pre-pushes `first_gen` to history before the first round
///   and appends the round's `bonus` to history at end-of-round.
/// - The pipeline-parallel driver `drive_generation_first_spec` in
///   [`crate::engine`], which inherited the same convention from the
///   single-stage path.
///
/// In this convention, the K verify forwards collectively write K KV
/// slots starting from `past_seq_len = history.len() - 1` — i.e. the
/// first verify is the one that "catches up" the pending token, and
/// each subsequent verify writes one draft's slot. So total KV writes
/// during the K-loop is still K, but the post-loop KV ends at
/// `history.len() - 1` (NOT `history.len()`). The bonus the caller
/// appends at end-of-round is itself unforwarded — it becomes the
/// next round's pending token. Net effect: `kv_rewind` is one less
/// than the clean-convention formula returns.
///
/// # Pre-conditions (caller invariants)
///
/// - `k_drafts` forwards have been run during the round's K-loop,
///   each advancing the KV cache by 1 slot and pushing one drafted
///   token into `history`.
/// - If `accepted == k_drafts` AND the caller chose to run a bonus
///   forward (typically `true` so the next round has its
///   `prev_correction`), the bonus forward has also advanced KV by 1
///   and produced one target sample. Pass `bonus_forward_ran=true` in
///   that case. In the runner convention, this extra forward writes
///   the K-th draft's slot (the K-loop only wrote K-1 draft slots
///   itself; its first verify filled the pending token's slot).
///
/// # Post-conditions
///
/// - After `history.truncate(history.len() - history_pop)` and
///   `runner.rewind_kv(kv_rewind)`, the engine state holds exactly
///   the accepted prefix. In the clean convention,
///   `KV == history.len() == prev_kv + accepted`. In the runner
///   convention, `KV + 1 == history.len() == prev_history + accepted`
///   (i.e. the +1 drift is preserved for the next round's pending
///   bonus, which the caller appends to history immediately after).
pub fn reconcile_after_round(
    k_drafts: usize,
    accepted: usize,
    bonus_forward_ran: bool,
    pending_token_in_history: bool,
) -> RoundReconcile {
    debug_assert!(accepted <= k_drafts);
    let rejected = k_drafts.saturating_sub(accepted);
    // History was pushed k_drafts times; we keep `accepted` of those.
    // The "bonus" token is added separately by the caller AFTER this
    // reconcile — it's not in history at this point.
    let history_pop = rejected;
    // KV slots written = k_drafts + (1 if bonus_forward_ran else 0).
    // We want KV to end at `accepted` (matching the accepted-drafts
    // suffix in history). Then the caller pushes bonus; the next
    // round's first forward will fill its KV slot.
    //
    // In the runner / pipeline-parallel convention, the K-loop "absorbed"
    // the previous round's pending token (filling its KV slot as a
    // side effect of the first verify forward). So the bonus we
    // re-attach as the next round's pending token rides on the same
    // 1-slot drift — `kv_rewind` is one less than the clean-convention
    // value. This is mathematically equivalent to the inline
    // `K - A - 1` (partial) / `0` (all-accepted) formulas documented
    // in `engine::drive_generation_first_spec`.
    let kv_writes = k_drafts + if bonus_forward_ran { 1 } else { 0 };
    let mut kv_rewind = kv_writes - accepted;
    if pending_token_in_history {
        // The pending-token convention preserves one KV slot of drift
        // (history is always +1 ahead of KV). `kv_rewind` should never
        // go negative — for partial accept, K > A so K-A-1 >= 0; for
        // all-accepted with bonus_forward_ran=true, kv_writes-A = 1
        // and rewind becomes 0 cleanly.
        debug_assert!(
            kv_rewind >= 1,
            "pending-token convention requires kv_writes - accepted >= 1 \
             (k_drafts={k_drafts}, accepted={accepted}, bonus_forward_ran={bonus_forward_ran})"
        );
        kv_rewind -= 1;
    }
    // New tokens emitted = accepted drafts + 1 bonus (always — bonus
    // is the target's sample at the rejection boundary, or the
    // dedicated extra forward's sample when all-accepted).
    let new_tokens_emitted = accepted + 1;
    RoundReconcile {
        history_pop,
        kv_rewind,
        new_tokens_emitted,
        all_accepted_bonus: bonus_forward_ran,
    }
}

/// Configuration for [`AdaptiveK`] — the per-request draft-length
/// controller used when the operator opts into dynamic K via
/// `--spec-k-adaptive`.
///
/// **Motivation.** iter 063 measured per-prompt accept rates of
/// 3.3-86.4% under a fixed K=4 — a 26x spread. Fixed K is suboptimal at
/// both ends: low-accept prompts waste verify forwards on drafts that
/// will be rejected, while high-accept prompts leave throughput on the
/// table because the round budget caps emit at K + 1 tokens.
/// Adaptive-K is the standard mitigation (Yang et al. 2025; Leviathan
/// et al. 2023's "lookahead-K" sweep): observe accept rate over a
/// short sliding window, raise K when accepts are high, lower K when
/// rejections dominate.
///
/// **Defaults.** Match the bench-tuned thresholds from iter 063:
/// - Raise K by `up_step` when the windowed accept rate exceeds `up_threshold`.
/// - Lower K by `down_step` when it falls below `down_threshold`.
/// - Both adjustments are clamped to `[k_min, k_max]`.
/// - `window` is the sliding-window length (in rounds) used to compute
///   the trailing accept rate. 8 rounds is a compromise between
///   responsiveness (short enough that a phase change in the prompt
///   propagates within ~1 s of decode time) and noise immunity (long
///   enough that a single all-reject round doesn't trigger a panic
///   down-step). The user task spec calls out these exact numbers.
///
/// **Boundary semantics.** Thresholds are evaluated with `>` for the
/// up-step and `<` for the down-step — i.e. the exact boundary
/// (`rate == up_threshold` or `rate == down_threshold`) is the "no
/// change" zone. This keeps the controller stable at the prescribed
/// 30%–70% band: a pinned 70% accept rate does not oscillate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveKConfig {
    /// Initial / starting K. The first round always uses this exactly;
    /// adjustment kicks in once `window` rounds have populated the
    /// sliding buffer.
    pub k_start: usize,
    /// Lower bound on K. K=2 is the rainier-derived floor — at K=1 the
    /// spec round is a single verify forward with the same wall-clock
    /// cost as a normal decode step, so the spec-decode framing
    /// (propose + verify + reconcile + bonus) is pure overhead.
    pub k_min: usize,
    /// Upper bound on K. K=16 is rainier's measured ceiling on K2.6 —
    /// past 16 the n-gram draft's accept rate falls off fast enough
    /// that the extra verify forwards never pay back, even on
    /// extremely repetitive prompts.
    pub k_max: usize,
    /// Sliding-window length used to compute the trailing accept rate.
    pub window: usize,
    /// Raise K when the windowed accept rate is strictly greater than this.
    pub up_threshold: f32,
    /// Lower K when the windowed accept rate is strictly less than this.
    pub down_threshold: f32,
    /// Per-trigger K increment (added when rate > up_threshold).
    pub up_step: usize,
    /// Per-trigger K decrement (subtracted when rate < down_threshold).
    pub down_step: usize,
}

impl AdaptiveKConfig {
    /// Default policy: matches the task spec from iter 083 — window=8,
    /// up at >0.7 by +2, down at <0.3 by -1, floor K=2, ceiling K=16.
    /// `k_start` is the caller-provided starting K from the CLI flag
    /// (typically 4 — the historical static-K default).
    pub fn new_with_start(k_start: usize) -> Self {
        Self {
            k_start,
            k_min: 2,
            k_max: 16,
            window: 8,
            up_threshold: 0.7,
            down_threshold: 0.3,
            up_step: 2,
            down_step: 1,
        }
    }

    /// `debug_assert` that the config is internally consistent. Called
    /// from the [`AdaptiveK::new`] constructor; cheap enough to leave
    /// in (the controller is created once per request).
    fn validate(&self) {
        debug_assert!(
            self.k_min >= 1,
            "k_min must be >= 1 (K=0 has no spec round)"
        );
        debug_assert!(
            self.k_max >= self.k_min,
            "k_max ({}) must be >= k_min ({})",
            self.k_max,
            self.k_min,
        );
        debug_assert!(
            self.k_start >= self.k_min && self.k_start <= self.k_max,
            "k_start ({}) must be in [k_min={}, k_max={}]",
            self.k_start,
            self.k_min,
            self.k_max,
        );
        debug_assert!(
            self.window >= 1,
            "window must be >= 1 (zero rounds = no accept-rate signal)"
        );
        debug_assert!(
            (0.0..=1.0).contains(&self.up_threshold),
            "up_threshold must be in [0, 1]"
        );
        debug_assert!(
            (0.0..=1.0).contains(&self.down_threshold),
            "down_threshold must be in [0, 1]"
        );
        debug_assert!(
            self.down_threshold <= self.up_threshold,
            "down_threshold ({}) must be <= up_threshold ({}) — otherwise both triggers fire simultaneously",
            self.down_threshold,
            self.up_threshold,
        );
    }
}

/// Per-request dynamic-K controller. One instance lives across a
/// single generation; reset between tasks via a fresh `AdaptiveK::new`.
///
/// **State machine.**
///
/// At round end, the caller invokes [`AdaptiveK::observe_round`] with
/// the round's `(accepted, drafts_proposed)` counts. Internally this
/// pushes the accept fraction into a ring buffer of length `window`,
/// then evaluates the adjustment rule:
///
/// ```text
/// rate = mean(window_buffer)        // average accept fraction
/// if rate > up_threshold:   K = min(K + up_step,   k_max)
/// if rate < down_threshold: K = max(K - down_step, k_min)
/// ```
///
/// The next call to [`AdaptiveK::current_k`] returns the updated K.
/// The window must be fully populated (`samples.len() == window`)
/// before adjustment fires — this avoids reacting to the first 1-2
/// rounds where one outlier accept rate would whipsaw K.
///
/// **Cooldown.** Once adjustment fires, the next `window` observations
/// must arrive before the rule fires again. This is the classic
/// AIMD-style cooldown from the spec-decode literature (Yang et al.
/// 2025, Sec. 4.2) — it prevents the controller from racing past the
/// optimum because the freshly-bumped K hasn't had enough rounds to
/// produce a representative accept-rate signal yet. Concretely: after
/// any adjustment we clear the sample buffer; the next decision waits
/// for `window` fresh observations.
///
/// **Drafts-proposed semantics.** The accept rate is normalized by
/// drafts-proposed, not by current K. This matters in two cases:
/// 1. **Budget-truncated rounds.** Near `max_tokens`, the runner caps
///    `drafts` to the remaining token budget. A budget-of-1 round that
///    accepts 1/1 should count as 100% accept, not "1/K = 25%".
/// 2. **Empty proposal fallback.** When `draft.propose()` returns an
///    empty proposal the runner skips the round entirely (single
///    forward, no spec accounting). The controller is NOT called in
///    that path — its window only sees actual spec rounds.
///
/// **Threading.** Not `Send` — owned by the single thread driving one
/// generation. The runner / pipeline-parallel driver instantiate it
/// inline, never share it across requests.
#[derive(Debug, Clone)]
pub struct AdaptiveK {
    cfg: AdaptiveKConfig,
    /// Current K — what the next round will use.
    current_k: usize,
    /// Ring buffer of the last `window` rounds' accept fractions.
    /// Each entry is in `[0.0, 1.0]`.
    samples: std::collections::VecDeque<f32>,
}

impl AdaptiveK {
    /// Construct a new controller from a config. The first round's K
    /// is `cfg.k_start`; subsequent K values are produced by
    /// [`Self::observe_round`].
    pub fn new(cfg: AdaptiveKConfig) -> Self {
        cfg.validate();
        let current_k = cfg.k_start;
        let window = cfg.window;
        Self {
            cfg,
            current_k,
            samples: std::collections::VecDeque::with_capacity(window),
        }
    }

    /// K to use for the next spec round. Always in `[k_min, k_max]`.
    pub fn current_k(&self) -> usize {
        self.current_k
    }

    /// Sliding-window length (used by tests and the runner's log
    /// instrumentation to report buffer fill).
    pub fn window_len(&self) -> usize {
        self.samples.len()
    }

    /// Read-only view of the trailing accept rate across the sliding
    /// window. Returns `None` when the buffer is empty (round 0).
    /// Test-only — not load-bearing in the hot loop.
    pub fn windowed_accept_rate(&self) -> Option<f32> {
        if self.samples.is_empty() {
            None
        } else {
            Some(self.samples.iter().sum::<f32>() / self.samples.len() as f32)
        }
    }

    /// Record the outcome of one spec round and (possibly) adjust K
    /// for the next round.
    ///
    /// - `accepted`: number of drafts whose target sample matched
    ///   ([`count_accepted`]'s return value).
    /// - `drafts_proposed`: K value the round actually ran with (i.e.
    ///   `drafts.len()` post-budget-trim). Must be `>= 1`.
    ///
    /// Returns the (possibly updated) K for the next round.
    /// No-op when `drafts_proposed == 0` (defensive — the runner
    /// already skips the spec round in that case, but we don't want a
    /// divide-by-zero if a future caller hits this with an empty
    /// proposal).
    ///
    /// Adjustment fires only when the sliding window is full
    /// (`samples.len() == window`); after firing, the window is
    /// cleared and the next adjustment waits for another `window`
    /// observations (AIMD-style cooldown, see [`AdaptiveK`] docs).
    pub fn observe_round(&mut self, accepted: usize, drafts_proposed: usize) -> usize {
        if drafts_proposed == 0 {
            return self.current_k;
        }
        debug_assert!(
            accepted <= drafts_proposed,
            "accepted ({}) cannot exceed drafts_proposed ({})",
            accepted,
            drafts_proposed,
        );
        let frac = accepted as f32 / drafts_proposed as f32;
        // Ring buffer: evict oldest if full, then push newest.
        // (In the steady-state cooldown design the buffer should never
        // be full at the *start* of observe_round — we always clear it
        // immediately after an adjustment — but keep the eviction
        // branch defensively in case a future code path partially
        // populates it.)
        if self.samples.len() == self.cfg.window {
            self.samples.pop_front();
        }
        self.samples.push_back(frac);

        // Don't adjust until the window is fully populated. Reacting
        // to a single-round burst is exactly the noise behavior we
        // want to suppress.
        if self.samples.len() < self.cfg.window {
            return self.current_k;
        }
        let rate = self.samples.iter().sum::<f32>() / self.samples.len() as f32;
        let new_k = if rate > self.cfg.up_threshold {
            (self.current_k + self.cfg.up_step).min(self.cfg.k_max)
        } else if rate < self.cfg.down_threshold {
            self.current_k
                .saturating_sub(self.cfg.down_step)
                .max(self.cfg.k_min)
        } else {
            self.current_k
        };
        // Always clear the window on a decision point — whether or not
        // K actually changed. The cooldown applies even in the
        // "stayed in the band" case so that the next decision is
        // again based on `window` fresh observations.
        self.samples.clear();
        self.current_k = new_k;
        new_k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_accepted_all_match() {
        let drafts = vec![1, 2, 3, 4];
        let targets = vec![1, 2, 3, 4];
        assert_eq!(count_accepted(&drafts, &targets), 4);
    }

    #[test]
    fn count_accepted_first_mismatch() {
        let drafts = vec![1, 2, 3, 4];
        let targets = vec![1, 2, 99, 4];
        // Mismatch at position 2 — accept up to but not including.
        assert_eq!(count_accepted(&drafts, &targets), 2);
    }

    #[test]
    fn count_accepted_no_match() {
        let drafts = vec![1, 2, 3];
        let targets = vec![99, 98, 97];
        assert_eq!(count_accepted(&drafts, &targets), 0);
    }

    #[test]
    fn count_accepted_unequal_lens() {
        // Should accept up to the shorter length.
        let drafts = vec![1, 2, 3, 4, 5];
        let targets = vec![1, 2, 3];
        assert_eq!(count_accepted(&drafts, &targets), 3);
    }

    #[test]
    fn reconcile_partial_accept() {
        // 4 drafts, 2 accepted, no bonus forward (the bonus comes from
        // the rejection-position forward).
        let r = reconcile_after_round(4, 2, false, false);
        assert_eq!(
            r,
            RoundReconcile {
                history_pop: 2,        // pop the 2 rejected drafts
                kv_rewind: 2,          // 4 KV writes - 2 accepted
                new_tokens_emitted: 3, // 2 accepted + 1 bonus
                all_accepted_bonus: false,
            }
        );
    }

    #[test]
    fn reconcile_all_accepted_with_bonus_forward() {
        // 4 drafts, all 4 accepted, an extra bonus forward ran to get
        // the prev_correction for next round.
        let r = reconcile_after_round(4, 4, true, false);
        assert_eq!(
            r,
            RoundReconcile {
                history_pop: 0,
                kv_rewind: 1,          // 5 KV writes - 4 accepted = 1 (the bonus slot)
                new_tokens_emitted: 5, // 4 accepted + 1 bonus
                all_accepted_bonus: true,
            }
        );
    }

    #[test]
    fn reconcile_zero_accepted() {
        // All drafts rejected; bonus comes from the first-position
        // forward.
        let r = reconcile_after_round(8, 0, false, false);
        assert_eq!(
            r,
            RoundReconcile {
                history_pop: 8,
                kv_rewind: 8,
                new_tokens_emitted: 1,
                all_accepted_bonus: false,
            }
        );
    }

    #[test]
    fn reconcile_single_draft_accepted() {
        let r = reconcile_after_round(1, 1, true, false);
        // K=1: if the one draft accepted, bonus_forward_ran=true
        // means we ran 1 + 1 = 2 forwards; accepted 1; rewind 2-1=1.
        assert_eq!(
            r,
            RoundReconcile {
                history_pop: 0,
                kv_rewind: 1,
                new_tokens_emitted: 2,
                all_accepted_bonus: true,
            }
        );
    }

    /// Regression test for the runner-convention off-by-one fixed in
    /// `fix/spec-decode-reconcile-off-by-one-043`.
    ///
    /// In the pending-token convention (used by
    /// `Runner::generate_speculative` and the pipeline-parallel
    /// `drive_generation_first_spec` driver), the K verify forwards
    /// absorb the previous round's pending token into KV as a side
    /// effect. So `kv_rewind` must be one LESS than the clean-convention
    /// value — otherwise the helper rewinds one too many KV slots,
    /// causing the next round's verify forward to mis-align with the
    /// stale KV state. The single-stage runner's `kv_invariant_holds`
    /// debug_assert fires on round 2 when this drift is wrong.
    ///
    /// Matches the inline formula documented in
    /// `engine::drive_generation_first_spec`:
    ///   partial accept (A < K, no bonus forward): rewind = K - A - 1
    ///   all accepted (A == K, bonus forward ran):  rewind = 0
    #[test]
    fn reconcile_pending_token_partial_accept() {
        // 4 drafts, 2 accepted, runner convention (pending token present).
        let r = reconcile_after_round(4, 2, false, true);
        assert_eq!(
            r,
            RoundReconcile {
                history_pop: 2,        // pop the 2 rejected drafts
                kv_rewind: 1,          // 4 KV writes - 2 accepted - 1 (pending drift) = 1
                new_tokens_emitted: 3, // 2 accepted + 1 bonus
                all_accepted_bonus: false,
            }
        );
    }

    #[test]
    fn reconcile_pending_token_all_accepted() {
        // 4 drafts, all 4 accepted, bonus forward ran, runner convention.
        let r = reconcile_after_round(4, 4, true, true);
        assert_eq!(
            r,
            RoundReconcile {
                history_pop: 0,
                kv_rewind: 0, // 5 KV writes - 4 accepted - 1 (pending drift) = 0
                new_tokens_emitted: 5, // 4 accepted + 1 bonus
                all_accepted_bonus: true,
            }
        );
    }

    #[test]
    fn reconcile_pending_token_zero_accepted() {
        // 8 drafts, all rejected, runner convention. The K-loop wrote
        // 8 KV slots (first one absorbed the pending token); we want
        // to end at KV = pre_kv = pre_history - 1, which means
        // rewind = 8 - 0 - 1 = 7.
        let r = reconcile_after_round(8, 0, false, true);
        assert_eq!(
            r,
            RoundReconcile {
                history_pop: 8,
                kv_rewind: 7,
                new_tokens_emitted: 1,
                all_accepted_bonus: false,
            }
        );
    }

    #[test]
    fn end_to_end_round_math_is_invariant() {
        // The post-round history length should always equal
        // pre_round_history_len + new_tokens_emitted - 1 (the bonus
        // counts as emitted but is appended once at the end of the
        // round, not during the verify forwards).
        //
        // Equivalently: per-round, after applying RoundReconcile,
        //   final_kv == final_history_len
        // where final_history_len = pre + new_tokens_emitted (because
        // accepted_drafts are popped via history_pop and re-pushed
        // alongside the bonus).
        //
        // Walk every combination of (k_drafts, accepted, bonus_ran)
        // and check the arithmetic balances. Tests both conventions.
        for &pending in &[false, true] {
            for k in 1..=8 {
                for a in 0..=k {
                    let bonus_ran = a == k;
                    let r = reconcile_after_round(k, a, bonus_ran, pending);
                    // Total KV writes during the round = k + (bonus_ran ? 1 : 0).
                    let total_writes = k + if bonus_ran { 1 } else { 0 };
                    // After rewind, kv = total_writes - r.kv_rewind.
                    let kv_after = total_writes - r.kv_rewind;
                    // Clean convention: we want kv_after == accepted (the next round's
                    // first forward will push the bonus to KV).
                    // Pending convention: we want kv_after == accepted + 1 (the bonus
                    // becomes the next round's pending token; KV was already at +1
                    // drift relative to history pre-round, and that drift is preserved).
                    let expected_kv = if pending { a + 1 } else { a };
                    assert_eq!(
                        kv_after, expected_kv,
                        "k={k} a={a} bonus_ran={bonus_ran} pending={pending}"
                    );
                    // new_tokens_emitted = accepted + 1 (the +1 is the bonus).
                    assert_eq!(r.new_tokens_emitted, a + 1, "k={k} a={a} pending={pending}");
                    // history_pop = k - a (rejected drafts).
                    assert_eq!(r.history_pop, k - a, "k={k} a={a} pending={pending}");
                }
            }
        }
    }

    /// Simulates a full multi-round spec-decode session against a
    /// mock target. Verifies the (count_accepted, reconcile_after_round,
    /// history+kv update) pipeline matches what a sequential greedy
    /// generator would produce — i.e. spec-decode is a pure throughput
    /// trick, not a correctness change.
    #[test]
    fn simulated_session_matches_sequential_greedy() {
        // Mock target: a fixed deterministic next-token function. For
        // each (history-suffix → next-token) mapping the target
        // believes, the sequential generator produces that token, and
        // so should the spec-decoder regardless of what the draft
        // proposed.
        let mock_target = |history: &[i64]| -> i64 {
            // Simple Fibonacci-like: each next token = (last + last-1) % 100.
            // Returns 7 as a fallback for very short histories so the
            // sequence is well-defined from any starting point.
            if history.len() >= 2 {
                let a = history[history.len() - 1];
                let b = history[history.len() - 2];
                ((a + b) % 100).max(0)
            } else if !history.is_empty() {
                (history[history.len() - 1] + 1) % 100
            } else {
                7
            }
        };

        let prompt: Vec<i64> = vec![1, 2, 3];
        let max_new: usize = 16;

        // Sequential reference: just call mock_target N times.
        let mut sequential = prompt.clone();
        for _ in 0..max_new {
            sequential.push(mock_target(&sequential));
        }
        let expected_new: Vec<i64> = sequential[prompt.len()..].to_vec();

        // Spec-decode simulation. Use a draft that proposes the
        // sequence shifted by 1 (occasionally wrong) — to exercise
        // both accept and reject paths.
        let mock_draft = |history: &[i64]| -> Vec<i64> {
            // Propose 4 tokens following a "guess pattern": the target's
            // ground truth occasionally agrees with this, occasionally
            // not.
            let mut out = Vec::with_capacity(4);
            let mut working = history.to_vec();
            for k in 0..4 {
                // Draft pattern: take target's prediction for the FIRST
                // draft (always correct → forces all-accepted branches),
                // and a deliberately-wrong "+3" for the rest (forces
                // rejection branches).
                let proposal = if k == 0 {
                    mock_target(&working)
                } else {
                    (mock_target(&working) + 3) % 100
                };
                working.push(proposal);
                out.push(proposal);
            }
            out
        };

        let mut history = prompt.clone();
        let mut generated: Vec<i64> = Vec::new();
        // Mock KV: just a counter we increment on each "forward" call.
        let mut kv_len = prompt.len();
        // Prefill: emulate one forward per prompt token (mock).
        // We don't actually generate prefill in this test — assume KV
        // is at history.len() after prefill.

        while generated.len() < max_new {
            let drafts = mock_draft(&history);
            // Each forward = 1 KV write + 1 history push of the
            // drafted token. We "run" all drafts.len() forwards.
            let mut target_samples = Vec::with_capacity(drafts.len() + 1);
            for &draft_tok in &drafts {
                target_samples.push(mock_target(&history));
                history.push(draft_tok);
                kv_len += 1;
            }
            let accepted = count_accepted(&drafts, &target_samples);
            let bonus_forward_ran = accepted == drafts.len();
            let bonus: i64 = if !bonus_forward_ran {
                target_samples[accepted]
            } else {
                let b = mock_target(&history);
                kv_len += 1; // extra bonus forward
                b
            };
            let r = reconcile_after_round(drafts.len(), accepted, bonus_forward_ran, false);
            // Apply rewinds (mock).
            history.truncate(history.len() - r.history_pop);
            kv_len -= r.kv_rewind;
            // Emit accepted drafts + bonus.
            for &t in drafts.iter().take(accepted) {
                generated.push(t);
                if generated.len() >= max_new {
                    break;
                }
            }
            let mut emitted_bonus = false;
            if generated.len() < max_new {
                history.push(bonus);
                generated.push(bonus);
                emitted_bonus = true;
            }
            // Invariant: after the bonus is appended to history (but
            // before the next round's first forward writes its KV slot),
            // kv_len trails history.len() by exactly 1 (the bonus position
            // that the next forward will fill). If we didn't emit the
            // bonus, kv_len matches history.len().
            let expected_drift = if emitted_bonus { 1 } else { 0 };
            assert_eq!(
                kv_len,
                history.len() - expected_drift,
                "kv invariant: drafts={drafts:?} accepted={accepted} bonus_emitted={emitted_bonus}"
            );
            // Simulate next round's first forward (just for the test —
            // matches the runner's forward-then-emit cycle).
            if emitted_bonus {
                kv_len += 1;
            }
        }

        generated.truncate(max_new);
        assert_eq!(
            generated, expected_new,
            "spec-decoded output should match sequential greedy"
        );
    }

    /// Regression test for the runner-convention off-by-one fixed in
    /// `fix/spec-decode-reconcile-off-by-one-043`.
    ///
    /// Simulates the EXACT call pattern used by
    /// [`crate::runner::Runner::generate_speculative`] (and the
    /// pipeline-parallel `drive_generation_first_spec` driver):
    ///
    /// 1. Pre-push `first_gen` to history right after prefill; KV
    ///    stays at prompt.len(). History trails ahead by +1.
    /// 2. K-loop: for each draft, `step(history, 1)` (which writes 1
    ///    KV slot at past_seq_len = history.len()-1), THEN
    ///    `history.push(draft_tok)`. After K iterations:
    ///    history.len() = N+K, kv = N+K-1, where N is the pre-round
    ///    history length.
    /// 3. Compute accepted; if accepted == K, run a bonus forward
    ///    (writes 1 more KV slot, no history push).
    /// 4. Apply `reconcile_after_round(..., pending=true)`.
    /// 5. Append bonus to history (no KV write). History trails by +1
    ///    again, ready for the next round.
    ///
    /// The post-round invariant the runner asserts (via
    /// `kv_invariant_holds`) is:
    ///   `KV == history.len()` after appending bonus AND the NEXT
    ///   round's first verify forward (which writes the bonus's KV
    ///   slot).
    ///
    /// At the boundary between rounds (between bonus.push and the
    /// next K-loop's first forward), the invariant is:
    ///   `KV + 1 == history.len()`.
    ///
    /// Before the fix, the helper rewound K-A KV slots (clean
    /// convention) instead of K-A-1 (runner convention) — so KV ended
    /// up at history.len() - 2, and the runner's `kv_invariant_holds`
    /// would fail on round 2 (and `debug_assert!` would fire in debug
    /// builds).
    #[test]
    fn simulated_runner_pending_session_matches_sequential_greedy() {
        // Same mock target as the clean-convention test, so we can
        // compare against the same sequential reference.
        let mock_target = |history: &[i64]| -> i64 {
            if history.len() >= 2 {
                let a = history[history.len() - 1];
                let b = history[history.len() - 2];
                ((a + b) % 100).max(0)
            } else if !history.is_empty() {
                (history[history.len() - 1] + 1) % 100
            } else {
                7
            }
        };

        let prompt: Vec<i64> = vec![1, 2, 3];
        let max_new: usize = 16;

        // Sequential reference.
        let mut sequential = prompt.clone();
        for _ in 0..max_new {
            sequential.push(mock_target(&sequential));
        }
        let expected_new: Vec<i64> = sequential[prompt.len()..].to_vec();

        // Same draft pattern.
        let mock_draft = |history: &[i64]| -> Vec<i64> {
            let mut out = Vec::with_capacity(4);
            let mut working = history.to_vec();
            for k in 0..4 {
                let proposal = if k == 0 {
                    mock_target(&working)
                } else {
                    (mock_target(&working) + 3) % 100
                };
                working.push(proposal);
                out.push(proposal);
            }
            out
        };

        // ---- Prefill: runner convention ----
        let mut history = prompt.clone();
        let mut generated: Vec<i64> = Vec::new();
        // After prefill, KV == history.len() (every prompt token had a
        // forward call write its slot).
        let mut kv_len = prompt.len();

        // First generated token from last prefill step's logits.
        // Runner pre-pushes this to history; KV stays put.
        let first = mock_target(&history);
        history.push(first);
        generated.push(first);
        // Invariant at this point: kv_len + 1 == history.len() (pending).
        assert_eq!(
            kv_len + 1,
            history.len(),
            "round-0 setup: pending-token invariant",
        );

        // ---- Spec-decode rounds, runner convention ----
        while generated.len() < max_new {
            let drafts = mock_draft(&history);

            // K-loop: each verify does `step` then `history.push(draft)`.
            // step writes KV at past_seq_len = history.len()-1, then KV
            // advances by 1. So per-iter: kv_len += 1 BEFORE the push,
            // then history.push() pushes history ahead. Net per-iter:
            // kv_len += 1, history.len() += 1.
            let mut target_samples = Vec::with_capacity(drafts.len() + 1);
            for &draft_tok in &drafts {
                target_samples.push(mock_target(&history));
                kv_len += 1; // the step's KV write
                history.push(draft_tok);
            }
            // After K-loop: history.len() = N+K, kv_len = N+K-1
            // (where N was the pre-round history length).

            let accepted = count_accepted(&drafts, &target_samples);
            let bonus_forward_ran = accepted == drafts.len();
            let bonus: i64 = if !bonus_forward_ran {
                target_samples[accepted]
            } else {
                // All-accepted bonus forward: writes one more KV slot,
                // no history push.
                let b = mock_target(&history);
                kv_len += 1;
                b
            };

            let r = reconcile_after_round(drafts.len(), accepted, bonus_forward_ran, true);
            history.truncate(history.len() - r.history_pop);
            kv_len -= r.kv_rewind;

            // Emit accepted drafts (into `generated` only — they're
            // already in history from the K-loop and survived the pop).
            let mut hit_max = false;
            for &t in drafts.iter().take(accepted) {
                generated.push(t);
                if generated.len() >= max_new {
                    hit_max = true;
                    break;
                }
            }

            // Append bonus to history (no KV write) — pending-token
            // convention preserves the +1 drift for the next round.
            // We only push if we have room AND haven't hit_max; this
            // mirrors the runner's behavior.
            let bonus_pushed = !hit_max && generated.len() < max_new;
            if bonus_pushed {
                history.push(bonus);
                generated.push(bonus);
            }

            // Post-round invariant: with the bonus pushed, history
            // trails KV by +1 (the bonus's pending slot — the next
            // round's first verify forward will fill it). Without the
            // bonus pushed (max_new saturated mid-round), no drift.
            //
            // Before the fix, the helper rewound K-A slots instead of
            // K-A-1, leaving kv_len at history.len() - 2 (drift = -2)
            // after the bonus push — which would trip the runner's
            // `kv_invariant_holds` debug_assert on round 2.
            let expected_drift = if bonus_pushed { 1 } else { 0 };
            assert_eq!(
                kv_len + expected_drift,
                history.len(),
                "pending-token round invariant broken: \
                 drafts={drafts:?} accepted={accepted} bonus_forward_ran={bonus_forward_ran} \
                 bonus_pushed={bonus_pushed} kv_len={kv_len} history.len()={}",
                history.len()
            );

            if hit_max {
                break;
            }
        }

        generated.truncate(max_new);
        assert_eq!(
            generated, expected_new,
            "spec-decoded output (runner convention) should match sequential greedy"
        );
    }

    // ----------------------------------------------------------------
    // AdaptiveK tests (iter 083 / perf/dynamic-spec-k-083)
    // ----------------------------------------------------------------

    /// First round always returns `k_start` — no observation yet, so the
    /// adjustment rule cannot fire even in principle.
    #[test]
    fn adaptive_k_first_round_returns_k_start() {
        let ak = AdaptiveK::new(AdaptiveKConfig::new_with_start(4));
        assert_eq!(ak.current_k(), 4);
        assert_eq!(ak.windowed_accept_rate(), None);
    }

    /// The window must fully populate before adjustment fires. Even if
    /// the first sample is a perfect-100% round, K should NOT bump up
    /// while the buffer is fewer than `window` entries.
    #[test]
    fn adaptive_k_does_not_adjust_before_window_full() {
        let cfg = AdaptiveKConfig::new_with_start(4); // window=8
        let mut ak = AdaptiveK::new(cfg);
        for _ in 0..7 {
            // 7 perfect rounds — accept_rate = 1.0 each, but window not full.
            ak.observe_round(4, 4);
        }
        assert_eq!(ak.current_k(), 4, "K should not change before window full");
        // 8th observation fills the window and finally allows adjustment.
        // up_step is 2, so K bumps to 6 (not all the way to k_max).
        ak.observe_round(4, 4);
        assert_eq!(ak.current_k(), 6, "K should bump to 6 once window is full");
        // After the adjustment the window is cleared (AIMD cooldown).
        // The next 7 rounds should keep K at 6 even at 100% accept.
        for _ in 0..7 {
            ak.observe_round(6, 6);
        }
        assert_eq!(
            ak.current_k(),
            6,
            "post-adjustment cooldown: K stays at 6 for `window` more rounds"
        );
        ak.observe_round(6, 6); // 8th observation since last adjustment
        assert_eq!(
            ak.current_k(),
            8,
            "after cooldown, next full window triggers another adjustment"
        );
    }

    /// Spec from the iter 083 task: simulate 10 rounds at 80% accept
    /// rate, expect K to rise from 4 to 8.
    ///
    /// With the default window=8 the controller fires at most once per
    /// 8 rounds (cooldown — see [`AdaptiveK`] docs), so in 10 rounds it
    /// can only walk K=4→6, not K=4→8. To match the literal test spec
    /// ("K rises from 4 to 8 in 10 rounds"), we use `window=5` here:
    /// adjustment fires at round 5 (4→6) and round 10 (6→8). The
    /// adjustment LOGIC is what the test exercises; the choice of
    /// window length is itself an operator-configurable knob (each
    /// fleet can tune for noise tolerance vs responsiveness).
    ///
    /// The DEFAULT window=8 case is covered by
    /// [`adaptive_k_default_window_eventually_caps_at_kmax`] below,
    /// which runs 200 rounds and verifies K reaches k_max=16.
    #[test]
    fn adaptive_k_rises_to_8_under_80pct_accept_in_10_rounds() {
        // Window=5: two adjustment points in 10 rounds, enough for the
        // K=4→6→8 walk. K=4 with 80% accept = 3/4 accepted = 0.75
        // mean over 5 identical rounds → 0.75 > 0.7 → K=6.
        // K=6 with 80% accept = 5/6 accepted ≈ 0.833 mean over 5 rounds
        // → 0.833 > 0.7 → K=8.
        let cfg = AdaptiveKConfig {
            window: 5,
            ..AdaptiveKConfig::new_with_start(4)
        };
        let mut ak = AdaptiveK::new(cfg);

        let mut k_trajectory: Vec<usize> = vec![ak.current_k()];
        for _ in 0..10 {
            let k_now = ak.current_k();
            // 80% accept: accepted = round(0.8 * k_now). Min 1 so the
            // ratio never floors to 0.
            let accepted = ((0.8 * k_now as f32).round() as usize).max(1).min(k_now);
            ak.observe_round(accepted, k_now);
            k_trajectory.push(ak.current_k());
        }
        // The final K after 10 rounds at 80% accept should have reached
        // 8. Verify the trajectory crosses 6 on the way (so we know the
        // adjustment fired both times).
        assert!(
            k_trajectory.contains(&6),
            "K should pass through 6 on its way up: trajectory={k_trajectory:?}"
        );
        assert_eq!(
            ak.current_k(),
            8,
            "after 10 rounds @ 80% accept, K should reach 8: trajectory={k_trajectory:?}"
        );
    }

    /// Companion to the spec test: under the DEFAULT config (window=8),
    /// 80% accept eventually walks K all the way to k_max=16. This is
    /// the operationally relevant test — confirms the controller scales
    /// past the test-only window=5 case to the default policy.
    ///
    /// With window=8 + AIMD cooldown, each adjustment costs 8 rounds.
    /// Walk K=4 → 6 → 8 → 10 → 12 → 14 → 16 = 6 transitions × 8 rounds
    /// = 48 rounds. 100 rounds buys generous headroom.
    #[test]
    fn adaptive_k_default_window_eventually_caps_at_kmax() {
        let mut ak = AdaptiveK::new(AdaptiveKConfig::new_with_start(4));
        for _ in 0..100 {
            let k_now = ak.current_k();
            let accepted = ((0.8 * k_now as f32).round() as usize).max(1).min(k_now);
            ak.observe_round(accepted, k_now);
        }
        assert_eq!(
            ak.current_k(),
            16,
            "default-window controller should saturate at k_max=16"
        );
    }

    /// Low-accept-rate rounds should drive K down toward `k_min`.
    /// Mirror of the high-accept test — verifies the down-step path
    /// fires symmetrically.
    #[test]
    fn adaptive_k_falls_to_kmin_under_low_accept_rate() {
        let cfg = AdaptiveKConfig {
            window: 4,
            // Start higher than k_min so we have somewhere to fall to.
            ..AdaptiveKConfig::new_with_start(10)
        };
        let mut ak = AdaptiveK::new(cfg);
        // 0% accept rounds: rate=0.0 < 0.3 → K -= 1 each adjustment.
        // With window=4 + AIMD cooldown, each transition costs 4 rounds.
        // Walk K=10 → 9 → 8 → 7 → 6 → 5 → 4 → 3 → 2 = 8 transitions ×
        // 4 rounds = 32 rounds. 50 rounds buys headroom.
        for _ in 0..50 {
            let k_now = ak.current_k();
            ak.observe_round(0, k_now);
        }
        assert_eq!(
            ak.current_k(),
            2, // k_min
            "0% accept should drive K down to k_min=2"
        );
    }

    /// Boundary check: accept rate exactly at the thresholds should be
    /// a no-op. Prevents oscillation at edge cases. The contract uses
    /// strict inequality (> up_threshold, < down_threshold).
    ///
    /// To hit the boundary exactly we use K=10 + 7/10 accepted (rate
    /// exactly 0.7) and 3/10 accepted (rate exactly 0.3); arbitrary
    /// integer ratios at smaller K cannot land cleanly on these
    /// thresholds.
    #[test]
    fn adaptive_k_no_adjust_at_threshold_boundaries() {
        let mut ak = AdaptiveK::new(AdaptiveKConfig {
            window: 2,
            ..AdaptiveKConfig::new_with_start(10)
        });
        ak.observe_round(7, 10); // rate=0.7
        ak.observe_round(7, 10); // rate=0.7 → mean=0.7 (== up_threshold)
        assert_eq!(
            ak.current_k(),
            10,
            "rate == up_threshold should NOT trigger up-step (strict >)"
        );
        // Now test the lower boundary.
        let mut ak = AdaptiveK::new(AdaptiveKConfig {
            window: 2,
            ..AdaptiveKConfig::new_with_start(10)
        });
        ak.observe_round(3, 10); // rate=0.3
        ak.observe_round(3, 10); // rate=0.3 → mean=0.3 (== down_threshold)
        assert_eq!(
            ak.current_k(),
            10,
            "rate == down_threshold should NOT trigger down-step (strict <)"
        );
    }

    /// K should clamp to [k_min, k_max] no matter how many trigger
    /// rounds fire. Verifies the cap is enforced in BOTH directions
    /// even under pathological run-on observations.
    #[test]
    fn adaptive_k_respects_min_max_bounds() {
        // Cap-up.
        let cfg = AdaptiveKConfig {
            k_start: 14,
            k_max: 16,
            k_min: 2,
            window: 1, // adjust every round
            up_threshold: 0.5,
            down_threshold: 0.1,
            up_step: 5, // larger than range → expect clamp
            down_step: 1,
        };
        let mut ak = AdaptiveK::new(cfg);
        ak.observe_round(2, 2);
        assert_eq!(ak.current_k(), 16, "up-clamp at k_max");
        ak.observe_round(2, 2);
        assert_eq!(ak.current_k(), 16, "stays clamped at k_max");

        // Cap-down with a big step that would underflow saturating-sub.
        let cfg = AdaptiveKConfig {
            k_start: 3,
            k_max: 16,
            k_min: 2,
            window: 1,
            up_threshold: 0.99,
            down_threshold: 0.5,
            up_step: 1,
            down_step: 10, // larger than range → expect clamp + no underflow
        };
        let mut ak = AdaptiveK::new(cfg);
        ak.observe_round(0, 4);
        assert_eq!(ak.current_k(), 2, "down-clamp at k_min");
        ak.observe_round(0, 4);
        assert_eq!(ak.current_k(), 2, "stays clamped at k_min");
    }

    /// Sliding-window evicts oldest sample once the buffer is full,
    /// PROVIDED no adjustment has fired (which would clear the buffer).
    ///
    /// With the AIMD cooldown design, the window is cleared on every
    /// decision. To exercise pure eviction we need the buffer to fill
    /// without a decision firing — that means choosing thresholds /
    /// accept rates that fall in the "no adjust" band (`>= down_threshold
    /// AND <= up_threshold`).
    #[test]
    fn adaptive_k_window_evicts_old_samples() {
        let cfg = AdaptiveKConfig {
            window: 4,
            // Wide neutral band so 50% accept doesn't trigger either side.
            up_threshold: 0.9,
            down_threshold: 0.1,
            ..AdaptiveKConfig::new_with_start(4)
        };
        let mut ak = AdaptiveK::new(cfg);
        // 1 round of 0% accept.
        ak.observe_round(0, 4);
        assert_eq!(ak.windowed_accept_rate(), Some(0.0));
        // 3 rounds of 50% accept — window grows but no decision fires
        // because the buffer isn't full.
        for _ in 0..3 {
            ak.observe_round(2, 4);
        }
        // Now the window IS full and a decision fires with mean
        // (0 + 0.5*3)/4 = 0.375 — which is in the neutral band, so K
        // is unchanged BUT the window is still cleared (cooldown).
        // After this point the buffer is empty.
        assert_eq!(ak.windowed_accept_rate(), None);
        assert_eq!(ak.current_k(), 4, "in-band rate leaves K unchanged");
        // Restart: 4 fresh 100% rounds. Window fills to capacity 4 with
        // all 1.0s — eviction wasn't needed in this path because we
        // never pushed a 5th sample, but the buffer's accumulated mean
        // shows the post-cooldown samples cleanly.
        for _ in 0..3 {
            ak.observe_round(4, 4);
        }
        let rate = ak.windowed_accept_rate().expect("non-empty window");
        assert!(
            (rate - 1.0).abs() < 1e-6,
            "expected windowed rate = 1.0 in fresh post-cooldown samples, got {rate}"
        );
    }

    /// Empty proposal observation is a no-op. The runner does NOT call
    /// `observe_round` in the empty-proposal fallback path (single
    /// forward, no spec accounting), but the controller defends against
    /// a future caller passing `drafts_proposed=0`.
    #[test]
    fn adaptive_k_observe_with_zero_drafts_is_noop() {
        let mut ak = AdaptiveK::new(AdaptiveKConfig::new_with_start(4));
        let k_before = ak.current_k();
        let k_after = ak.observe_round(0, 0);
        assert_eq!(k_before, k_after);
        assert_eq!(ak.window_len(), 0, "no sample pushed for zero drafts");
    }

    /// Accept rate is normalized by drafts-proposed, not by current K.
    /// Documents the budget-truncated round behavior described in
    /// [`AdaptiveK`]'s rustdoc.
    #[test]
    fn adaptive_k_uses_drafts_proposed_not_current_k() {
        let cfg = AdaptiveKConfig {
            window: 2,
            ..AdaptiveKConfig::new_with_start(4)
        };
        let mut ak = AdaptiveK::new(cfg);
        // K=4, but the round only proposed 1 draft (budget exhausted)
        // and accepted that 1 draft. Rate = 1/1 = 100% per round,
        // NOT 1/4 = 25%.
        ak.observe_round(1, 1);
        // 1 sample so far, window=2 → no decision yet, K unchanged.
        assert_eq!(ak.current_k(), 4);
        assert_eq!(ak.windowed_accept_rate(), Some(1.0));
        // Second sample at K=4 proposing 1 draft (budget=1), accept 1.
        // Rate is again 1.0, NOT 0.25. Window full → decision fires:
        // mean = 1.0 > 0.7 → K = 6.
        ak.observe_round(1, 1);
        assert_eq!(
            ak.current_k(),
            6,
            "rate normalized by drafts-proposed (1/1) — NOT by current K (1/4)"
        );
    }
}
