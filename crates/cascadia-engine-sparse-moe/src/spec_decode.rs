//! Speculative-decode helpers: pure functions + unit tests.
//!
//! The actual spec-decode loop lives in
//! [`crate::runner::Runner::generate_speculative`] because it needs
//! access to the Runner's private KV state. This file holds the
//! pure-logic primitives — acceptance counting + KV/history rewind
//! math — extracted so we can test them without loading a 553 GB
//! model.
//!
//! Two functions:
//! - [`count_accepted`]: greedy acceptance — longest matching prefix
//!   between drafted tokens and target-sampled tokens.
//! - [`reconcile_after_round`]: computes the post-round state
//!   transitions (how many KV slots to rewind, how many tokens to
//!   keep in history) given the accept count + whether the all-accepted
//!   bonus forward was run.
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
}
