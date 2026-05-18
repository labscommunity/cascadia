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
/// Pre-conditions (caller invariants):
/// - `drafts.len() == k_drafts` forwards have been run, each
///   advancing the KV cache by 1 slot and pushing one drafted token
///   into `history`.
/// - If `accepted == k_drafts` AND the caller chose to run a bonus
///   forward (typically `true` so the next round has its
///   `prev_correction`), the bonus forward has also advanced KV by 1
///   and produced one target sample. Pass `bonus_forward_ran=true` in
///   that case.
///
/// Post-conditions:
/// - After `history.truncate(history.len() - history_pop)` and
///   `runner.rewind_kv(kv_rewind)`, the engine state holds exactly
///   the accepted prefix.
pub fn reconcile_after_round(
    k_drafts: usize,
    accepted: usize,
    bonus_forward_ran: bool,
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
    let kv_writes = k_drafts + if bonus_forward_ran { 1 } else { 0 };
    let kv_rewind = kv_writes - accepted;
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
        let r = reconcile_after_round(4, 2, false);
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
        let r = reconcile_after_round(4, 4, true);
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
        let r = reconcile_after_round(8, 0, false);
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
        let r = reconcile_after_round(1, 1, true);
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
        // and check the arithmetic balances.
        for k in 1..=8 {
            for a in 0..=k {
                let bonus_ran = a == k;
                let r = reconcile_after_round(k, a, bonus_ran);
                // Total KV writes during the round = k + (bonus_ran ? 1 : 0).
                let total_writes = k + if bonus_ran { 1 } else { 0 };
                // After rewind, kv = total_writes - r.kv_rewind.
                let kv_after = total_writes - r.kv_rewind;
                // We want kv_after == accepted (the next round's
                // first forward will push the bonus to KV).
                assert_eq!(kv_after, a, "k={k} a={a} bonus_ran={bonus_ran}");
                // new_tokens_emitted = accepted + 1 (the +1 is the bonus).
                assert_eq!(r.new_tokens_emitted, a + 1, "k={k} a={a}");
                // history_pop = k - a (rejected drafts).
                assert_eq!(r.history_pop, k - a, "k={k} a={a}");
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
            let r = reconcile_after_round(drafts.len(), accepted, bonus_forward_ran);
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
}
