//! Pure helpers for the qwen35 staged engine's greedy n-gram (prompt-lookup) speculative
//! decoding and its `CASCADIA_QWEN36_PROFILE=1` instrumentation. No OpenVINO here: everything
//! in this module is testable on a stub build, and `qwen36.rs` only wires it to the stage chain.
//!
//! The algorithm (standard greedy verification):
//!
//! 1. Draft: the last `ngram` tokens of the history (prompt + every token fed so far, including
//!    the pending token) are looked up backwards in that history; the up-to-`k` tokens that
//!    followed the most recent earlier occurrence are the draft `d_1..d_k`. No match ⇒ a plain
//!    single-token step, no speculation cost.
//! 2. Verify: `[last_token, d_1..d_k]` runs as ONE `T = k+1` span at logical positions
//!    `[p, p+k+1)`. The last stage returns logits for every position; `o_i = argmax(row i)`.
//!    `j` = number of leading `i` with `o_i == d_{i+1}`; the step emits `last_token, d_1..d_j`
//!    (`j+1` tokens) and the next pending token is `o_j`.
//! 3. Rollback (`j < k`): the span consumed `k+1` positions but only `j+1` are valid. The
//!    attention KV of the whole span is masked out of every later pass ([`CacheMask`]), the
//!    Gated DeltaNet `conv`/`ssm` recurrent states — fixed-shape, not maskable — are restored
//!    from a snapshot taken right before the verify pass, and the accepted `j+1` tokens are
//!    re-fed as a `T = j+1` pass whose logits are discarded. Logical positions continue from
//!    the accepted count; the cache holds `k+1` dead slots per rollback.

use std::collections::BTreeMap;

/// `|`-separated name filter selecting the Gated DeltaNet recurrent states
/// (`cache_params.past.conv.N` / `cache_params.past.ssm.N`; `get_name()` fuses the
/// `present` alias onto the end, hence substring matching) for `Runtime::get_state_vars`.
pub const DELTANET_STATE_FILTER: &str = "past.conv.|past.ssm.";

/// Largest draft the engine will verify in one span (`PREFILL_CHUNK` bounds the transient
/// `[1, T, vocab]` logits buffer at 256 rows; a draft past 64 has negative ROI anyway).
pub const MAX_SPEC_K: usize = 64;

/// Propose up to `k` tokens: find the most recent EARLIER occurrence of the trailing `ngram`
/// tokens of `history` and return the tokens that followed it (fewer than `k` when the match
/// runs into the end of the history). Empty when `ngram == 0`, `k == 0`, the history is
/// shorter than `ngram + 1`, or nothing matches.
pub fn draft_ngram(history: &[u32], ngram: usize, k: usize) -> Vec<u32> {
    if ngram == 0 || k == 0 || history.len() < ngram + 1 {
        return Vec::new();
    }
    let tail = &history[history.len() - ngram..];
    // Candidate start positions, most recent first; `history.len() - ngram` is the tail itself.
    for s in (0..history.len() - ngram).rev() {
        if &history[s..s + ngram] == tail {
            let start = s + ngram;
            let end = (start + k).min(history.len());
            if end > start {
                return history[start..end].to_vec();
            }
        }
    }
    Vec::new()
}

/// Number of leading draft tokens the verifier agreed with: `j` such that
/// `outs[i] == draft[i]` for all `i < j`.
pub fn count_accepted(draft: &[u32], outs: &[u32]) -> usize {
    draft
        .iter()
        .zip(outs.iter())
        .take_while(|(d, o)| d == o)
        .count()
}

/// Result of verifying one span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecOutcome {
    /// Positions of the span that are valid: `j + 1` (the pending token plus `j` accepted
    /// drafts). Always `>= 1` and `<= draft.len() + 1`.
    pub accepted: usize,
    /// Every draft token was accepted: nothing to roll back.
    pub full: bool,
    /// The next pending token: `o_j`, the verifier's own prediction after the last accepted
    /// position.
    pub next: u32,
}

/// Combine the verifier's per-position argmaxes (`outs.len() == draft.len() + 1`) with the
/// draft. `None` when the verifier returned the wrong number of rows.
pub fn spec_outcome(draft: &[u32], outs: &[u32]) -> Option<SpecOutcome> {
    if outs.len() != draft.len() + 1 {
        return None;
    }
    let j = count_accepted(draft, outs);
    Some(SpecOutcome {
        accepted: j + 1,
        full: j == draft.len(),
        next: outs[j],
    })
}

/// Draft length worth verifying this step: `k`, capped at [`MAX_SPEC_K`] and at the tokens the
/// budget can still emit AFTER the pending token (`max_tokens - gen_len - 1`), and at the
/// positions left in the context window (`max_ctx - pos - 1`). 0 ⇒ do not speculate.
pub fn effective_k(
    k: usize,
    gen_len: usize,
    max_tokens: usize,
    pos: usize,
    max_ctx: Option<usize>,
) -> usize {
    let budget = max_tokens.saturating_sub(gen_len + 1);
    let window = max_ctx
        .map(|m| m.saturating_sub(pos + 1))
        .unwrap_or(usize::MAX);
    k.min(MAX_SPEC_K).min(budget).min(window)
}

/// Validity mask over the attention-KV cache slots of the current task.
///
/// Clean (no rollback yet this task) the cache depth equals the logical position and the mask
/// is all ones — byte-identical to what the engine fed before speculation existed, including
/// after a prefix-cache / plane warm restore (whose depth this type never learns; it does not
/// need to). The first rollback materialises the slot vector from the logical position the
/// span started at; from then on the mask is `slots ++ ones(n)` and every pass appends `n`
/// valid slots. Every rank keeps its own copy and applies the same operations in lockstep.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheMask {
    /// One entry per cached slot (1 valid / 0 rejected); empty while clean.
    slots: Vec<i64>,
}

impl CacheMask {
    pub fn new() -> Self {
        Self::default()
    }

    /// Back to the clean state (every `reset_state` / `recreate_request` of the stages).
    pub fn reset(&mut self) {
        self.slots.clear();
    }

    pub fn is_clean(&self) -> bool {
        self.slots.is_empty()
    }

    /// Cached slots, given the logical position (clean ⇒ the two are equal).
    pub fn cache_len(&self, logical: usize) -> usize {
        if self.is_clean() {
            logical
        } else {
            self.slots.len()
        }
    }

    /// Dead (rejected) slots in the cache.
    pub fn rejected(&self) -> usize {
        self.slots.iter().filter(|&&v| v == 0).count()
    }

    /// `attention_mask` for a pass of `n` new tokens at logical position `logical`:
    /// length `cache_len + n`, zeros at rejected slots, ones for the new tokens.
    pub fn mask_for(&self, logical: usize, n: usize) -> Vec<i64> {
        if self.is_clean() {
            return vec![1i64; logical + n];
        }
        let mut m = Vec::with_capacity(self.slots.len() + n);
        m.extend_from_slice(&self.slots);
        m.extend(std::iter::repeat(1i64).take(n));
        m
    }

    /// Record that a pass of `n` tokens just wrote `n` valid slots (no-op while clean: the
    /// logical position already tracks the depth).
    pub fn append(&mut self, n: usize) {
        if !self.is_clean() {
            self.slots.extend(std::iter::repeat(1i64).take(n));
        }
    }

    /// Mark the `span_len` slots a verify span wrote as rejected. `logical_before` is the
    /// logical position the span started at (= the cache depth before it while clean).
    pub fn rollback(&mut self, logical_before: usize, span_len: usize) {
        if self.is_clean() {
            self.slots = vec![1i64; logical_before + span_len];
        }
        let len = self.slots.len();
        let lo = len.saturating_sub(span_len);
        for v in &mut self.slots[lo..] {
            *v = 0;
        }
    }
}

/// Per-task speculative-decode counters, logged on "qwen36 task done" and folded into the
/// `cascadia_spec_*` metric families at finalize.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpecStats {
    pub steps: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub full_accepts: u64,
    pub rollbacks: u64,
    pub snapshot_us: u64,
    pub restore_us: u64,
    pub verify_us: u64,
    pub refeed_us: u64,
    /// Set when speculation was switched off mid-task (snapshot unavailable, say).
    pub disabled: Option<String>,
}

impl SpecStats {
    pub fn record(&mut self, drafted: usize, accepted_drafts: usize, full: bool) {
        self.steps += 1;
        self.drafted += drafted as u64;
        self.accepted += accepted_drafts as u64;
        if full {
            self.full_accepts += 1;
        } else {
            self.rollbacks += 1;
        }
    }

    /// accepted / drafted, 0 when nothing was drafted.
    pub fn acceptance(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }
}

// ---- pipeline frame bodies ----------------------------------------------------------------

/// `FRAME_TOKENS` body: `[u32 count][i32 token]*count[u32 infer_us]`, big-endian.
pub fn encode_tokens_body(tokens: &[i32], infer_us: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + 4 * tokens.len());
    v.extend_from_slice(&(tokens.len() as u32).to_be_bytes());
    for t in tokens {
        v.extend_from_slice(&t.to_be_bytes());
    }
    v.extend_from_slice(&infer_us.to_be_bytes());
    v
}

/// Largest `count` a `FRAME_TOKENS` body may carry (`MAX_SPEC_K + 1` rows, with headroom).
pub const MAX_TOKENS_FRAME: usize = 256;

/// Parse the part of a `FRAME_TOKENS` body AFTER the count: `count` tokens and the infer time.
pub fn parse_tokens_tail(count: usize, tail: &[u8]) -> Option<(Vec<i32>, u32)> {
    if count > MAX_TOKENS_FRAME || tail.len() != 4 * count + 4 {
        return None;
    }
    let tokens = tail[..4 * count]
        .chunks_exact(4)
        .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let u = &tail[4 * count..];
    Some((tokens, u32::from_be_bytes([u[0], u[1], u[2], u[3]])))
}

/// `FRAME_COMMIT` body: `[u32 span_len]` big-endian (the accepted count rides the header `pos`).
pub fn encode_commit_body(span_len: u32) -> [u8; 4] {
    span_len.to_be_bytes()
}

// ---- profiling (`CASCADIA_QWEN36_PROFILE=1`) -------------------------------------------------

/// `p`-th percentile of an ASCENDING sample list (nearest-rank; `p` in `[0, 1]`).
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p.clamp(0.0, 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Named sample series (microseconds) for the per-step profile summaries.
#[derive(Clone, Debug, Default)]
pub struct ProfileAcc {
    samples: BTreeMap<String, Vec<f64>>,
}

impl ProfileAcc {
    pub fn record(&mut self, key: &str, us: f64) {
        match self.samples.get_mut(key) {
            Some(v) => v.push(us),
            None => {
                self.samples.insert(key.to_string(), vec![us]);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Samples recorded under `key`.
    pub fn count(&self, key: &str) -> usize {
        self.samples.get(key).map_or(0, Vec::len)
    }

    /// `key=p50/p95us` per series, space-separated, keys sorted.
    pub fn summary(&self) -> String {
        let mut parts = Vec::with_capacity(self.samples.len());
        for (k, v) in &self.samples {
            let mut s = v.clone();
            s.sort_by(f64::total_cmp);
            parts.push(format!(
                "{k}={:.0}/{:.0}us",
                percentile(&s, 0.50),
                percentile(&s, 0.95)
            ));
        }
        parts.join(" ")
    }
}

/// Summarise one `Runtime::profiling()` TSV (`node_name\tnode_type\texec_type\treal_us\tcpu_us`
/// per line): total real vs cpu time, nodes with non-zero real time ("executed"), and the top
/// `top` node types by real time — one line per row, for the `PERF_COUNT=YES` dump.
pub fn summarize_ov_profile(tsv: &str, top: usize) -> Vec<String> {
    let mut by_type: BTreeMap<&str, (u64, u64, u32)> = BTreeMap::new();
    let (mut total_real, mut total_cpu, mut executed, mut nodes) = (0u64, 0u64, 0u32, 0u32);
    for line in tsv.lines() {
        let mut it = line.split('\t');
        let (_name, ntype) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
        let _exec = it.next();
        let real = it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        let cpu = it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
        if ntype.is_empty() && real == 0 && cpu == 0 {
            continue;
        }
        nodes += 1;
        total_real += real;
        total_cpu += cpu;
        if real > 0 {
            executed += 1;
        }
        let e = by_type.entry(ntype).or_default();
        e.0 += real;
        e.1 += cpu;
        e.2 += 1;
    }
    let mut rows: Vec<_> = by_type.into_iter().collect();
    rows.sort_by_key(|(_, (real, _, _))| std::cmp::Reverse(*real));
    let mut out = vec![format!(
        "total real {total_real} us, cpu {total_cpu} us; executed {executed} of {nodes} nodes"
    )];
    for (ntype, (real, cpu, count)) in rows.iter().take(top) {
        let share = if total_real > 0 {
            100.0 * *real as f64 / total_real as f64
        } else {
            0.0
        };
        out.push(format!(
            "{ntype:<28} real {real:>9} us ({share:>5.1}%)  cpu {cpu:>9} us  x{count}"
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- drafting ----

    #[test]
    fn draft_finds_the_most_recent_earlier_occurrence() {
        // "the cat sat on the mat . the cat sat" — tail [the, cat, sat] last occurred at 0..3,
        // followed by [on, the, mat, ., ...].
        let h = [10, 20, 30, 40, 10, 50, 60, 10, 20, 30];
        assert_eq!(draft_ngram(&h, 3, 4), vec![40, 10, 50, 60]);
        // k caps the proposal.
        assert_eq!(draft_ngram(&h, 3, 2), vec![40, 10]);
        // n = 1: the tail is [30], whose only earlier occurrence is index 2 → [40, 10, 50].
        assert_eq!(draft_ngram(&h, 1, 3), vec![40, 10, 50]);
        // Most recent earlier match wins: tail [10, 20] occurred at 0 and at 7 (the tail);
        // only 0 is earlier → 30. With tail [10] the newest earlier 10 is index 4 → 50.
        assert_eq!(
            draft_ngram(&[10, 20, 30, 40, 10, 50, 60, 10], 1, 1),
            vec![50]
        );
    }

    #[test]
    fn draft_continuation_may_overlap_the_tail() {
        // [1,1,1,1]: tail [1,1] matched at s=1 → continuation [1] (end clipped to history).
        assert_eq!(draft_ngram(&[1, 1, 1, 1], 2, 5), vec![1]);
    }

    #[test]
    fn draft_is_empty_without_a_match_or_with_degenerate_params() {
        assert!(draft_ngram(&[1, 2, 3, 4, 5], 2, 4).is_empty());
        assert!(
            draft_ngram(&[1, 2], 2, 4).is_empty(),
            "needs ngram + 1 tokens"
        );
        assert!(draft_ngram(&[1, 2, 1, 2], 0, 4).is_empty());
        assert!(draft_ngram(&[1, 2, 1, 2], 2, 0).is_empty());
        assert!(draft_ngram(&[], 2, 4).is_empty());
    }

    // ---- accept / rollback arithmetic ----

    #[test]
    fn count_accepted_is_the_leading_agreement() {
        assert_eq!(count_accepted(&[1, 2, 3], &[1, 2, 3, 9]), 3);
        assert_eq!(count_accepted(&[1, 2, 3], &[1, 7, 3, 9]), 1);
        assert_eq!(count_accepted(&[1, 2, 3], &[5, 2, 3, 9]), 0);
        assert_eq!(count_accepted(&[], &[5]), 0);
    }

    #[test]
    fn outcome_full_partial_and_zero_accept() {
        // k = 3, all accepted: 4 valid positions, next = o_3.
        let full = spec_outcome(&[1, 2, 3], &[1, 2, 3, 9]).unwrap();
        assert_eq!(
            full,
            SpecOutcome {
                accepted: 4,
                full: true,
                next: 9
            }
        );
        // Rejected at i = 1: last_token + d_1 valid, next = o_1 (the verifier's correction).
        let part = spec_outcome(&[1, 2, 3], &[1, 7, 3, 9]).unwrap();
        assert_eq!(
            part,
            SpecOutcome {
                accepted: 2,
                full: false,
                next: 7
            }
        );
        // Nothing accepted: only the pending token is valid, next = o_0 — exactly a plain step.
        let zero = spec_outcome(&[1, 2, 3], &[5, 2, 3, 9]).unwrap();
        assert_eq!(
            zero,
            SpecOutcome {
                accepted: 1,
                full: false,
                next: 5
            }
        );
        assert!(spec_outcome(&[1, 2, 3], &[1, 2, 3]).is_none(), "row count");
    }

    #[test]
    fn effective_k_respects_budget_window_and_cap() {
        assert_eq!(effective_k(5, 0, 100, 10, None), 5);
        // 2 tokens left after the pending one.
        assert_eq!(effective_k(5, 97, 100, 10, None), 2);
        assert_eq!(effective_k(5, 99, 100, 10, None), 0);
        assert_eq!(effective_k(5, 100, 100, 10, None), 0);
        // Window: positions 10..16 usable, the pending token takes 10 → 5 drafts fit.
        assert_eq!(effective_k(8, 0, 100, 10, Some(16)), 5);
        assert_eq!(effective_k(8, 0, 100, 15, Some(16)), 0);
        assert_eq!(effective_k(1000, 0, 10_000, 0, None), MAX_SPEC_K);
    }

    // ---- cache mask ----

    #[test]
    fn clean_mask_is_all_ones_of_logical_plus_n() {
        let m = CacheMask::new();
        assert!(m.is_clean());
        assert_eq!(m.mask_for(7, 3), vec![1i64; 10]);
        assert_eq!(m.cache_len(7), 7);
        let mut m2 = m.clone();
        m2.append(3);
        assert!(m2.is_clean(), "append while clean stays clean");
    }

    #[test]
    fn rollback_then_refeed_masks_the_dead_span() {
        let mut m = CacheMask::new();
        // Verify span of k+1 = 4 at logical 10; cache now 14 slots; 2 accepted (j = 1).
        m.append(4);
        m.rollback(10, 4);
        assert!(!m.is_clean());
        assert_eq!(m.cache_len(10), 14);
        assert_eq!(m.rejected(), 4);
        // Re-feed of the 2 accepted tokens at logical 10: mask = 10 ones, 4 zeros, 2 ones.
        let mask = m.mask_for(10, 2);
        assert_eq!(mask.len(), 16);
        assert!(mask[..10].iter().all(|&v| v == 1));
        assert!(mask[10..14].iter().all(|&v| v == 0));
        assert!(mask[14..].iter().all(|&v| v == 1));
        m.append(2);
        assert_eq!(m.cache_len(12), 16);
        // Next plain step at logical 12 sees the same zeros.
        let mask = m.mask_for(12, 1);
        assert_eq!(mask.len(), 17);
        assert_eq!(mask[10..14], [0, 0, 0, 0]);
        assert_eq!(mask[16], 1);
        m.append(1);
        // A second rollback appends more zeros without touching the first run.
        m.append(3);
        m.rollback(13, 3);
        assert_eq!(m.rejected(), 7);
        assert_eq!(m.cache_len(13), 20);
        m.reset();
        assert!(m.is_clean());
        assert_eq!(m.mask_for(0, 1), vec![1]);
    }

    #[test]
    fn stats_record_and_acceptance() {
        let mut s = SpecStats::default();
        s.record(4, 4, true);
        s.record(4, 1, false);
        assert_eq!((s.steps, s.drafted, s.accepted), (2, 8, 5));
        assert_eq!((s.full_accepts, s.rollbacks), (1, 1));
        assert!((s.acceptance() - 0.625).abs() < 1e-9);
        assert_eq!(SpecStats::default().acceptance(), 0.0);
    }

    // ---- frame bodies ----

    #[test]
    fn tokens_body_roundtrips() {
        let body = encode_tokens_body(&[7, -1, 248319], 4321);
        assert_eq!(body.len(), 4 + 12 + 4);
        let count = u32::from_be_bytes([body[0], body[1], body[2], body[3]]) as usize;
        assert_eq!(count, 3);
        let (toks, us) = parse_tokens_tail(count, &body[4..]).unwrap();
        assert_eq!(toks, vec![7, -1, 248319]);
        assert_eq!(us, 4321);
        assert!(parse_tokens_tail(3, &body[4..body.len() - 1]).is_none());
        assert!(parse_tokens_tail(MAX_TOKENS_FRAME + 1, &[]).is_none());
        assert_eq!(encode_commit_body(6), 6u32.to_be_bytes());
    }

    // ---- profiling ----

    #[test]
    fn percentile_nearest_rank() {
        let s = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&s, 0.5), 6.0);
        assert_eq!(percentile(&s, 0.95), 10.0);
        assert_eq!(percentile(&s, 0.0), 1.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn profile_summary_lists_sorted_keys_with_p50_p95() {
        let mut acc = ProfileAcc::default();
        for i in 1..=20 {
            acc.record("s0.infer", i as f64 * 100.0);
        }
        acc.record("embed", 50.0);
        let s = acc.summary();
        assert!(s.starts_with("embed=50/50us "), "{s}");
        assert!(s.contains("s0.infer=1100/1900us"), "{s}");
        assert_eq!(acc.count("s0.infer"), 20);
        acc.clear();
        assert!(acc.is_empty());
    }

    #[test]
    fn ov_profile_summary_totals_and_top_types() {
        let tsv = "a\tMatMul\tjit\t500\t450\nb\tMatMul\tjit\t300\t280\nc\tSoftmax\tref\t100\t90\n\
                   d\tReshape\tunknown\t0\t0\n";
        let rows = summarize_ov_profile(tsv, 2);
        assert_eq!(
            rows[0],
            "total real 900 us, cpu 820 us; executed 3 of 4 nodes"
        );
        assert_eq!(rows.len(), 3, "header + top 2");
        assert!(rows[1].starts_with("MatMul"), "{}", rows[1]);
        assert!(rows[1].contains("800 us ( 88.9%)"), "{}", rows[1]);
        assert!(rows[2].starts_with("Softmax"), "{}", rows[2]);
    }
}
