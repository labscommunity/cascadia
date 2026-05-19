//! Token sampling: temperature / top-p / repetition penalty / EOS stop.
//!
//! All sampling math runs on f32 logits. The deterministic argmax path
//! is preserved for `temperature == 0`.
//!
//! Adaptive early-stop conditions (user-supplied stop sequences,
//! degenerate-repetition detection) live alongside the sampler in
//! [`StopConditions`]. The engine's generation loop consults them after
//! every emitted token; the model's tokenizer EOS path is still primary
//! and runs first.

/// Sampling configuration for one decode call.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// 0.0 → greedy argmax (deterministic). Higher → flatter softmax.
    pub temperature: f32,
    /// Nucleus (top-p). 0.0 or 1.0 → disabled.
    pub top_p: f32,
    /// Repetition penalty α applied to logits of recently-emitted tokens.
    /// 1.0 → no penalty. Typical values: 1.05–1.3. See "CTRL: A Conditional
    /// Transformer Language Model for Controllable Generation" (Keskar et
    /// al., 2019, §4.1) — for any token in the running history, replace
    /// `logit_i` with `logit_i / α` if positive else `logit_i * α`.
    pub repetition_penalty: f32,
    /// How many of the most recent tokens to apply the repetition
    /// penalty to. Zero → all of `history`.
    pub repetition_window: usize,
    /// PRNG seed. None → use a system-entropy seed.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 0,
            seed: None,
        }
    }
}

/// Pick the next token id given raw logits + running history.
pub fn sample(logits: &[f32], history: &[i64], cfg: &SamplingConfig, rng_state: &mut u64) -> i64 {
    let mut work: Vec<f32> = logits.to_vec();

    // Repetition penalty.
    if (cfg.repetition_penalty - 1.0).abs() > f32::EPSILON {
        let start = if cfg.repetition_window == 0 {
            0
        } else {
            history.len().saturating_sub(cfg.repetition_window)
        };
        let alpha = cfg.repetition_penalty;
        for &tok in history[start..].iter() {
            let i = tok as usize;
            if i < work.len() {
                let l = work[i];
                work[i] = if l > 0.0 { l / alpha } else { l * alpha };
            }
        }
    }

    // Greedy fallback.
    if cfg.temperature <= 0.0 {
        return argmax(&work);
    }

    // Temperature.
    let inv_t = 1.0 / cfg.temperature;
    for v in work.iter_mut() {
        *v *= inv_t;
    }

    // Softmax (stable).
    let max = work.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in work.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in work.iter_mut() {
        *v /= sum;
    }

    // Top-p (nucleus). Disabled when p ∈ {0, 1}.
    if cfg.top_p > 0.0 && cfg.top_p < 1.0 {
        let mut idx: Vec<usize> = (0..work.len()).collect();
        idx.sort_by(|&a, &b| {
            work[b]
                .partial_cmp(&work[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0f32;
        let mut keep = vec![false; work.len()];
        for &i in idx.iter() {
            keep[i] = true;
            cum += work[i];
            if cum >= cfg.top_p {
                break;
            }
        }
        let mut renorm = 0.0f32;
        for (i, v) in work.iter_mut().enumerate() {
            if !keep[i] {
                *v = 0.0;
            } else {
                renorm += *v;
            }
        }
        if renorm > 0.0 {
            for v in work.iter_mut() {
                *v /= renorm;
            }
        }
    }

    // Categorical sample.
    let u = next_uniform(rng_state);
    let mut cum = 0.0f32;
    for (i, &p) in work.iter().enumerate() {
        cum += p;
        if cum >= u {
            return i as i64;
        }
    }
    // Numerical fallback if cum < u due to fp rounding.
    (work.len() - 1) as i64
}

/// xorshift64*: small, deterministic, dependency-free PRNG. Seeded from
/// the sampling config or a fresh entropy mix at engine start.
pub fn xorshift64_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x.wrapping_mul(0x2545F4914F6CDD1Du64)
}

/// Uniform [0, 1) sample. Cheap; not crypto-grade.
pub fn next_uniform(state: &mut u64) -> f32 {
    let r = xorshift64_next(state);
    ((r >> 40) as f32) / ((1u32 << 24) as f32)
}

/// Initialize the PRNG state. If `seed` is None, mix a few entropy
/// sources so successive runs differ but stay reproducible inside one
/// process.
pub fn init_rng(seed: Option<u64>) -> u64 {
    if let Some(s) = seed {
        return s.max(1);
    }
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 | ((d.as_secs() as u64) << 32))
        .unwrap_or(0xDEADBEEFCAFEBABE);
    nanos.max(1)
}

fn argmax(xs: &[f32]) -> i64 {
    let mut best = 0i64;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as i64;
        }
    }
    best
}

/// Adaptive early-stop configuration.
///
/// EOS (the model's tokenizer eos_token_id) is handled separately in
/// `Runner::generate` because it's intrinsic to the model and applies
/// even when no adaptive stop is requested. The conditions below are
/// opt-in:
///
/// - `stop`: user-supplied strings the decoded text must not end with.
/// - `stop_on_repetition`: degenerate 4-gram loop detector.
///
/// Cheap to evaluate even at the high end (a long stop list of 8
/// sequences vs the 32-char text tail = 256 byte compares per token,
/// negligible against a ~10 s/token decode).
#[derive(Clone, Debug, Default)]
pub struct StopConditions {
    pub stop: Vec<String>,
    pub stop_on_repetition: bool,
}

impl StopConditions {
    /// True if any condition is configured. The engine skips even the
    /// constant-cost setup work when this is false.
    pub fn any(&self) -> bool {
        !self.stop.is_empty() || self.stop_on_repetition
    }
}

/// True if `text` ends with any string in `stops`. Empty stop entries
/// are ignored (they would otherwise match every step and immediately
/// end generation, which is never what the caller wants).
pub fn text_ends_with_any(text: &str, stops: &[String]) -> bool {
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if text.ends_with(s.as_str()) {
            return true;
        }
    }
    false
}

/// Width of the n-gram tracked by `is_repetition_loop`. 4 catches the
/// most common modes — "Paris. Paris. Paris." (1-gram repetition is
/// caught at n=1 within a 4-gram), single-word loops, and 2-word loops
/// like "very very very" — without false-flagging legitimate code or
/// list output.
pub const REPETITION_NGRAM: usize = 4;
/// How many times the same n-gram must repeat in the trailing window
/// to count as a degenerate loop. 3 is the threshold rainier converged
/// on for the K2.6 quality eval (DISCOVERIES, 2026-04).
pub const REPETITION_THRESHOLD: usize = 3;
/// Number of trailing tokens scanned for n-gram repetition. Large
/// enough to catch loops that span a few "this is …" preludes, small
/// enough that the scan is microseconds.
pub const REPETITION_WINDOW: usize = 20;

/// Detect a degenerate n-gram repetition in the trailing
/// [`REPETITION_WINDOW`] tokens. The most recent `REPETITION_NGRAM`
/// tokens form the candidate; the function returns true if that exact
/// sequence appears at least [`REPETITION_THRESHOLD`] times within the
/// trailing window (overlapping occurrences allowed).
///
/// Conservative by design: only flags exact n-gram repeats. A real
/// monolithic ".\n\n.\n\n…" loop trips it; a "1. foo, 2. foo, 3. foo"
/// list does not.
pub fn is_repetition_loop(tokens: &[i64]) -> bool {
    if tokens.len() < REPETITION_NGRAM * REPETITION_THRESHOLD {
        return false;
    }
    let n = REPETITION_NGRAM;
    let window_start = tokens.len().saturating_sub(REPETITION_WINDOW);
    let window = &tokens[window_start..];
    if window.len() < n {
        return false;
    }
    // Candidate = the most recent n tokens of the full stream.
    let cand = &tokens[tokens.len() - n..];
    let mut hits = 0usize;
    // i is the start of a candidate-length slice inside `window`.
    for i in 0..=window.len() - n {
        if &window[i..i + n] == cand {
            hits += 1;
            if hits >= REPETITION_THRESHOLD {
                return true;
            }
        }
    }
    false
}

/// Why the engine stopped generating. Reported in info-level logs at
/// the end of each task so operators can attribute early stops to the
/// right cause.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    MaxTokens,
    StopSequence,
    Repetition,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let l = vec![0.1, 0.4, 0.2, 0.05];
        let mut s = 1;
        assert_eq!(sample(&l, &[], &SamplingConfig::default(), &mut s), 1);
    }

    #[test]
    fn temperature_does_not_panic_on_uniform() {
        let l = vec![0.0; 100];
        let mut s = 1;
        let mut cfg = SamplingConfig::default();
        cfg.temperature = 1.0;
        let r = sample(&l, &[], &cfg, &mut s);
        assert!((0..100).contains(&(r as usize)));
    }

    #[test]
    fn repetition_penalty_avoids_repeat_with_temp_zero() {
        // History contains 5 and the original argmax is 5; penalty
        // should drop logit[5] so a different token wins.
        let mut l = vec![0.0_f32; 10];
        l[5] = 10.0;
        l[3] = 2.0;
        let history = vec![5_i64];
        let mut s = 1;
        let mut cfg = SamplingConfig::default();
        cfg.repetition_penalty = 10.0;
        // After penalty: logit[5] = 10.0 / 10.0 = 1.0. logit[3] = 2.0
        // wins. Greedy → 3.
        assert_eq!(sample(&l, &history, &cfg, &mut s), 3);
    }

    #[test]
    fn top_p_does_not_pick_long_tail() {
        // logit[0] = high, others all = -inf. top_p=0.5 keeps just [0].
        let mut l = vec![f32::NEG_INFINITY; 50];
        l[0] = 10.0;
        let mut s = 1;
        let mut cfg = SamplingConfig::default();
        cfg.temperature = 1.0;
        cfg.top_p = 0.5;
        let r = sample(&l, &[], &cfg, &mut s);
        assert_eq!(r, 0);
    }

    #[test]
    fn stop_conditions_any_reflects_state() {
        let mut sc = StopConditions::default();
        assert!(!sc.any());
        sc.stop_on_repetition = true;
        assert!(sc.any());
        sc.stop_on_repetition = false;
        sc.stop.push("\n\n".into());
        assert!(sc.any());
    }

    #[test]
    fn text_ends_with_any_matches_simple_sequence() {
        let stops = vec!["Human:".to_string(), "\n\n".to_string()];
        assert!(text_ends_with_any("Reply.\n\n", &stops));
        assert!(text_ends_with_any("Hello\nHuman:", &stops));
        assert!(!text_ends_with_any("Hello", &stops));
    }

    #[test]
    fn text_ends_with_any_ignores_empty_entries() {
        // Empty stop strings would match every step. Skip them.
        let stops = vec!["".to_string()];
        assert!(!text_ends_with_any("anything", &stops));
    }

    #[test]
    fn repetition_loop_detects_quadrugram_loop() {
        // "a b c d a b c d a b c d" — same 4-gram repeated 3 times.
        let tokens: Vec<i64> = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
        assert!(is_repetition_loop(&tokens));
    }

    #[test]
    fn repetition_loop_ignores_unique_tail() {
        // 11 distinct tokens — no n-gram repeats.
        let tokens: Vec<i64> = (0..16).collect();
        assert!(!is_repetition_loop(&tokens));
    }

    #[test]
    fn repetition_loop_no_false_positive_below_threshold() {
        // Same 4-gram appears only twice — threshold is 3.
        let tokens: Vec<i64> = vec![9, 9, 9, 9, 1, 2, 3, 4, 1, 2, 3, 4];
        assert!(!is_repetition_loop(&tokens));
    }

    #[test]
    fn repetition_loop_no_false_positive_on_short_history() {
        // Below the n*threshold floor — caller doesn't have enough
        // tokens to make a confident call.
        let tokens: Vec<i64> = vec![1, 2, 3, 4, 1, 2];
        assert!(!is_repetition_loop(&tokens));
    }

    #[test]
    fn repetition_loop_detects_word_repetition() {
        // "very very very very …" — same single token, well above the
        // threshold once the 4-gram of `very` shows up 3+ times. We
        // need at least REPETITION_NGRAM * REPETITION_THRESHOLD = 12
        // tokens before the detector commits.
        let very = 42i64;
        let mut tokens: Vec<i64> = vec![1, 2];
        for _ in 0..12 {
            tokens.push(very);
        }
        assert!(is_repetition_loop(&tokens));
    }

    #[test]
    fn repetition_loop_quiet_just_below_floor() {
        // Same `very` loop but only 11 tokens total — exactly one
        // below the n*threshold floor. Should not trip.
        let very = 42i64;
        let mut tokens: Vec<i64> = vec![1, 2];
        for _ in 0..9 {
            tokens.push(very);
        }
        assert_eq!(tokens.len(), 11);
        assert!(!is_repetition_loop(&tokens));
    }
}
