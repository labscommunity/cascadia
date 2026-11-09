//! Token sampling: temperature / top-p / repetition penalty / EOS stop.
//!
//! All sampling math runs on f32 logits. The deterministic argmax path
//! is preserved for `temperature == 0`.

use cascadia_engine::{EngineError, EngineResult};
use cascadia_grammar::GrammarMask;

/// Sampling configuration for one decode call.
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// 0.0 → greedy argmax (deterministic). Higher → flatter softmax.
    pub temperature: f32,
    /// Nucleus (top-p). 0.0 or 1.0 → disabled.
    pub top_p: f32,
    /// Top-k truncation. 0 → disabled. Keeps only the k highest-probability
    /// tokens before nucleus sampling.
    pub top_k: u32,
    /// OpenAI frequency penalty: subtract `frequency_penalty * count(token)`
    /// from each logit, scaled by how often the token already appeared.
    /// 0.0 → disabled.
    pub frequency_penalty: f32,
    /// OpenAI presence penalty: subtract `presence_penalty` once from any
    /// logit whose token already appeared at all. 0.0 → disabled.
    pub presence_penalty: f32,
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
            top_k: 0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            repetition_window: 0,
            seed: None,
        }
    }
}

/// Pick the next token id given raw logits + running history. `mask` (a
/// structured-output grammar) first sets every token that may not come next
/// to `-inf`, before penalties, temperature, top-k and top-p, so no later
/// step can bring a masked token back. `Err` only from the mask: nothing in
/// the row is allowed (a grammar dead end).
pub fn sample(
    logits: &[f32],
    history: &[i64],
    cfg: &SamplingConfig,
    rng_state: &mut u64,
    mask: Option<&mut GrammarMask>,
) -> EngineResult<i64> {
    let Some(mask) = mask else {
        return Ok(sample_row(logits.to_vec(), history, cfg, rng_state));
    };
    let mut masked = logits.to_vec();
    mask.apply(&mut masked)?;
    sample_masked_row(masked, history, cfg, rng_state)
}

/// [`sample`] under a mask that came over the stage wire: bit `i` of
/// `words` (32 tokens per word, little bit first) allows token `i`; tokens
/// past the last word are not allowed. `Err` when nothing in the row is.
pub fn sample_with_words(
    logits: &[f32],
    history: &[i64],
    cfg: &SamplingConfig,
    rng_state: &mut u64,
    words: &[u32],
) -> EngineResult<i64> {
    let mut masked = logits.to_vec();
    let mut any = false;
    for (i, l) in masked.iter_mut().enumerate() {
        if words.get(i / 32).is_some_and(|w| (w >> (i % 32)) & 1 == 1) {
            any = true;
        } else {
            *l = f32::NEG_INFINITY;
        }
    }
    if !any {
        return Err(EngineError::Backend(
            "grammar: no token of the logits row is allowed".into(),
        ));
    }
    sample_masked_row(masked, history, cfg, rng_state)
}

/// Sample a row whose disallowed entries are already `-inf`.
fn sample_masked_row(
    masked: Vec<f32>,
    history: &[i64],
    cfg: &SamplingConfig,
    rng_state: &mut u64,
) -> EngineResult<i64> {
    let pick = sample_row(masked.clone(), history, cfg, rng_state);
    if masked.get(pick as usize) != Some(&f32::NEG_INFINITY) {
        return Ok(pick);
    }
    // No finite positive mass survived (a temperature so small that 1/t
    // overflows, NaN logits): the row's fallback can land on a masked id.
    // Take the best allowed logit instead.
    masked_argmax(&masked).map(|i| i as i64).ok_or_else(|| {
        EngineError::Backend("grammar: no token of the logits row is allowed".into())
    })
}

/// The highest finite logit among the allowed (not `-inf`) entries of a
/// masked row; the first allowed entry when none is finite (NaN logits).
/// `None` when nothing is allowed.
fn masked_argmax(masked: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    let mut first_allowed = None;
    for (i, &v) in masked.iter().enumerate() {
        if v == f32::NEG_INFINITY {
            continue;
        }
        first_allowed.get_or_insert(i);
        if v.is_finite() && best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i).or(first_allowed)
}

/// [`sample`] for a row no grammar constrains (normal requests, and the
/// stages that never see a grammar task).
pub fn sample_unmasked(
    logits: &[f32],
    history: &[i64],
    cfg: &SamplingConfig,
    rng_state: &mut u64,
) -> i64 {
    sample_row(logits.to_vec(), history, cfg, rng_state)
}

fn sample_row(
    mut work: Vec<f32>,
    history: &[i64],
    cfg: &SamplingConfig,
    rng_state: &mut u64,
) -> i64 {
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
            // A masked (`-inf`) logit stays masked whatever alpha is.
            if i < work.len() && work[i] != f32::NEG_INFINITY {
                let l = work[i];
                work[i] = if l > 0.0 { l / alpha } else { l * alpha };
            }
        }
    }

    // OpenAI frequency / presence penalties (subtractive, on logits). Applied
    // over the full running `history` — which includes the prompt tokens, like
    // the repetition penalty above. OpenAI scopes these to the completion only;
    // including the prompt is a deliberate, common local-sampler choice (it also
    // discourages parroting the prompt). Gated so there's zero cost when unused.
    if cfg.frequency_penalty != 0.0 || cfg.presence_penalty != 0.0 {
        let mut seen = std::collections::HashSet::new();
        for &tok in history.iter() {
            let i = tok as usize;
            if i < work.len() && work[i] != f32::NEG_INFINITY {
                // frequency: once per occurrence => count * penalty in total.
                work[i] -= cfg.frequency_penalty;
                // presence: once per distinct token.
                if seen.insert(tok) {
                    work[i] -= cfg.presence_penalty;
                }
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

    // Top-k. Disabled at 0. Keep the k highest-probability tokens, zero the
    // rest, renormalize. Applied before top-p (k bounds the candidate set,
    // p then trims within it — matches common samplers).
    if cfg.top_k > 0 && (cfg.top_k as usize) < work.len() {
        let k = cfg.top_k as usize;
        let mut idx: Vec<usize> = (0..work.len()).collect();
        idx.sort_by(|&a, &b| {
            work[b]
                .partial_cmp(&work[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut keep = vec![false; work.len()];
        for &i in idx.iter().take(k) {
            keep[i] = true;
        }
        let mut renorm = 0.0f32;
        for (i, v) in work.iter_mut().enumerate() {
            if keep[i] {
                renorm += *v;
            } else {
                *v = 0.0;
            }
        }
        if renorm > 0.0 {
            for v in work.iter_mut() {
                *v /= renorm;
            }
        }
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
    pick_categorical(&work, u) as i64
}

/// Inverse-CDF draw from normalized `probs` at `u` in [0, 1).
fn pick_categorical(probs: &[f32], u: f32) -> usize {
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        // A token without mass (cut by top-k/top-p, or masked) is never drawn,
        // even at `u == 0`.
        if p <= 0.0 {
            continue;
        }
        cum += p;
        if cum >= u {
            return i;
        }
    }
    // Numerical fallback if cum < u due to fp rounding: the last token that
    // has any mass (the old `len - 1` could be a token top-k/top-p cut).
    probs
        .iter()
        .rposition(|&p| p > 0.0)
        .unwrap_or(probs.len() - 1)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let l = vec![0.1, 0.4, 0.2, 0.05];
        let mut s = 1;
        assert_eq!(
            sample_unmasked(&l, &[], &SamplingConfig::default(), &mut s),
            1
        );
    }

    #[test]
    fn temperature_does_not_panic_on_uniform() {
        let l = vec![0.0; 100];
        let mut s = 1;
        let mut cfg = SamplingConfig::default();
        cfg.temperature = 1.0;
        let r = sample_unmasked(&l, &[], &cfg, &mut s);
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
        assert_eq!(sample_unmasked(&l, &history, &cfg, &mut s), 3);
    }

    #[test]
    fn frequency_penalty_demotes_repeated_token_greedy() {
        // logit[5] highest; history has 5 three times. A frequency penalty of
        // 2.0 subtracts 3*2=6 → logit[5]=10-6=4, below logit[3]=5. Greedy → 3.
        let mut l = vec![0.0_f32; 10];
        l[5] = 10.0;
        l[3] = 5.0;
        let history = vec![5_i64, 5, 5];
        let mut s = 1;
        let mut cfg = SamplingConfig::default();
        cfg.frequency_penalty = 2.0;
        assert_eq!(sample_unmasked(&l, &history, &cfg, &mut s), 3);
    }

    #[test]
    fn presence_penalty_is_count_independent() {
        // Presence subtracts once regardless of count. logit[5]=10, history has
        // 5 five times; presence 1.5 → logit[5]=8.5 (not 10-5*1.5). Still > 8.
        let mut l = vec![0.0_f32; 10];
        l[5] = 10.0;
        l[3] = 8.0;
        let history = vec![5_i64, 5, 5, 5, 5];
        let mut s = 1;
        let mut cfg = SamplingConfig::default();
        cfg.presence_penalty = 1.5;
        assert_eq!(sample_unmasked(&l, &history, &cfg, &mut s), 5);
    }

    #[test]
    fn top_k_one_forces_argmax_under_temperature() {
        // With top_k=1, only the single highest-prob token survives, so even
        // with temperature on, sampling must return the argmax deterministically.
        let mut l = vec![1.0_f32; 8];
        l[4] = 5.0;
        let mut s = 12345;
        let mut cfg = SamplingConfig::default();
        cfg.temperature = 1.0;
        cfg.top_k = 1;
        for _ in 0..20 {
            assert_eq!(sample_unmasked(&l, &[], &cfg, &mut s), 4);
        }
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
        let r = sample_unmasked(&l, &[], &cfg, &mut s);
        assert_eq!(r, 0);
    }

    // The invariant the discarded-prefill-sample fix rests on: a temperature-0 draw must not consume
    // RNG. `engine.rs` routes DISCARDED prefill samples through a temperature-0 config precisely so a
    // warm-resumed run — which skips `warm_prefix` forwards — leaves the RNG at the same offset a cold
    // run does. If temperature 0 ever starts drawing, warm and cold silently diverge for every
    // temperature > 0 request, which no greedy cert can observe.
    #[test]
    fn temperature_zero_does_not_advance_the_rng() {
        let mut l = vec![0.1_f32; 32];
        l[7] = 9.0;
        let mut cfg = SamplingConfig::default();
        cfg.temperature = 0.0;
        let mut s = 0xDEAD_BEEF_u64;
        let before = s;
        for _ in 0..16 {
            assert_eq!(sample_unmasked(&l, &[], &cfg, &mut s), 7);
        }
        assert_eq!(
            s, before,
            "temperature-0 sampling must not touch the RNG state"
        );
    }

    #[test]
    fn temperature_above_zero_does_advance_the_rng() {
        // Guards the test above from passing vacuously (e.g. if `sample` stopped using this state at
        // all): the >0 path must move what the 0 path leaves alone.
        let l = vec![1.0_f32; 32];
        let mut cfg = SamplingConfig::default();
        cfg.temperature = 1.0;
        let mut s = 0xDEAD_BEEF_u64;
        let before = s;
        let _ = sample_unmasked(&l, &[], &cfg, &mut s);
        assert_ne!(
            s, before,
            "temperature>0 sampling must advance the RNG state"
        );
    }

    // ---- grammar mask ----

    use crate::constrained::test_env;

    /// The request defaults a json task gets (repetition penalty 1.05 over 64
    /// tokens), at `temperature`.
    fn task_cfg(temperature: f32) -> SamplingConfig {
        SamplingConfig {
            temperature,
            repetition_penalty: 1.05,
            repetition_window: 64,
            ..SamplingConfig::default()
        }
    }

    /// Mid-string state of the weather schema: many bytes allowed, but not
    /// control bytes, not the EOS, not a raw newline.
    fn mid_string_mask() -> GrammarMask {
        let mut gm = test_env::mask(test_env::WEATHER);
        gm.replay(&br#"{"city":"Pa"#.map(u32::from)).unwrap();
        gm
    }

    /// Logits that favour masked tokens: control bytes and EOS score highest,
    /// and every masked id also appears in the history (so the penalties
    /// touch `-inf` entries).
    fn adversarial_logits(seed: u64) -> (Vec<f32>, Vec<i64>) {
        let mut s = seed.max(1);
        let mut l: Vec<f32> = (0..test_env::VOCAB)
            .map(|_| next_uniform(&mut s) * 4.0 - 2.0)
            .collect();
        l[..0x20].fill(50.0);
        l[test_env::EOS as usize] = 60.0;
        let history: Vec<i64> = (0..0x20).chain([test_env::EOS as i64, 0x41]).collect();
        (l, history)
    }

    #[test]
    fn masked_greedy_never_picks_a_masked_id() {
        for seed in 1..50u64 {
            let mut gm = mid_string_mask();
            let (l, history) = adversarial_logits(seed);
            let mut s = 7;
            let t = sample(&l, &history, &task_cfg(0.0), &mut s, Some(&mut gm)).unwrap();
            assert!(gm.allowed(t as u32), "seed {seed}: picked masked id {t}");
        }
    }

    #[test]
    fn masked_sampling_at_temperature_one_never_picks_a_masked_id() {
        let mut gm = mid_string_mask();
        let (l, history) = adversarial_logits(3);
        let mut cfg = task_cfg(1.0);
        cfg.top_p = 0.95;
        cfg.top_k = 40;
        cfg.frequency_penalty = 0.5;
        cfg.presence_penalty = 0.5;
        let mut s = init_rng(Some(42));
        let mut seen = std::collections::HashSet::new();
        for i in 0..1000 {
            let t = sample(&l, &history, &cfg, &mut s, Some(&mut gm)).unwrap();
            assert!(gm.allowed(t as u32), "draw {i}: picked masked id {t}");
            seen.insert(t);
        }
        assert!(seen.len() > 5, "sampling should still vary: {seen:?}");
    }

    #[test]
    fn masked_sampling_without_top_k_or_top_p_never_picks_a_masked_id() {
        let mut gm = mid_string_mask();
        let (l, history) = adversarial_logits(9);
        let mut s = init_rng(Some(1));
        for i in 0..1000 {
            let t = sample(&l, &history, &task_cfg(1.0), &mut s, Some(&mut gm)).unwrap();
            assert!(gm.allowed(t as u32), "draw {i}: picked masked id {t}");
        }
    }

    #[test]
    fn mask_dead_end_is_an_error_not_a_pick() {
        let mut gm = test_env::mask(test_env::WEATHER);
        // The first token must be `{`; a row too short to hold it has nothing allowed.
        let l = vec![1.0f32; usize::from(b'{')];
        let mut s = 1;
        assert!(sample(&l, &[], &task_cfg(0.0), &mut s, Some(&mut gm)).is_err());
    }

    #[test]
    fn no_mask_is_the_unmasked_sampler() {
        let (l, history) = adversarial_logits(5);
        let cfg = SamplingConfig {
            temperature: 0.8,
            top_p: 0.9,
            ..task_cfg(0.8)
        };
        let (mut a, mut b) = (11u64, 11u64);
        for _ in 0..200 {
            assert_eq!(
                sample(&l, &history, &cfg, &mut a, None).unwrap(),
                sample_unmasked(&l, &history, &cfg, &mut b)
            );
        }
    }

    #[test]
    fn rounding_fallback_returns_the_last_token_with_mass() {
        // The kept mass sums a little under `u`: the draw must land on the
        // last token top-k/top-p kept, never on a zeroed tail token.
        let probs = [0.5f32, 0.499, 0.0, 0.0];
        assert_eq!(pick_categorical(&probs, 0.9999), 1);
    }

    #[test]
    fn zero_mass_tokens_are_never_drawn_even_at_u_zero() {
        let probs = [0.0f32, 0.0, 1.0];
        assert_eq!(pick_categorical(&probs, 0.0), 2);
    }

    #[test]
    fn masked_pick_with_no_finite_mass_falls_back_to_the_masked_argmax() {
        // A temperature this small overflows 1/t to inf; the softmax is NaN.
        let mut gm = mid_string_mask();
        let (l, history) = adversarial_logits(4);
        let cfg = task_cfg(1e-40);
        let mut s = init_rng(Some(3));
        let t = sample(&l, &history, &cfg, &mut s, Some(&mut gm)).unwrap();
        assert!(gm.allowed(t as u32), "picked masked id {t}");
    }

    #[test]
    fn masked_pick_over_nan_logits_stays_allowed() {
        let mut gm = test_env::mask(test_env::WEATHER);
        let l = vec![f32::NAN; test_env::VOCAB];
        for temperature in [0.0, 1.0] {
            let mut s = init_rng(Some(5));
            let t = sample(&l, &[], &task_cfg(temperature), &mut s, Some(&mut gm)).unwrap();
            assert_eq!(t, i64::from(b'{'), "temperature {temperature}");
        }
    }
}
