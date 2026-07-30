//! Token sampling: temperature / top-p / repetition penalty / EOS stop.
//!
//! All sampling math runs on f32 logits. The deterministic argmax path
//! is preserved for `temperature == 0`.

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

    // OpenAI frequency / presence penalties (subtractive, on logits). Applied
    // over the full running `history` — which includes the prompt tokens, like
    // the repetition penalty above. OpenAI scopes these to the completion only;
    // including the prompt is a deliberate, common local-sampler choice (it also
    // discourages parroting the prompt). Gated so there's zero cost when unused.
    if cfg.frequency_penalty != 0.0 || cfg.presence_penalty != 0.0 {
        let mut seen = std::collections::HashSet::new();
        for &tok in history.iter() {
            let i = tok as usize;
            if i < work.len() {
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
        assert_eq!(sample(&l, &history, &cfg, &mut s), 3);
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
        assert_eq!(sample(&l, &history, &cfg, &mut s), 5);
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
            assert_eq!(sample(&l, &[], &cfg, &mut s), 4);
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
        let r = sample(&l, &[], &cfg, &mut s);
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
            assert_eq!(sample(&l, &[], &cfg, &mut s), 7);
        }
        assert_eq!(s, before, "temperature-0 sampling must not touch the RNG state");
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
        let _ = sample(&l, &[], &cfg, &mut s);
        assert_ne!(s, before, "temperature>0 sampling must advance the RNG state");
    }
}
