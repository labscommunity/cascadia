//! The manifest's pre- and postprocessing steps, ported from Intel's
//! `physicalai` runtime (`inference/preprocessors`, `inference/postprocessors`
//! at `e3dd944`).
//!
//! Stats arithmetic runs in f64 because Intel's stats are float64 arrays
//! (`np.asarray` of JSON floats) and numpy promotes the f32 input to them; the
//! `1e-8` eps would vanish in f32.

use cascadia_engine::{EngineError, EngineResult};

use super::manifest::{FeatureStats, NormMode, StatsStep};

/// `stats_normalizer.py` / `stats_denormalizer.py` `_EPS`.
pub const EPS: f64 = 1e-8;

/// Number of state bins in the π0.5 prompt (`pi05.py` `_NUM_BINS`).
pub const NUM_BINS: usize = 256;

fn stat(v: &Option<Vec<f64>>) -> &[f64] {
    v.as_deref()
        .expect("manifest parsing checked the mode's stats")
}

/// Apply `step` to `x`, whose last axis has the stats' length.
fn apply(
    step: &StatsStep,
    key: &str,
    x: &[f64],
    f: impl Fn(NormMode, &FeatureStats, usize, f64) -> f64,
) -> EngineResult<Vec<f64>> {
    if step.mode == NormMode::Identity {
        return Ok(x.to_vec());
    }
    let Some(s) = step.stats_for(key) else {
        return Ok(x.to_vec());
    };
    let n = match step.mode {
        NormMode::MeanStd => stat(&s.mean).len(),
        NormMode::MinMax => stat(&s.min).len(),
        NormMode::Quantiles => stat(&s.q01).len(),
        NormMode::Identity => unreachable!(),
    };
    if !x.len().is_multiple_of(n) {
        return Err(EngineError::InvalidConfig(format!(
            "{key} has {} values; its stats have {n}",
            x.len()
        )));
    }
    Ok(x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let d = i % n;
            match &s.mask {
                Some(m) if !m[d] => v,
                _ => f(step.mode, s, d, v),
            }
        })
        .collect())
}

/// Intel's quantiles denominator: `q99 - q01`, replaced by eps only where it is 0.
fn q_denom(s: &FeatureStats, d: usize) -> f64 {
    let denom = stat(&s.q99)[d] - stat(&s.q01)[d];
    if denom == 0.0 {
        EPS
    } else {
        denom
    }
}

/// `normalize` (`stats_normalizer.py` `_normalize`).
pub fn normalize(step: &StatsStep, key: &str, x: &[f64]) -> EngineResult<Vec<f64>> {
    apply(step, key, x, |mode, s, d, x| match mode {
        NormMode::MeanStd => (x - stat(&s.mean)[d]) / (stat(&s.std)[d] + EPS),
        NormMode::MinMax => {
            let (lo, hi) = (stat(&s.min)[d], stat(&s.max)[d]);
            2.0 * (x - lo) / (hi - lo + EPS) - 1.0
        }
        NormMode::Quantiles => 2.0 * (x - stat(&s.q01)[d]) / q_denom(s, d) - 1.0,
        NormMode::Identity => x,
    })
}

/// `denormalize` (`stats_denormalizer.py` `_denormalize`); mean_std and
/// min_max take no eps there.
pub fn denormalize(step: &StatsStep, key: &str, x: &[f64]) -> EngineResult<Vec<f64>> {
    apply(step, key, x, |mode, s, d, x| match mode {
        NormMode::MeanStd => x * stat(&s.std)[d] + stat(&s.mean)[d],
        NormMode::MinMax => {
            let (lo, hi) = (stat(&s.min)[d], stat(&s.max)[d]);
            (x + 1.0) / 2.0 * (hi - lo) + lo
        }
        NormMode::Quantiles => (x + 1.0) * q_denom(s, d) / 2.0 + stat(&s.q01)[d],
        NormMode::Identity => x,
    })
}

/// π0.5 state bins: `np.digitize(x, np.linspace(-1, 1, 257)[:-1]) - 1`.
///
/// -1 below -1, 255 at or above `1 - 2/256`; NaN sorts last in numpy, so 255.
pub fn pi05_bins(state: &[f64]) -> Vec<i32> {
    // linspace's step 2/256 is exact in binary, so the edges are -1 + i/128 exactly.
    let edge = |i: usize| -1.0 + i as f64 * (2.0 / NUM_BINS as f64);
    state
        .iter()
        .map(|&x| {
            let count = if x.is_nan() {
                NUM_BINS
            } else {
                (0..NUM_BINS).take_while(|&i| edge(i) <= x).count()
            };
            count as i32 - 1
        })
        .collect()
}

/// Python `str.isspace`: Rust's White_Space plus the separators U+001C..U+001F.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// π0.5 prompt text (`pi05.py` `__call__`).
pub fn pi05_prompt(task: &str, bins: &[i32]) -> String {
    let cleaned = task.trim_matches(py_space).replace(['_', '\n'], " ");
    let state = bins
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    format!("Task: {cleaned}, State: {state};\nAction: ")
}

/// `new_line`: append `"\n"` unless the task already ends with one.
pub fn new_line(task: &str) -> String {
    if task.ends_with('\n') {
        task.to_string()
    } else {
        format!("{task}\n")
    }
}

/// One uint8 HWC RGB frame to f32 CHW in `[-1, 1]`, appended to `out`.
///
/// Same f32 arithmetic as Intel (`x.astype(f32) / 255.0`, then `* 2 - 1`).
pub fn image_to_chw(hwc: &[u8], height: usize, width: usize, out: &mut Vec<f32>) {
    debug_assert_eq!(hwc.len(), height * width * 3);
    for c in 0..3 {
        for px in 0..height * width {
            out.push(hwc[px * 3 + c] as f32 / 255.0 * 2.0 - 1.0);
        }
    }
}

/// `action_chunk_trimmer`: keep the first `n` of `steps` rows of `dim` values.
pub fn trim_actions(
    mut actions: Vec<f32>,
    steps: usize,
    dim: usize,
    n: usize,
) -> (Vec<f32>, usize) {
    if steps > n {
        actions.truncate(n * dim);
        (actions, n)
    } else {
        (actions, steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(mode: NormMode, s: FeatureStats) -> StatsStep {
        StatsStep {
            mode,
            features: vec![("state".into(), s)],
        }
    }

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 1e-12, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn mean_std_divides_by_std_plus_eps() {
        let s = step(
            NormMode::MeanStd,
            FeatureStats {
                mean: Some(vec![1.0, 0.0]),
                std: Some(vec![2.0, 0.0]),
                ..Default::default()
            },
        );
        let y = normalize(&s, "state", &[3.0, 1e-8]).unwrap();
        // (3-1)/(2+1e-8); a zero std divides by eps alone.
        close(&y, &[2.0 / (2.0 + 1e-8), 1.0]);
        assert_ne!(y[0], 1.0, "eps must survive in f64");
        let back = denormalize(&s, "state", &[2.0, 1.0]).unwrap();
        // x*std + mean, no eps.
        assert_eq!(back, vec![5.0, 0.0]);
    }

    #[test]
    fn min_max_maps_to_minus_one_one() {
        let s = step(
            NormMode::MinMax,
            FeatureStats {
                min: Some(vec![0.0]),
                max: Some(vec![4.0]),
                ..Default::default()
            },
        );
        let y = normalize(&s, "state", &[0.0, 4.0, 1.0]).unwrap();
        close(
            &y,
            &[-1.0, 2.0 * 4.0 / (4.0 + EPS) - 1.0, 2.0 / (4.0 + EPS) - 1.0],
        );
        assert_eq!(
            denormalize(&s, "state", &[-1.0, 1.0, 0.0]).unwrap(),
            vec![0.0, 4.0, 2.0]
        );
    }

    #[test]
    fn quantiles_use_eps_only_for_a_zero_range() {
        // stats_normalizer.py:214: `np.where(denom == 0, _EPS, denom)`, no eps otherwise.
        let s = step(
            NormMode::Quantiles,
            FeatureStats {
                q01: Some(vec![-0.5, 3.0]),
                q99: Some(vec![1.5, 3.0]),
                ..Default::default()
            },
        );
        let y = normalize(&s, "state", &[0.5, 3.0]).unwrap();
        assert_eq!(y[0], 0.0, "exact with no eps: 2*(1.0)/2.0 - 1");
        assert_eq!(y[1], -1.0, "2*0/eps - 1");
        let back = denormalize(&s, "state", &[1.0, 1.0]).unwrap();
        // (x+1)*denom/2 + q01, with denom eps for the flat dim.
        close(&back, &[1.5, 3.0 + EPS]);
    }

    #[test]
    fn identity_mode_mask_and_broadcast() {
        let id = StatsStep {
            mode: NormMode::Identity,
            features: vec![],
        };
        assert_eq!(normalize(&id, "state", &[7.0]).unwrap(), vec![7.0]);
        let s = step(
            NormMode::MeanStd,
            FeatureStats {
                mean: Some(vec![1.0, 1.0]),
                std: Some(vec![1.0, 1.0]),
                mask: Some(vec![true, false]),
                ..Default::default()
            },
        );
        // [2, 2] actions broadcast the per-dim stats over rows; masked dims pass through.
        let y = denormalize(&s, "state", &[1.0, 1.0, 2.0, 2.0]).unwrap();
        assert_eq!(y, vec![2.0, 1.0, 3.0, 2.0]);
        assert!(normalize(&s, "state", &[1.0, 2.0, 3.0]).is_err());
        // A key without stats passes through, as Intel skips it.
        assert_eq!(normalize(&s, "action", &[9.0]).unwrap(), vec![9.0]);
    }

    #[test]
    fn pi05_bins_follow_np_digitize() {
        let edge = 1.0 - 2.0 / 256.0;
        let got = pi05_bins(&[
            -1.5,
            -1.0 - 1e-12,
            -1.0,
            -1.0 + 1.0 / 128.0 - 1e-12,
            -1.0 + 1.0 / 128.0,
            0.0,
            edge - 1e-12,
            edge,
            1.0,
            7.0,
            f64::NAN,
        ]);
        // -1 below the first edge (pi05.py:90 does not clip), 255 from the last edge up.
        assert_eq!(got, vec![-1, -1, 0, 0, 1, 128, 254, 255, 255, 255, 255]);
    }

    #[test]
    fn pi05_prompt_text() {
        let p = pi05_prompt("  pick_up the\nblock \t", &[-1, 0, 255]);
        assert_eq!(p, "Task: pick up the block, State: -1 0 255;\nAction: ");
        // Python's strip() also drops U+001C..U+001F; Rust's trim() does not.
        assert_eq!(
            pi05_prompt("\u{1f}go\u{a0}", &[]),
            "Task: go, State: ;\nAction: "
        );
    }

    #[test]
    fn new_line_appends_once() {
        assert_eq!(new_line("open the drawer"), "open the drawer\n");
        assert_eq!(new_line("open the drawer\n"), "open the drawer\n");
        assert_eq!(new_line(""), "\n");
    }

    #[test]
    fn image_becomes_chw_in_minus_one_one() {
        // 1x2 frame: pixels (0, 255, 51) and (102, 204, 255).
        let mut out = Vec::new();
        image_to_chw(&[0, 255, 51, 102, 204, 255], 1, 2, &mut out);
        let f = |v: u8| v as f32 / 255.0 * 2.0 - 1.0;
        assert_eq!(out, vec![-1.0, f(102), 1.0, f(204), f(51), 1.0]);
    }

    #[test]
    fn trimmer_keeps_the_first_rows() {
        let a: Vec<f32> = (0..6).map(|v| v as f32).collect();
        assert_eq!(
            trim_actions(a.clone(), 3, 2, 2),
            (vec![0.0, 1.0, 2.0, 3.0], 2)
        );
        // Shorter chunks are kept whole (action_chunk_trimmer.py: `shape[1] > n`).
        assert_eq!(trim_actions(a.clone(), 3, 2, 5), (a, 3));
    }
}
