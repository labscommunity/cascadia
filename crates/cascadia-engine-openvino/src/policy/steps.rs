//! The manifest's pre- and postprocessing steps, ported from Intel's
//! `physicalai` runtime (`inference/preprocessors`, `inference/postprocessors`
//! at `e3dd944`).
//!
//! Stats arithmetic runs in f64 because Intel's stats are float64 arrays
//! (`np.asarray` of JSON floats) and numpy promotes the f32 input to them; the
//! `1e-8` eps would vanish in f32.

use cascadia_engine::{EngineError, EngineResult};

use super::manifest::{Bounds, FeatureStats, NormMode, StatsStep, XVLA_ACTION_DIM};

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
    f: impl Fn(NormMode, Bounds, &FeatureStats, usize, f64) -> f64,
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
        NormMode::MinMaxClip => stat(s.bounds(step.bounds).0).len(),
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
                _ => f(step.mode, step.bounds, s, d, v),
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

/// `min_max_clip`'s `(lo, hi)` for dimension `d`.
fn clip_bounds(s: &FeatureStats, b: Bounds, d: usize) -> (f64, f64) {
    let (lo, hi) = s.bounds(b);
    (stat(lo)[d], stat(hi)[d])
}

/// `np.isclose(hi, lo)` with numpy's defaults (`rtol` 1e-5, `atol` 1e-8).
fn is_close(hi: f64, lo: f64) -> bool {
    (hi - lo).abs() <= 1e-8 + 1e-5 * lo.abs()
}

/// `normalize` (`stats_normalizer.py` `_normalize`; `min_max_clip`: GR00T's
/// `normalize_values_minmax`, then the clip).
pub fn normalize(step: &StatsStep, key: &str, x: &[f64]) -> EngineResult<Vec<f64>> {
    apply(step, key, x, |mode, b, s, d, x| match mode {
        NormMode::MeanStd => (x - stat(&s.mean)[d]) / (stat(&s.std)[d] + EPS),
        NormMode::MinMax => {
            let (lo, hi) = (stat(&s.min)[d], stat(&s.max)[d]);
            2.0 * (x - lo) / (hi - lo + EPS) - 1.0
        }
        NormMode::Quantiles => 2.0 * (x - stat(&s.q01)[d]) / q_denom(s, d) - 1.0,
        NormMode::MinMaxClip => {
            let (lo, hi) = clip_bounds(s, b, d);
            if is_close(hi, lo) {
                0.0
            } else {
                (2.0 * (x - lo) / (hi - lo) - 1.0).clamp(-1.0, 1.0)
            }
        }
        NormMode::Identity => x,
    })
}

/// `denormalize` (`stats_denormalizer.py` `_denormalize`); mean_std and
/// min_max take no eps there.
pub fn denormalize(step: &StatsStep, key: &str, x: &[f64]) -> EngineResult<Vec<f64>> {
    apply(step, key, x, |mode, b, s, d, x| match mode {
        NormMode::MeanStd => x * stat(&s.std)[d] + stat(&s.mean)[d],
        NormMode::MinMax => {
            let (lo, hi) = (stat(&s.min)[d], stat(&s.max)[d]);
            (x + 1.0) / 2.0 * (hi - lo) + lo
        }
        NormMode::Quantiles => (x + 1.0) * q_denom(s, d) / 2.0 + stat(&s.q01)[d],
        NormMode::MinMaxClip => {
            let (lo, hi) = clip_bounds(s, b, d);
            (x.clamp(-1.0, 1.0) + 1.0) / 2.0 * (hi - lo) + lo
        }
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
pub fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python `str.strip()`.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

/// `gripper_binarize` on row-major `[steps, dim]` actions; NaN goes below, as in numpy.
pub fn gripper_binarize(
    actions: &mut [f64],
    dim: usize,
    index: usize,
    threshold: f64,
    above: f64,
    below: f64,
) {
    for row in actions.chunks_mut(dim) {
        row[index] = if row[index] > threshold { above } else { below };
    }
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

/// [`image_to_chw`] of the frame rotated by 180 degrees (`image_flip`;
/// `torch.flip(x, dims=[H, W])`): pixel `p` of `h*w` reads pixel `h*w - 1 - p`.
pub fn image_to_chw_rot180(hwc: &[u8], height: usize, width: usize, out: &mut Vec<f32>) {
    debug_assert_eq!(hwc.len(), height * width * 3);
    let n = height * width;
    for c in 0..3 {
        for px in 0..n {
            out.push(hwc[(n - 1 - px) * 3 + c] as f32 / 255.0 * 2.0 - 1.0);
        }
    }
}

/// `xvla_rotation_6d_to_axis_angle` on `steps` rows of `dim` values: each row
/// becomes position (0..3), [`rotate6d_to_axis_angle`] of 3..9, and the
/// gripper (9) as 1 above 0.5, else -1. The manifest load checked `dim >= 10`.
pub fn xvla_rotation_6d_to_axis_angle(actions: &[f64], steps: usize, dim: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(steps * XVLA_ACTION_DIM);
    for row in actions.chunks_exact(dim).take(steps) {
        out.extend_from_slice(&row[..3]);
        let r6d: [f64; 6] = row[3..9].try_into().expect("six values");
        out.extend_from_slice(&rotate6d_to_axis_angle(&r6d));
        out.push(if row[9] > 0.5 { 1.0 } else { -1.0 });
    }
    out
}

/// LeRobot's X-VLA `rotate6d_to_axis_angle` (`utils.py`) for one row, in f64:
/// Gram-Schmidt with its `1e-6` eps, `mat2quat` (the top eigenvector of
/// Bar-Itzhack's K, sign fixed to `w >= 0`), then `quat2axisangle`.
pub fn rotate6d_to_axis_angle(r6d: &[f64; 6]) -> [f64; 3] {
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let a1 = [r6d[0], r6d[1], r6d[2]];
    let a2 = [r6d[3], r6d[4], r6d[5]];
    let n1 = norm(a1) + 1e-6;
    let b1 = a1.map(|x| x / n1);
    let dot = b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2];
    let orth = [
        a2[0] - dot * b1[0],
        a2[1] - dot * b1[1],
        a2[2] - dot * b1[2],
    ];
    let n2 = norm(orth) + 1e-6;
    let b2 = orth.map(|x| x / n2);
    let b3 = [
        b1[1] * b2[2] - b1[2] * b2[1],
        b1[2] * b2[0] - b1[0] * b2[2],
        b1[0] * b2[1] - b1[1] * b2[0],
    ];
    // np.stack([b1, b2, b3], axis=-1): b1, b2, b3 are the columns.
    let m = |r: usize, c: usize| [b1, b2, b3][c][r];
    quat2axisangle(mat2quat(m))
}

/// `mat2quat`: `(x, y, z, w)`.
fn mat2quat(m: impl Fn(usize, usize) -> f64) -> [f64; 4] {
    let (m00, m01, m02) = (m(0, 0), m(0, 1), m(0, 2));
    let (m10, m11, m12) = (m(1, 0), m(1, 1), m(1, 2));
    let (m20, m21, m22) = (m(2, 0), m(2, 1), m(2, 2));
    // np.linalg.eigh reads the lower triangle.
    let lower = [
        [m00 - m11 - m22, 0.0, 0.0, 0.0],
        [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
        [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
        [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
    ];
    let mut k = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..=i {
            k[i][j] = lower[i][j] / 3.0;
            k[j][i] = k[i][j];
        }
    }
    let (values, vectors) = jacobi_eigen(k);
    let top = (0..4).fold(0, |b, i| if values[i] > values[b] { i } else { b });
    let mut q = [0.0; 4];
    for (i, qi) in q.iter_mut().enumerate() {
        *qi = vectors[i][top];
    }
    // K's rows are (x, y, z, w); LeRobot negates when w < 0.
    if q[3] < 0.0 {
        q = q.map(|x| -x);
    }
    q
}

/// Eigenvalues and eigenvectors (columns) of a symmetric 4x4, by cyclic Jacobi.
fn jacobi_eigen(mut a: [[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    let mut v = [[0.0; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let off: f64 = (0..4)
            .flat_map(|i| (0..4).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..3 {
            for q in p + 1..4 {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                let (rp, rq) = (a[p], a[q]);
                a[p] = std::array::from_fn(|k| c * rp[k] - s * rq[k]);
                a[q] = std::array::from_fn(|k| s * rp[k] + c * rq[k]);
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2], a[3][3]], v)
}

/// `quat2axisangle`: the unit axis scaled by the angle; zero for no rotation.
fn quat2axisangle(q: [f64; 4]) -> [f64; 3] {
    let w = q[3].clamp(-1.0, 1.0);
    let den = (1.0 - w * w).sqrt();
    // math.isclose(den, 0.0) with its defaults is true only at exactly 0.
    if den == 0.0 {
        return [0.0; 3];
    }
    let scale = 2.0 * w.acos() / den;
    [q[0] * scale, q[1] * scale, q[2] * scale]
}

/// `action_chunk_trimmer`: keep the first `n` of `steps` rows of `dim` values.
pub fn trim_actions<T>(mut actions: Vec<T>, steps: usize, dim: usize, n: usize) -> (Vec<T>, usize) {
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

    #[test]
    fn gripper_binarize_is_strictly_above_and_nan_goes_below() {
        let mut a = vec![0.1, 0.5, -0.2, 0.0, 0.9, 1e-9, 3.0, f64::NAN];
        gripper_binarize(&mut a, 2, 1, 0.0, 1.0, -1.0);
        assert_eq!(a[..7], [0.1, 1.0, -0.2, -1.0, 0.9, 1.0, 3.0]);
        assert_eq!(a[7], -1.0, "NaN > threshold is false");
        let mut b = vec![0.5, 0.4];
        gripper_binarize(&mut b, 1, 0, 0.5, 2.0, 0.0);
        assert_eq!(b, [0.0, 0.0], "equal to the threshold is below");
    }

    #[test]
    fn py_strip_is_pythons_strip() {
        assert_eq!(py_strip(" \t pick up the bowl\n\u{1f}"), "pick up the bowl");
        assert_eq!(py_strip("a b"), "a b");
    }

    fn step(mode: NormMode, s: FeatureStats) -> StatsStep {
        StatsStep {
            mode,
            features: vec![("state".into(), s)],
            bounds: Bounds::MinMax,
            pad_to: None,
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
    fn min_max_clip_zeroes_a_flat_range_and_clips_both_ways() {
        // GR00T's normalize_values_minmax: 0 where np.isclose(hi, lo), then clip; decode clips first.
        let mut s = step(
            NormMode::MinMaxClip,
            FeatureStats {
                q01: Some(vec![0.0, 2.0, -1.0]),
                q99: Some(vec![4.0, 2.0 + 1e-9, 1.0]),
                min: Some(vec![-9.0, -9.0, -9.0]),
                max: Some(vec![9.0, 9.0, 9.0]),
                ..Default::default()
            },
        );
        s.bounds = Bounds::Quantiles;
        let y = normalize(&s, "state", &[1.0, 5.0, 3.0, 9.0, 0.0, -2.0]).unwrap();
        assert_eq!(y, vec![-0.5, 0.0, 1.0, 1.0, 0.0, -1.0]);
        let back = denormalize(&s, "state", &[0.5, 0.0, 2.0, -3.0, 1.0, 0.0]).unwrap();
        close(&back, &[3.0, 2.0 + 0.5e-9, 1.0, 0.0, 2.0 + 1e-9, 0.0]);
        s.bounds = Bounds::MinMax;
        assert_eq!(
            normalize(&s, "state", &[0.0, 0.0, 0.0]).unwrap(),
            vec![0.0; 3]
        );
    }

    #[test]
    fn identity_mode_mask_and_broadcast() {
        let id = StatsStep {
            mode: NormMode::Identity,
            features: vec![],
            bounds: Bounds::MinMax,
            pad_to: None,
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
    fn rot180_reverses_the_pixel_order_per_channel() {
        // 1x3 frame, pixels (0,1,2) (3,4,5) (6,7,8).
        let mut out = Vec::new();
        image_to_chw_rot180(&[0, 1, 2, 3, 4, 5, 6, 7, 8], 1, 3, &mut out);
        let f = |v: u8| v as f32 / 255.0 * 2.0 - 1.0;
        let want: Vec<f32> = [6, 3, 0, 7, 4, 1, 8, 5, 2].into_iter().map(f).collect();
        assert_eq!(out, want);
    }

    /// Rows of LeRobot's `rotate6d_to_axis_angle`, mostly near pi where the
    /// axis sign is most fragile, plus scaled and non-orthogonal rows.
    #[test]
    fn rotation_6d_matches_lerobot() {
        let v: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/policy/xvla-rotation-reference.json"
        ))
        .unwrap();
        let rows: Vec<[f64; 6]> = serde_json::from_value(v["r6d"].clone()).unwrap();
        let want: Vec<[f64; 3]> = serde_json::from_value(v["axis_angle"].clone()).unwrap();
        assert_eq!(rows.len(), want.len());
        for (r, w) in rows.iter().zip(&want) {
            let got = rotate6d_to_axis_angle(r);
            for (g, w) in got.iter().zip(w) {
                // LeRobot's mat2quat runs in f32; the port runs in f64.
                assert!((g - w).abs() <= 1e-5, "{r:?}: {got:?} vs {w:?}");
            }
        }
        assert!(
            want.iter()
                .any(
                    |w| (w.iter().map(|x| x * x).sum::<f64>().sqrt() - std::f64::consts::PI).abs()
                        < 2e-3
                ),
            "the fixture covers angles near pi"
        );
    }

    #[test]
    fn xvla_rows_become_position_axis_angle_and_a_binary_gripper() {
        // Two 12-wide rows: identity rotation, gripper at 0.6 then at 0.5.
        let row = |g: f64| {
            let mut r = vec![0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, g, 9.0, 9.0];
            r.truncate(12);
            r
        };
        let a: Vec<f64> = [row(0.6), row(0.5)].concat();
        let out = xvla_rotation_6d_to_axis_angle(&a, 2, 12);
        assert_eq!(out.len(), 14);
        assert_eq!(&out[..3], &[0.1, -0.2, 0.3]);
        assert_eq!(&out[3..6], &[0.0, 0.0, 0.0]);
        // np.where(x > 0.5, 1, -1): 0.5 itself is -1.
        assert_eq!((out[6], out[13]), (1.0, -1.0));
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
