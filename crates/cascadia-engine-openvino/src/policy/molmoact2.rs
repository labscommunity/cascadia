//! MolmoAct2 host steps, ported from Intel's `physicalai` runtime at `e3dd944`
//! (`preprocessors/molmoact2/{processor,inputs,image}.py`,
//! `postprocessors/molmoact2/processor.py`): the prompt with state tokens,
//! BOS and image-token expansion, Molmo patches and pooling indices, and the
//! clipped quantile denormalize. Intel holds the stats as f32, so the state and
//! action arithmetic is f32 here too.
//!
//! Frames arrive at `image_size`, so Intel's `INTER_LINEAR_EXACT` resize is the
//! identity and is skipped.

use cascadia_engine::{EngineError, EngineResult};
use cascadia_types::Image;

use super::cameras::CameraLayout;
use super::manifest::{
    F32Quantiles, FeatureType, MolmoAct2InputsStep, MolmoAct2Step, PolicyManifest,
};
use super::steps::py_space;

fn refuse<T>(msg: impl std::fmt::Display) -> EngineResult<T> {
    Err(EngineError::InvalidConfig(format!(
        "policy molmoact2: {msg}"
    )))
}

/// `_EPS` of Intel's stats steps, in f32 as `np.where` keeps it.
const EPS: f32 = 1e-8;

/// What the package fixes: frame size, cameras, patch and pooling grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MolmoAct2Geometry {
    pub hw: [usize; 2],
    /// Advertised camera names (`image_keys` without `images.`), in order.
    pub cameras: Vec<String>,
    pub state_len: usize,
    /// `[rows, cols]` of patches per frame.
    pub patches: [usize; 2],
    /// `[rows, cols]` after pooling (the patch grid padded up to whole cells).
    pub pooled: [usize; 2],
    /// Pixels per patch row: `patch² · 3`.
    pub patch_pixels: usize,
}

impl MolmoAct2Geometry {
    pub fn new(m: &PolicyManifest) -> EngineResult<Self> {
        let Some((pre, inputs)) = m.molmoact2_steps() else {
            return refuse("molmoact2 needs molmoact2 and molmoact2_inputs");
        };
        if pre.image_size != inputs.image_size {
            return refuse(format!(
                "molmoact2 image_size {:?} differs from molmoact2_inputs' {:?}",
                pre.image_size, inputs.image_size
            ));
        }
        let hw = pre.image_size;
        let p = inputs.patch_size;
        if hw[0] % p != 0 || hw[1] % p != 0 {
            return refuse(format!(
                "image_size {hw:?} is not a multiple of patch_size {p}"
            ));
        }
        let mut cameras: Vec<String> = Vec::new();
        for key in &pre.image_keys {
            let name = key.strip_prefix("images.").unwrap_or(key).to_string();
            if name.is_empty() || cameras.contains(&name) {
                return refuse(format!("image key {key:?} is empty or repeated"));
            }
            cameras.push(name);
        }
        for f in m
            .input_features
            .iter()
            .filter(|f| f.ftype == FeatureType::Visual)
        {
            let name = f.name.strip_prefix("images.").unwrap_or(&f.name);
            if !cameras.iter().any(|c| c == name) {
                return refuse(format!("camera {:?} is not in image_keys", f.name));
            }
            if f.shape != [3, hw[0] as i64, hw[1] as i64] {
                return refuse(format!(
                    "camera {:?} has shape {:?}; image_size is {hw:?}",
                    f.name, f.shape
                ));
            }
        }
        let state_len = pre.state.q01.len();
        if let Some([n]) = m
            .input_features
            .iter()
            .find(|f| f.ftype == FeatureType::State)
            .map(|f| f.shape.as_slice())
        {
            if *n as usize != state_len {
                return refuse(format!(
                    "the STATE feature has {n} values; state_stats have {state_len}"
                ));
            }
        }
        let patches = [hw[0] / p, hw[1] / p];
        let [ph, pw] = inputs.pooling_size;
        Ok(Self {
            hw,
            cameras,
            state_len,
            patches,
            pooled: [patches[0].div_ceil(ph), patches[1].div_ceil(pw)],
            patch_pixels: p * p * 3,
        })
    }

    pub fn patches_per_image(&self) -> usize {
        self.patches[0] * self.patches[1]
    }

    pub fn pooled_per_image(&self) -> usize {
        self.pooled[0] * self.pooled[1]
    }
}

/// Case-insensitive match of an ASCII lowercase `word` at the start of `s`,
/// as Python's `re.IGNORECASE` reads it (`İ`/`ı` for `i`, `ſ` for `s`, the
/// Kelvin sign for `k`); the byte length matched.
fn ci_prefix(s: &str, word: &str) -> Option<usize> {
    let mut at = 0;
    let mut chars = s.chars();
    for w in word.chars() {
        let c = chars.next()?;
        let same = c.to_ascii_lowercase() == w
            || (w == 'i' && matches!(c, 'İ' | 'ı'))
            || (w == 's' && c == 'ſ')
            || (w == 'k' && c == '\u{212a}');
        if !same {
            return None;
        }
        at += c.len_utf8();
    }
    Some(at)
}

/// `^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*`.
fn strip_label(s: &str) -> Option<&str> {
    let words: [&[&str]; 4] = [
        &["task"],
        &["instruction"],
        &["language", "instruction"],
        &["goal"],
    ];
    for parts in words {
        let mut rest = s;
        let mut ok = true;
        for (i, part) in parts.iter().enumerate() {
            if i > 0 {
                match rest.chars().next() {
                    Some(c @ ('_' | ' ')) => rest = &rest[c.len_utf8()..],
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            match ci_prefix(rest, part) {
                Some(n) => rest = &rest[n..],
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let rest = rest.trim_start_matches(py_space);
        if let Some(rest) = rest.strip_prefix([':', '-']) {
            return Some(rest.trim_start_matches(py_space));
        }
    }
    None
}

/// `^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+`.
fn strip_phrase(s: &str) -> Option<&str> {
    for first in ["the", "your"] {
        let mut rest = s;
        let mut ok = true;
        for word in [first, "task", "is", "to"] {
            if word != first {
                let trimmed = rest.trim_start_matches(py_space);
                if trimmed.len() == rest.len() {
                    ok = false;
                    break;
                }
                rest = trimmed;
            }
            match ci_prefix(rest, word) {
                Some(n) => rest = &rest[n..],
                None => {
                    ok = false;
                    break;
                }
            }
        }
        let trimmed = rest.trim_start_matches(py_space);
        if ok && trimmed.len() < rest.len() {
            return Some(trimmed);
        }
    }
    None
}

/// `_normalize_text`: whitespace runs to one space, one leading label or
/// "the task is to" dropped, trailing `.,!?;:` dropped, lowercased.
pub fn normalize_text(task: &str) -> String {
    let mut text = String::with_capacity(task.len());
    let mut space = false;
    for c in task.chars() {
        if py_space(c) {
            space = true;
        } else {
            if space {
                text.push(' ');
            }
            space = false;
            text.push(c);
        }
    }
    let mut text = text.trim_matches(py_space).to_string();
    if let Some(rest) = strip_label(&text) {
        text = rest.trim_matches(py_space).to_string();
    }
    if let Some(rest) = strip_phrase(&text) {
        text = rest.trim_matches(py_space).to_string();
    }
    text.trim_end_matches(['.', ',', '!', '?', ';', ':'])
        .trim_matches(py_space)
        .to_lowercase()
}

/// `_wrap`: `start + value + end` unless empty, disabled or already wrapped.
fn wrap(value: &str, start: &str, end: &str, enabled: bool) -> String {
    if value.is_empty() || !enabled || (value.starts_with(start) && value.ends_with(end)) {
        value.to_string()
    } else {
        format!("{start}{value}{end}")
    }
}

/// `_discrete_state` on the normalized, clipped state.
pub fn discrete_state(state: &[f32], num_tokens: usize) -> String {
    let top = (num_tokens - 1) as f32;
    let mut out = String::from("<state_start>");
    for &s in state {
        let s = if s.is_nan() {
            0.0
        } else if s == f32::INFINITY {
            1.0
        } else if s == f32::NEG_INFINITY {
            -1.0
        } else {
            s
        };
        let bin = ((s.clamp(-1.0, 1.0) + 1.0) / 2.0 * top).round_ties_even() as i64;
        out.push_str(&format!("<state_{bin}>"));
    }
    out.push_str("<state_end>");
    out
}

/// Every state token the prompt can carry.
pub fn state_tokens(num_tokens: usize) -> Vec<String> {
    (0..num_tokens).map(|i| format!("<state_{i}>")).collect()
}

/// The f32 quantile normalize with its mask (`stats_normalizer._normalize`).
fn normalize(q: &F32Quantiles, x: &[f32]) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let d = i % q.q01.len();
            if q.mask.as_ref().is_some_and(|m| !m[d]) {
                return v;
            }
            let denom = q.q99[d] - q.q01[d];
            let denom = if denom == 0.0 { EPS } else { denom };
            2.0 * (v - q.q01[d]) / denom - 1.0
        })
        .collect()
}

/// `molmoact2_postprocess` on row-major actions: clip, then the f32 quantile
/// denormalize with its mask (`stats_denormalizer._denormalize`).
pub fn postprocess(q: &F32Quantiles, actions: &[f32]) -> Vec<f32> {
    actions
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let v = v.clamp(-1.0, 1.0);
            let d = i % q.q01.len();
            if q.mask.as_ref().is_some_and(|m| !m[d]) {
                return v;
            }
            let denom = q.q99[d] - q.q01[d];
            let denom = if denom == 0.0 { EPS } else { denom };
            (v + 1.0) * denom / 2.0 + q.q01[d]
        })
        .collect()
}

/// `_build_prompt` for one call: the raw `state` normalized, clipped and
/// written as state tokens; one `<|image|>` per camera.
pub fn render_prompt(p: &MolmoAct2Step, task: &str, state: &[f32]) -> String {
    let state: Vec<f32> = normalize(&p.state, state)
        .into_iter()
        .map(|x| x.clamp(-1.0, 1.0))
        .collect();
    let setup = wrap(
        &p.setup_type,
        "<setup_start>",
        "<setup_end>",
        p.add_setup_tokens,
    );
    let control = wrap(
        &p.control_mode,
        "<control_start>",
        "<control_end>",
        p.add_control_tokens,
    );
    let body = format!(
        "The task is to {}. The setup is {setup}. The current state of the robot is {}. \
             The expected control mode is {control}. Given these, what action should the robot \
             take to complete the task?",
        normalize_text(task),
        discrete_state(&state, p.num_state_tokens)
    );
    let n = p.image_keys.len();
    let images = if n == 1 {
        "<|image|>".to_string()
    } else {
        (1..=n).map(|i| format!("Image {i}<|image|>")).collect()
    };
    format!("{images}<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n<action_output>")
}

/// One package's MolmoAct2 host steps.
#[derive(Clone, Debug)]
pub struct MolmoAct2Host {
    pub geo: MolmoAct2Geometry,
    pre: MolmoAct2Step,
    inputs: MolmoAct2InputsStep,
}

impl MolmoAct2Host {
    pub fn new(m: &PolicyManifest) -> EngineResult<Self> {
        let geo = MolmoAct2Geometry::new(m)?;
        let (pre, inputs) = m.molmoact2_steps().expect("the geometry checked the steps");
        Ok(Self {
            geo,
            pre: pre.clone(),
            inputs: inputs.clone(),
        })
    }

    pub fn placeholder_id(&self) -> i64 {
        self.inputs.image_placeholder_token_id
    }

    pub fn num_state_tokens(&self) -> usize {
        self.pre.num_state_tokens
    }

    pub fn max_action_dim(&self) -> usize {
        self.inputs.max_action_dim
    }

    /// The tokens one `<|image|>` becomes: single-crop `resize` mode.
    pub fn image_tokens(&self) -> Vec<i64> {
        let q = &self.inputs;
        let [rows, cols] = self.geo.pooled;
        let mut row = vec![q.image_patch_id; cols];
        if let (true, Some(col)) = (q.single_crop_col_tokens, q.image_col_id) {
            row.push(col);
        }
        let mut out = vec![q.image_start_token_id];
        for _ in 0..rows {
            out.extend_from_slice(&row);
        }
        out.push(q.image_end_token_id);
        out
    }

    /// The model's sequence length for a tokenizer `width` wide (BOS added).
    pub fn seq_len(&self, width: usize) -> usize {
        width + 1 + self.geo.cameras.len() * (self.image_tokens().len() - 1)
    }

    /// The prompt text: the normalized state as tokens, the normalized task.
    pub fn prompt(&self, task: &str, state: &[f32]) -> String {
        render_prompt(&self.pre, task, state)
    }

    /// From the tokenizer's ids and mask: BOS prepended (unless already
    /// first), each placeholder expanded, re-padded to its width plus what the
    /// images add; `(input_ids, attention_mask, token_type_ids)`.
    pub fn inputs(
        &self,
        ids: &[i64],
        mask: &[bool],
    ) -> EngineResult<(Vec<i64>, Vec<i64>, Vec<i64>)> {
        let q = &self.inputs;
        let mut valid: Vec<i64> = ids
            .iter()
            .zip(mask)
            .filter(|(_, &m)| m)
            .map(|(&id, _)| id)
            .collect();
        let mut width = ids.len();
        if valid.first() != Some(&q.bos_token_id) {
            valid.insert(0, q.bos_token_id);
            width += 1;
        }
        let pads = width - valid.len();
        let image = self.image_tokens();
        let mut out = Vec::with_capacity(width + 2 * image.len());
        let mut images = 0;
        for &t in &valid {
            if t == q.image_placeholder_token_id {
                images += 1;
                out.extend_from_slice(&image);
            } else {
                out.push(t);
            }
        }
        if images != self.geo.cameras.len() {
            return Err(EngineError::InvalidConfig(format!(
                "the prompt has {images} image placeholders; the model has {} cameras",
                self.geo.cameras.len()
            )));
        }
        let len = out.len();
        let attention: Vec<i64> = (0..len + pads).map(|i| i64::from(i < len)).collect();
        out.resize(len + pads, q.pad_token_id);
        let types = out
            .iter()
            .zip(&attention)
            .map(|(t, &a)| i64::from(q.image_token_ids.contains(t)) * a)
            .collect();
        Ok((out, attention, types))
    }

    /// `images [cameras, patches, patch²·3]` f32 in camera order: `/255`,
    /// `(x − mean)/std`, row-major patches each flattened `(py, px, c)`.
    pub fn images(
        &self,
        layout: &CameraLayout,
        frames: &[(String, Image)],
    ) -> EngineResult<Vec<f32>> {
        let [h, w] = self.geo.hw;
        let p = self.inputs.patch_size;
        let (mean, std) = (self.inputs.image_mean, self.inputs.image_std);
        let mut out = Vec::with_capacity(layout.slots.len() * h * w * 3);
        for slot in &layout.slots {
            let Some(i) = slot else {
                return refuse("molmoact2 has no dummy camera slots");
            };
            let name = &layout.names[*i];
            let Some((_, img)) = frames.iter().find(|(n, _)| n == name) else {
                return Err(EngineError::InvalidConfig(format!(
                    "missing camera {name:?}"
                )));
            };
            if img.height as usize != h || img.width as usize != w || img.data.len() != h * w * 3 {
                return Err(EngineError::InvalidConfig(format!(
                    "camera {name:?} must be {h}x{w} RGB ({} bytes)",
                    h * w * 3
                )));
            }
            for gy in 0..h / p {
                for gx in 0..w / p {
                    for py in 0..p {
                        let row = ((gy * p + py) * w + gx * p) * 3;
                        for (k, &v) in img.data[row..row + p * 3].iter().enumerate() {
                            let c = k % 3;
                            out.push((f32::from(v) / 255.0 - mean[c]) / std[c]);
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// `token_pooling [cameras · pooled, pool_h · pool_w]`: each pooled cell's
    /// patch indices, −1 where the grid was padded, image `k` offset by
    /// `k · patches`.
    pub fn token_pooling(&self) -> Vec<i64> {
        let [rows, cols] = self.geo.patches;
        let [ph, pw] = self.inputs.pooling_size;
        let [prows, pcols] = self.geo.pooled;
        let (pad_h, pad_w) = (prows * ph - rows, pcols * pw - cols);
        let (top, left) = (pad_h / 2, pad_w / 2);
        let index = |y: usize, x: usize| -> i64 {
            match (y.checked_sub(top), x.checked_sub(left)) {
                (Some(r), Some(c)) if r < rows && c < cols => (r * cols + c) as i64,
                _ => -1,
            }
        };
        let mut one = Vec::with_capacity(prows * pcols * ph * pw);
        for cy in 0..prows {
            for cx in 0..pcols {
                for dy in 0..ph {
                    for dx in 0..pw {
                        one.push(index(cy * ph + dy, cx * pw + dx));
                    }
                }
            }
        }
        let per = self.geo.patches_per_image() as i64;
        (0..self.geo.cameras.len() as i64)
            .flat_map(|k| {
                one.iter()
                    .map(move |&i| if i >= 0 { i + k * per } else { i })
            })
            .collect()
    }

    /// `action_dim_is_pad [max_action_dim]`: false for the first `action_dim`.
    pub fn action_dim_is_pad(&self) -> Vec<bool> {
        (0..self.inputs.max_action_dim)
            .map(|i| i >= self.inputs.action_dim)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_text_is_normalized_like_intel() {
        assert_eq!(
            normalize_text("  Task:  Put  the bowl\tin the BASKET!! "),
            "put the bowl in the basket"
        );
        assert_eq!(normalize_text("language_instruction - Open it."), "open it");
        assert_eq!(normalize_text("LANGUAGE INSTRUCTION:open it"), "open it");
        assert_eq!(normalize_text("The task is to pick it up;"), "pick it up");
        assert_eq!(normalize_text("Your  TASK is\nto  go"), "go");
        // One label, then one phrase, each once and only at the start.
        assert_eq!(normalize_text("goal: the task is to stack"), "stack");
        assert_eq!(normalize_text("task: task: x"), "task: x");
        assert_eq!(normalize_text("the task is to"), "the task is to");
        assert_eq!(normalize_text("tasks: x"), "tasks: x");
        assert_eq!(normalize_text("goalkeeper: x"), "goalkeeper: x");
        assert_eq!(normalize_text("İnstruction: x"), "x");
        assert_eq!(normalize_text("Put ...!?"), "put");
        assert_eq!(normalize_text(""), "");
    }

    #[test]
    fn state_tokens_bin_the_clipped_state() {
        assert_eq!(
            discrete_state(&[-1.0, 1.0, 0.0, f32::NAN, f32::INFINITY, -7.0], 256),
            "<state_start><state_0><state_255><state_128><state_128><state_255><state_0><state_end>"
        );
        // 0 maps to 127.5, which rounds half to even.
        assert_eq!(
            discrete_state(&[0.0], 4),
            "<state_start><state_2><state_end>"
        );
        assert_eq!(state_tokens(3), ["<state_0>", "<state_1>", "<state_2>"]);
    }

    #[test]
    fn quantile_steps_are_f32_and_masked() {
        let q = F32Quantiles {
            q01: vec![-0.5, 3.0, 0.0],
            q99: vec![1.5, 3.0, 2.0],
            mask: Some(vec![true, true, false]),
        };
        assert_eq!(normalize(&q, &[0.5, 3.0, 9.0]), vec![0.0, -1.0, 9.0]);
        let back = postprocess(&q, &[1.0, 1.0, 5.0, -3.0, 0.0, -0.25]);
        assert_eq!(
            back,
            vec![1.5, 3.0 + 1e-8, 1.0, -0.5, 3.0 + 1e-8 / 2.0, -0.25]
        );
    }
}
