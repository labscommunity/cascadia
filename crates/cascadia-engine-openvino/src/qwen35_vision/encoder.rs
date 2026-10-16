//! Host-side inputs of the Qwen3.5 vision IRs, ported from transformers
//! 5.14.1 `vision_utils` (what `Qwen3_5MoeVisionModel.forward` calls) and
//! matching how GenAI's Qwen3-VL embedder drives the same three IRs:
//!
//! 1. `vision_embeddings`: `hidden_states [N, C*Tp*P*P]` -> patch embeddings
//!    `[N, D]` (the preprocessed [`super::Patches`]).
//! 2. `vision_embeddings_pos`: `input [4, N]` bilinear corner indices into the
//!    learned `S x S` position table -> `[4, N, D]`; the host sums the four
//!    corners with [`BilinearIndices::weights`] ([`weighted_corner_sum`]) and
//!    adds the result to the patch embeddings.
//! 3. `vision_embeddings_merger`: those `hidden_states [N, D]`,
//!    `rotary_pos_emb [N, R]` ([`rotary_pos_emb`]) and a block-diagonal
//!    `attention_mask [1, N, N]` ([`attention_mask`], one block per frame) ->
//!    `[N / merge^2, out_hidden]`, one row per prompt image token.
//!
//! Every per-patch array is in merge-block order (block row, block column,
//! then the `merge x merge` patches inside), the order of the preprocessed
//! rows and of the merger's output.

/// `torch.linspace(start, end, steps)` for float32 on CPU: the first half
/// counts up from `start`, the second half down from `end`, both with an f32
/// step. Matching it keeps corner indices identical where a coordinate lands
/// on an integer.
fn linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f32> {
    match steps {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (end - start) / (steps - 1) as f32;
            let half = steps / 2;
            (0..steps)
                .map(|i| {
                    if i < half {
                        start + step * i as f32
                    } else {
                        end - step * (steps - i - 1) as f32
                    }
                })
                .collect()
        }
    }
}

/// Patch index of every position in merge-block order, for one frame.
fn merge_order(h: usize, w: usize, m: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..h / m).flat_map(move |hb| {
        (0..w / m).flat_map(move |wb| {
            (0..m).flat_map(move |hs| (0..m).map(move |ws| (hb * m + hs, wb * m + ws)))
        })
    })
}

/// Bilinear interpolation of the learned `side x side` position table onto
/// each grid, as `get_vision_bilinear_indices_and_weights`.
#[derive(Clone, Debug, PartialEq)]
pub struct BilinearIndices {
    /// `[4, N]` row-major: corners (floor,floor), (floor,ceil), (ceil,floor),
    /// (ceil,ceil) as flat indices into the table.
    pub indices: Vec<i64>,
    /// `[4, N]` row-major weights, same corner order.
    pub weights: Vec<f32>,
    pub positions: usize,
}

/// `grids` are `[t, h, w]` in patches; `side` is `sqrt(num_position_embeddings)`.
pub fn bilinear_indices_and_weights(
    grids: &[[u32; 3]],
    side: usize,
    merge: usize,
) -> BilinearIndices {
    let n: usize = grids.iter().map(|g| (g[0] * g[1] * g[2]) as usize).sum();
    let mut indices = vec![0i64; 4 * n];
    let mut weights = vec![0f32; 4 * n];
    let mut at = 0;
    let max = side as i64 - 1;
    for &[t, h, w] in grids {
        let (h, w) = (h as usize, w as usize);
        let hg = linspace_f32(0.0, (side - 1) as f32, h);
        let wg = linspace_f32(0.0, (side - 1) as f32, w);
        for _ in 0..t {
            for (hi, wi) in merge_order(h, w, merge) {
                // `.int()` truncates; the fraction stays in f32.
                let (hf, wf) = (hg[hi] as i64, wg[wi] as i64);
                let (hc, wc) = ((hf + 1).min(max), (wf + 1).min(max));
                let (dh, dw) = (hg[hi] - hf as f32, wg[wi] - wf as f32);
                let s = side as i64;
                indices[at] = hf * s + wf;
                indices[n + at] = hf * s + wc;
                indices[2 * n + at] = hc * s + wf;
                indices[3 * n + at] = hc * s + wc;
                weights[at] = (1.0 - dh) * (1.0 - dw);
                weights[n + at] = (1.0 - dh) * dw;
                weights[2 * n + at] = dh * (1.0 - dw);
                weights[3 * n + at] = dh * dw;
                at += 1;
            }
        }
    }
    BilinearIndices {
        indices,
        weights,
        positions: n,
    }
}

/// `(pos_embed(indices) * weights[:, :, None]).sum(0)`: `corners` is the pos
/// IR's `[4, N, D]` output; returns `[N, D]`.
pub fn weighted_corner_sum(corners: &[f32], weights: &[f32], n: usize, d: usize) -> Vec<f32> {
    assert_eq!(corners.len(), 4 * n * d, "corners is not [4, {n}, {d}]");
    assert_eq!(weights.len(), 4 * n, "weights is not [4, {n}]");
    let mut out = vec![0f32; n * d];
    for c in 0..4 {
        for p in 0..n {
            let wgt = weights[c * n + p];
            let src = &corners[(c * n + p) * d..(c * n + p + 1) * d];
            for (o, &s) in out[p * d..(p + 1) * d].iter_mut().zip(src) {
                *o += s * wgt;
            }
        }
    }
    out
}

/// The vision rotary input `[N, dim]`: per patch, `row * inv_freq` then
/// `col * inv_freq`, where `inv_freq[i] = 1 / theta^(2i / dim)` over
/// `dim / 2` frequencies (`dim` = head_dim / 2 = 36 on Qwen3.5).
pub fn rotary_pos_emb(grids: &[[u32; 3]], merge: usize, dim: usize, theta: f32) -> Vec<f32> {
    let half = dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
        .collect();
    let mut out = Vec::new();
    for &[t, h, w] in grids {
        for _ in 0..t {
            for (hi, wi) in merge_order(h as usize, w as usize, merge) {
                out.extend(inv_freq.iter().map(|f| hi as f32 * f));
                out.extend(inv_freq.iter().map(|f| wi as f32 * f));
            }
        }
    }
    out
}

/// `get_vision_cu_seqlens`: one sequence per frame.
pub fn cu_seqlens(grids: &[[u32; 3]]) -> Vec<i32> {
    let mut out = vec![0i32];
    for &[t, h, w] in grids {
        for _ in 0..t {
            out.push(out.last().unwrap() + (h * w) as i32);
        }
    }
    out
}

/// The merger's `[1, N, N]` mask for [`cu_seqlens`]: 0 inside a frame,
/// `-inf` across frames (GenAI `qwen2_vl_utils::get_attention_mask`).
pub fn attention_mask(grids: &[[u32; 3]]) -> Vec<f32> {
    let cu = cu_seqlens(grids);
    let n = *cu.last().unwrap() as usize;
    let mut mask = vec![f32::NEG_INFINITY; n * n];
    for pair in cu.windows(2) {
        let (s, e) = (pair[0] as usize, pair[1] as usize);
        for row in s..e {
            mask[row * n + s..row * n + e].fill(0.0);
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_hits_both_ends_exactly() {
        let v = linspace_f32(0.0, 47.0, 22);
        assert_eq!((v[0], v[21]), (0.0, 47.0));
        assert_eq!(linspace_f32(0.0, 47.0, 1), vec![0.0]);
    }

    #[test]
    fn weights_sum_to_one_and_corners_stay_in_the_table() {
        let b = bilinear_indices_and_weights(&[[1, 14, 20], [2, 6, 8]], 48, 2);
        let n = b.positions;
        assert_eq!(n, 14 * 20 + 2 * 6 * 8);
        for p in 0..n {
            let s: f32 = (0..4).map(|c| b.weights[c * n + p]).sum();
            assert!((s - 1.0).abs() < 1e-5, "{p}: {s}");
        }
        assert!(b.indices.iter().all(|&i| (0..48 * 48).contains(&i)));
        // The first position is the table's corner, the last block's last
        // patch is the opposite corner.
        assert_eq!(b.indices[0], 0);
        assert_eq!(b.indices[3 * n + 14 * 20 - 1], 48 * 48 - 1);
    }

    #[test]
    fn rotary_layout_is_rows_then_columns() {
        let r = rotary_pos_emb(&[[1, 2, 4]], 2, 36, 10000.0);
        assert_eq!(r.len(), 8 * 36);
        // Merge order: block (0,0) holds patches (0,0),(0,1),(1,0),(1,1),
        // then block (0,1) starts at patch (0,2). Frequency 0 is 1, so the
        // first entry of each half is the row or column itself.
        let at = |patch: usize, half: usize| r[patch * 36 + half * 18];
        assert_eq!((at(0, 0), at(0, 1)), (0.0, 0.0));
        assert_eq!((at(1, 0), at(1, 1)), (0.0, 1.0));
        assert_eq!((at(2, 0), at(2, 1)), (1.0, 0.0));
        assert_eq!((at(4, 0), at(4, 1)), (0.0, 2.0));
    }

    #[test]
    fn mask_is_block_diagonal_per_frame() {
        let g = [[2, 2, 2]];
        assert_eq!(cu_seqlens(&g), vec![0, 4, 8]);
        let m = attention_mask(&g);
        assert_eq!(m[0], 0.0);
        assert_eq!(m[3 * 8 + 3], 0.0);
        assert_eq!(m[3 * 8 + 4], f32::NEG_INFINITY);
        assert_eq!(m[7 * 8 + 4], 0.0);
    }

    #[test]
    fn corner_sum_applies_weights() {
        let corners = [1.0, 2.0, 3.0, 4.0]; // n=1, d=1
        let w = [0.1, 0.2, 0.3, 0.4];
        let s = weighted_corner_sum(&corners, &w, 1, 1);
        assert!((s[0] - 3.0).abs() < 1e-6);
    }
}

/// The `vision_config` fields the encoder needs (`config.json`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionConfig {
    pub num_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
}

impl VisionConfig {
    /// Read `vision_config` from a Hugging Face `config.json`.
    pub fn from_config_json(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))?;
        let vc = v
            .get("vision_config")
            .ok_or("config.json has no vision_config")?;
        let get = |k: &str| {
            vc.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as usize)
                .ok_or_else(|| format!("vision_config.{k} missing"))
        };
        Ok(Self {
            num_position_embeddings: get("num_position_embeddings")?,
            hidden_size: get("hidden_size")?,
            num_heads: get("num_heads")?,
            spatial_merge_size: get("spatial_merge_size")?,
            out_hidden_size: get("out_hidden_size")?,
        })
    }

    pub fn grid_side(&self) -> usize {
        (self.num_position_embeddings as f64).sqrt().round() as usize
    }

    /// `rotary_pos_emb` width: half the attention head dimension.
    pub fn rotary_dim(&self) -> usize {
        self.hidden_size / self.num_heads / 2
    }
}

/// Every stage's output for one item, for parity checks.
#[derive(Clone, Debug)]
pub struct EncodeTrace {
    /// `[N, hidden]` from the patch embedding IR.
    pub patch_embeds: Vec<f32>,
    /// `[N, hidden]` patch embeddings plus interpolated positions.
    pub hidden_states: Vec<f32>,
    /// `[N / merge^2, out_hidden]`, one row per prompt image token.
    pub image_embeds: Vec<f32>,
    /// Qwen3-VL only: the merger's `deepstack_feature_lists`,
    /// `[layers, N / merge^2, out_hidden]` layer-major, the features the
    /// language model adds to its first `layers` decoder layers' output at the
    /// image tokens. `None` for a merger without that output (Qwen3.5).
    pub deepstack: Option<(usize, Vec<f32>)>,
}

/// The export's three vision IRs (`openvino_vision_embeddings{,_pos,_merger}_model.xml`)
/// driven with the host inputs above.
pub struct VisionEncoder {
    patch: cascadia_ov_genai_shim::Runtime,
    pos: cascadia_ov_genai_shim::Runtime,
    merger: cascadia_ov_genai_shim::Runtime,
    /// Output index of the merger's image rows (`last_hidden_state`).
    image_out: usize,
    /// Output index of the merger's `deepstack_feature_lists` (Qwen3-VL).
    deepstack_out: Option<usize>,
    cfg: VisionConfig,
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_f32(
    rt: &cascadia_ov_genai_shim::Runtime,
    what: &str,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    read_f32_at(rt, 0, what)
}

pub(crate) fn read_f32_at(
    rt: &cascadia_ov_genai_shim::Runtime,
    idx: usize,
    what: &str,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    let (dtype, shape, bytes) = rt.output(idx).map_err(|e| format!("{what}: {e}"))?;
    if dtype != cascadia_ov_genai_shim::DType::F32 {
        return Err(format!("{what}: output is {dtype:?}, expected f32"));
    }
    let v = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok((shape, v))
}

impl VisionEncoder {
    /// Compile the three IRs from `model_dir` on `device`.
    pub fn load(
        model_dir: &std::path::Path,
        device: &str,
        plugin: &cascadia_ov_genai_shim::PluginConfig,
        cfg: VisionConfig,
    ) -> Result<Self, String> {
        let compile = |suffix: &str| {
            let path = model_dir.join(format!("openvino_vision_embeddings{suffix}_model.xml"));
            cascadia_ov_genai_shim::Runtime::compile(&path.to_string_lossy(), device, plugin)
                .map_err(|e| format!("compiling {}: {e}", path.display()))
        };
        let merger = compile("_merger")?;
        let (mut image_out, mut deepstack_out) = (None, None);
        for i in 0..merger.output_count() {
            let names = merger
                .output_aliases(i)
                .map_err(|e| format!("merger output {i}: {e}"))?;
            if names.iter().any(|n| n.contains("deepstack")) {
                deepstack_out = Some(i);
            } else if image_out.is_none() {
                image_out = Some(i);
            }
        }
        let image_out = image_out.unwrap_or(0);
        Ok(Self {
            patch: compile("")?,
            pos: compile("_pos")?,
            merger,
            image_out,
            deepstack_out,
            cfg,
        })
    }

    /// Whether the merger also returns DeepStack features (Qwen3-VL).
    pub fn has_deepstack(&self) -> bool {
        self.deepstack_out.is_some()
    }

    /// Image embeddings `[tokens, out_hidden]` for one preprocessed item.
    pub fn encode(&mut self, p: &super::Patches) -> Result<Vec<f32>, String> {
        Ok(self.encode_traced(p)?.image_embeds)
    }

    /// [`Self::encode`], keeping every stage's output.
    pub fn encode_traced(&mut self, p: &super::Patches) -> Result<EncodeTrace, String> {
        use cascadia_ov_genai_shim::DType;
        let (n, d) = (p.rows, self.cfg.hidden_size);
        let grids = [p.grid_thw];
        let merge = self.cfg.spatial_merge_size;

        self.patch
            .set_input(
                "hidden_states",
                DType::F32,
                &[p.rows, p.cols],
                &f32_bytes(&p.pixel_values),
            )
            .and_then(|_| self.patch.infer())
            .map_err(|e| format!("patch embedding: {e}"))?;
        let (shape, patch_embeds) = read_f32(&self.patch, "patch embedding")?;
        if shape != [n, d] {
            return Err(format!(
                "patch embedding returned {shape:?}, expected [{n}, {d}]"
            ));
        }

        let b = bilinear_indices_and_weights(&grids, self.cfg.grid_side(), merge);
        let idx: Vec<u8> = b.indices.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.pos
            .set_input("input", DType::I64, &[4, n], &idx)
            .and_then(|_| self.pos.infer())
            .map_err(|e| format!("position embedding: {e}"))?;
        let (shape, corners) = read_f32(&self.pos, "position embedding")?;
        if shape != [4, n, d] {
            return Err(format!(
                "position embedding returned {shape:?}, expected [4, {n}, {d}]"
            ));
        }
        let mut hidden_states = weighted_corner_sum(&corners, &b.weights, n, d);
        for (h, pe) in hidden_states.iter_mut().zip(&patch_embeds) {
            *h += pe;
        }

        let rot_dim = self.cfg.rotary_dim();
        let rot = rotary_pos_emb(&grids, merge, rot_dim, 10_000.0);
        let mask = attention_mask(&grids);
        self.merger
            .set_input(
                "hidden_states",
                DType::F32,
                &[n, d],
                &f32_bytes(&hidden_states),
            )
            .and_then(|_| {
                self.merger.set_input(
                    "rotary_pos_emb",
                    DType::F32,
                    &[n, rot_dim],
                    &f32_bytes(&rot),
                )
            })
            .and_then(|_| {
                self.merger
                    .set_input("attention_mask", DType::F32, &[1, n, n], &f32_bytes(&mask))
            })
            .and_then(|_| self.merger.infer())
            .map_err(|e| format!("merger: {e}"))?;
        let (shape, image_embeds) = read_f32_at(&self.merger, self.image_out, "merger")?;
        let tokens = n / (merge * merge);
        let out = self.cfg.out_hidden_size;
        if shape != [tokens, out] {
            return Err(format!(
                "merger returned {shape:?}, expected [{tokens}, {out}]"
            ));
        }
        let deepstack = match self.deepstack_out {
            Some(i) => {
                let (shape, v) = read_f32_at(&self.merger, i, "merger deepstack")?;
                match shape[..] {
                    [layers, t, h] if t == tokens && h == out && layers > 0 => Some((layers, v)),
                    _ => {
                        return Err(format!(
                        "merger deepstack returned {shape:?}, expected [layers, {tokens}, {out}]"
                    ))
                    }
                }
            }
            None => None,
        };
        Ok(EncodeTrace {
            patch_embeds,
            hidden_states,
            image_embeds,
            deepstack,
        })
    }
}
