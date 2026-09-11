//! Inkling loader — builds layers / stages / a full [`Model`] from the layout
//! emitted by `tools/export_inkling.py`:
//!   <dir>/manifest.json                                  (arch == "inkling")
//!   <dir>/embed.safetensors                              embed.weight (bf16), embed_norm.weight (f32)
//!   <dir>/head.safetensors                               unembed.weight (bf16), norm.weight (f32)
//!   <dir>/shells/layer_NN.safetensors                    attention + norms + convs (+ router on MoE layers)
//!   <dir>/experts/layer_NN/expert_EEE.bin                routed experts (int4_bin, gate/up/down)
//!   <dir>/experts/layer_NN/expert_shared{0,1}.bin        the two shared experts (one bin each)
//!   <dir>/experts/layer_NN/dense.bin                     dense-layer MLP (`dense_layers`)
//!
//! Tensor names inside the safetensors are the checkpoint's own with the
//! `model.llm.` / `model.llm.layers.N.` prefix stripped (`attn.wq_du.weight`,
//! `attn.k_sconv.weight`, `mlp.gate.weight`, ...). Projections are held as
//! bf16 bits (the checkpoint dtype, lossless); norms, convs, the relative-bias
//! bank and the router stay f32. Expert bins are the glm/dsv4 int4 group-32
//! contract and load through the same [`load_expert_bin`] (eager f32 for
//! tiny/dev exports, mmap for the real model).

use std::path::Path;

use half::bf16;
use serde::Deserialize;

use super::attn::{AttentionLayer, AttnDims, AttnWeights};
use super::conv::ShortConv;
use super::ffn::AnyExpert;
use super::model::{Layer, LayerMlp, Model, WideTable};
use super::moe::{DenseMlp, MoeLayer, MoeWeights};
use super::relpos::RelPos;
use crate::dsv4::loader::{ExpertsMode, LoadError};
use crate::dsv4::st::StFile;
use crate::glm::loader::load_expert_bin;

/// The subset of `manifest.json` the Inkling shell needs (`PORT_SPEC.md` §1).
#[derive(Debug, Clone, Deserialize)]
pub struct InklingManifest {
    pub arch: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    /// Logits are sliced to this many entries; defaults to `vocab_size`.
    #[serde(default)]
    pub unpadded_vocab_size: Option<usize>,
    /// Global-layer attention shape.
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Sliding-layer attention shape (defaults to the global values).
    #[serde(default)]
    pub swa_num_attention_heads: Option<usize>,
    #[serde(default)]
    pub swa_num_kv_heads: Option<usize>,
    #[serde(default)]
    pub swa_head_dim: Option<usize>,
    pub d_rel: usize,
    /// Bias range on global layers; sliding layers use `sliding_window`.
    pub rel_extent: usize,
    pub sliding_window: usize,
    /// `"sliding"` | `"global"` per layer.
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub dense_layers: Vec<usize>,
    #[serde(default)]
    pub dense_intermediate: usize,
    pub moe_intermediate: usize,
    pub num_experts: usize,
    pub top_k: usize,
    #[serde(default = "two")]
    pub n_shared_experts: usize,
    #[serde(default = "one_f32")]
    pub route_scale: f32,
    pub rms_norm_eps: f32,
    /// `None` → no log scaling on global layers.
    #[serde(default)]
    pub log_scaling_n_floor: Option<f32>,
    #[serde(default)]
    pub log_scaling_alpha: f32,
    #[serde(default = "one_f32")]
    pub logits_mup_width_multiplier: f32,
    #[serde(default = "four")]
    pub conv_kernel_size: usize,
    #[serde(default)]
    pub eos_token_ids: Vec<u32>,
    #[serde(default)]
    pub has_mtp: bool,
}

fn one_f32() -> f32 {
    1.0
}
fn two() -> usize {
    2
}
fn four() -> usize {
    4
}

impl InklingManifest {
    pub fn unpadded_vocab(&self) -> usize {
        self.unpadded_vocab_size.unwrap_or(self.vocab_size)
    }
    pub fn is_sliding(&self, li: usize) -> bool {
        self.layer_types
            .get(li)
            .map(|t| t == "sliding")
            .unwrap_or(false)
    }
    /// `(n_heads, n_kv_heads, head_dim)` for layer `li`.
    pub fn attn_shape(&self, li: usize) -> (usize, usize, usize) {
        if self.is_sliding(li) {
            (
                self.swa_num_attention_heads
                    .unwrap_or(self.num_attention_heads),
                self.swa_num_kv_heads.unwrap_or(self.num_kv_heads),
                self.swa_head_dim.unwrap_or(self.head_dim),
            )
        } else {
            (self.num_attention_heads, self.num_kv_heads, self.head_dim)
        }
    }
}

pub fn read_manifest(dir: &Path) -> Result<InklingManifest, LoadError> {
    let m: InklingManifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)
            .map_err(|e| LoadError::Manifest(e.to_string()))?;
    if m.arch != "inkling" {
        return Err(LoadError::Manifest(format!(
            "arch is {:?}, expected \"inkling\"",
            m.arch
        )));
    }
    if m.layer_types.len() != m.num_layers {
        return Err(LoadError::Manifest(format!(
            "layer_types has {} entries for {} layers",
            m.layer_types.len(),
            m.num_layers
        )));
    }
    if let Some(bad) = m
        .layer_types
        .iter()
        .find(|t| t.as_str() != "sliding" && t.as_str() != "global")
    {
        return Err(LoadError::Manifest(format!(
            "unknown layer type {bad:?} (expected \"sliding\" | \"global\")"
        )));
    }
    if m.hidden_size % 32 != 0 || m.moe_intermediate % 32 != 0 {
        return Err(LoadError::Manifest(
            "hidden_size and moe_intermediate must be multiples of the int4 group (32)".into(),
        ));
    }
    if !m.dense_layers.is_empty() && m.dense_intermediate % 32 != 0 {
        return Err(LoadError::Manifest(
            "dense_intermediate must be a multiple of the int4 group (32)".into(),
        ));
    }
    if m.top_k == 0 || m.top_k > m.num_experts {
        return Err(LoadError::Manifest(format!(
            "top_k {} vs num_experts {}",
            m.top_k, m.num_experts
        )));
    }
    Ok(m)
}

fn to_bf16_bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| bf16::from_f32(*x).to_bits()).collect()
}

/// A conv weight tensor as exported (`[C, K]` or the checkpoint's `[C, 1, K]`).
fn conv_from(st: &StFile, name: &str, k: usize) -> Result<ShortConv, LoadError> {
    let (shape, w) = st.f32(name)?;
    let c = shape.first().copied().unwrap_or(0);
    if c == 0 || w.len() != c * k {
        return Err(LoadError::Manifest(format!(
            "{name}: shape {shape:?} is not [C, {k}]"
        )));
    }
    Ok(ShortConv::new(w, c, k))
}

/// Build one transformer layer `li` from its shell safetensors + expert bins.
pub fn load_layer(
    dir: &Path,
    m: &InklingManifest,
    li: usize,
    max_seq: usize,
    mode: ExpertsMode,
) -> Result<Layer, LoadError> {
    let (hidden, eps, k) = (m.hidden_size, m.rms_norm_eps, m.conv_kernel_size);
    let st = StFile::open(&dir.join(format!("shells/layer_{li:02}.safetensors")))?;
    let g = |n: &str| st.f32(n).map(|t| t.1);
    let gb = |n: &str| -> Result<Vec<u16>, LoadError> { Ok(to_bf16_bits(&g(n)?)) };

    let (hq, hkv, d) = m.attn_shape(li);
    let mut dims = if m.is_sliding(li) {
        AttnDims::sliding(hidden, hq, hkv, d, m.d_rel, m.sliding_window, eps)
    } else {
        AttnDims::global(hidden, hq, hkv, d, m.d_rel, max_seq, eps)
    };
    if let Some(n_floor) = m.log_scaling_n_floor {
        dims = dims.with_log_scaling(n_floor, m.log_scaling_alpha);
    }
    let aw = AttnWeights {
        wq: gb("attn.wq_du.weight")?,
        wk: gb("attn.wk_dv.weight")?,
        wv: gb("attn.wv_dv.weight")?,
        wr: gb("attn.wr_du.weight")?,
        wo: gb("attn.wo_ud.weight")?,
        q_norm: g("attn.q_norm.weight")?,
        k_norm: g("attn.k_norm.weight")?,
    };
    let (pshape, proj) = st.f32("attn.rel_logits_proj.proj")?;
    if pshape.len() != 2 || pshape[0] != m.d_rel {
        return Err(LoadError::Manifest(format!(
            "layer {li}: rel_logits_proj.proj shape {pshape:?}, expected [{}, extent]",
            m.d_rel
        )));
    }
    let extent = pshape[1];
    let want_extent = if m.is_sliding(li) {
        m.sliding_window
    } else {
        m.rel_extent
    };
    if extent != want_extent {
        return Err(LoadError::Manifest(format!(
            "layer {li}: rel_logits_proj extent {extent} != {want_extent}"
        )));
    }
    let attn = AttentionLayer::from_parts(
        dims,
        aw,
        conv_from(&st, "attn.k_sconv.weight", k)?,
        conv_from(&st, "attn.v_sconv.weight", k)?,
        RelPos::new(proj, m.d_rel, extent),
    );

    let edir = dir.join("experts").join(format!("layer_{li:02}"));
    let mlp = if m.dense_layers.contains(&li) {
        let w = load_expert_bin(&edir.join("dense.bin"), hidden, m.dense_intermediate, mode)?;
        let gs = g("mlp.global_scale")?;
        LayerMlp::Dense(DenseMlp::new(
            w,
            m.dense_intermediate,
            gs.first().copied().unwrap_or(1.0),
        ))
    } else {
        let inter = m.moe_intermediate;
        let mut experts: Vec<AnyExpert> = Vec::with_capacity(m.num_experts);
        for e in 0..m.num_experts {
            experts.push(load_expert_bin(
                &edir.join(format!("expert_{e:03}.bin")),
                hidden,
                inter,
                mode,
            )?);
        }
        let mut shared = Vec::with_capacity(m.n_shared_experts);
        for s in 0..m.n_shared_experts {
            shared.push(load_expert_bin(
                &edir.join(format!("expert_shared{s}.bin")),
                hidden,
                inter,
                mode,
            )?);
        }
        let gs = g("mlp.gate.global_scale")?;
        let mw = MoeWeights {
            router_w: g("mlp.gate.weight")?,
            router_bias: g("mlp.gate.bias")?,
            global_scale: gs.first().copied().unwrap_or(1.0),
            experts,
            shared,
        };
        LayerMlp::Moe(MoeLayer::new(hidden, inter, m.top_k, m.route_scale, mw))
    };

    Ok(Layer::new(
        hidden,
        eps,
        g("attn_norm.weight")?,
        attn,
        conv_from(&st, "attn_sconv.weight", k)?,
        g("mlp_norm.weight")?,
        mlp,
        conv_from(&st, "mlp_sconv.weight", k)?,
    ))
}

/// One pipeline stage: `embed` (+ `embed_norm`) present iff this rank owns
/// layer 0, `head` (final norm, unembed) iff it owns the last layer, and the
/// layer slice `[lo, hi)`.
pub struct InklingStage {
    pub embed: Option<(WideTable, Vec<f32>)>,
    pub layers: Vec<Layer>,
    pub head: Option<(Vec<f32>, WideTable)>,
    pub manifest: InklingManifest,
}

/// Read a `[vocab, hidden]` edge table as bf16 bits (the checkpoint dtype —
/// lossless, half the RAM of f32).
fn wide_table(
    st: &StFile,
    name: &str,
    vocab: usize,
    hidden: usize,
) -> Result<WideTable, LoadError> {
    let (shape, v) = st.f32(name)?;
    if v.len() != vocab * hidden {
        return Err(LoadError::Manifest(format!(
            "{name}: shape {shape:?} != [{vocab}, {hidden}]"
        )));
    }
    Ok(WideTable::Bf16(to_bf16_bits(&v)))
}

/// Load the layer slice `[lo, hi)`. Reads embed only when `first`, head only
/// when `last`. `max_seq` sizes the global layers' KV caches.
pub fn load_stage(
    dir: &Path,
    max_seq: usize,
    lo: usize,
    hi: usize,
    first: bool,
    last: bool,
    mode: ExpertsMode,
) -> Result<InklingStage, LoadError> {
    let m = read_manifest(dir)?;
    let (vocab, hidden) = (m.vocab_size, m.hidden_size);
    let embed = if first {
        let e = StFile::open(&dir.join("embed.safetensors"))?;
        Some((
            wide_table(&e, "embed.weight", vocab, hidden)?,
            e.f32("embed_norm.weight")?.1,
        ))
    } else {
        None
    };
    let head = if last {
        let h = StFile::open(&dir.join("head.safetensors"))?;
        Some((
            h.f32("norm.weight")?.1,
            wide_table(&h, "unembed.weight", vocab, hidden)?,
        ))
    } else {
        None
    };
    let mut layers = Vec::with_capacity(hi.saturating_sub(lo));
    for li in lo..hi {
        layers.push(load_layer(dir, &m, li, max_seq, mode)?);
    }
    Ok(InklingStage {
        embed,
        layers,
        head,
        manifest: m,
    })
}

/// Load a full single-stage model with eager (dequantized f32) experts — the
/// tiny/dev path and the single-process oracle the staged runner is validated
/// against.
pub fn load_model(dir: &Path, max_seq: usize) -> Result<Model, LoadError> {
    load_model_with(dir, max_seq, ExpertsMode::Eager)
}

pub fn load_model_with(dir: &Path, max_seq: usize, mode: ExpertsMode) -> Result<Model, LoadError> {
    let m = read_manifest(dir)?;
    let s = load_stage(dir, max_seq, 0, m.num_layers, true, true, mode)?;
    let (embed, embed_norm) = s.embed.expect("full model has an embed");
    let (norm, unembed) = s.head.expect("full model has a head");
    Ok(Model::new(
        m.hidden_size,
        m.vocab_size,
        m.unpadded_vocab(),
        m.rms_norm_eps,
        m.logits_mup_width_multiplier,
        embed,
        embed_norm,
        s.layers,
        norm,
        unembed,
    ))
}
