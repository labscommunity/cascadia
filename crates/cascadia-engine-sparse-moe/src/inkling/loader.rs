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
//! `attn.k_sconv.weight`, `mlp.gate.weight`, ...). Projections and the edge
//! tables are held as bf16 bits (the checkpoint dtype, lossless), read
//! straight off the file payload ([`StFile::bf16_bits`] — no f32 transient on
//! the 2.3 GiB embed / unembed tables or the ~36 GB of shells); norms, convs,
//! the relative-bias bank and the router stay f32. Expert bins are the
//! glm/dsv4 int4 group-32 contract and load through the same
//! [`load_expert_bin`] (eager f32 for tiny/dev exports, mmap for the real
//! model).

use std::path::Path;

use serde::Deserialize;

use super::attn::{AttentionLayer, AttnDims, AttnWeights};
use super::conv::ShortConv;
use super::ep::expert_home;
use super::ffn::AnyExpert;
use super::model::{Head, Layer, LayerMlp, Model, WideTable};
use super::moe::{DenseMlp, MoeLayer, MoeWeights};
use super::relpos::RelPos;
use crate::dsv4::loader::{ExpertsMode, LoadError};
use crate::dsv4::st::StFile;
use crate::glm::loader::load_expert_bin;

/// Which of a MoE layer's experts a load opens. Expert ids run `0..n_routed`
/// for the routed experts and `n_routed + s` for the `n_shared` shared ones
/// (the expert-parallel placement treats both alike — `docs/perf/INKLING_SCALING.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertSet {
    /// Every routed + shared expert (single-process / pipeline stage).
    All,
    /// No expert bins at all: the layer holds its router only (the
    /// expert-parallel DRIVER — experts are dispatched to workers).
    None,
    /// The ids homed on worker `index` of `count`
    /// ([`expert_home`]`(id, count) == index`) — what
    /// [`super::ep::load_expert_bank`] opens for a worker. A layer shell never
    /// holds a partial table, so [`load_layer`] rejects this variant.
    Shard { index: u32, count: u32 },
}

impl ExpertSet {
    /// Whether expert `id` (routed or shared) is in this set.
    pub fn owns(&self, id: usize) -> bool {
        match *self {
            ExpertSet::All => true,
            ExpertSet::None => false,
            ExpertSet::Shard { index, count } => expert_home(id, count as usize) == index as usize,
        }
    }
}

/// `CASCADIA_INKLING_PIN_EXPERTS`: `mlock` every mmap'd expert as it is
/// opened, so the whole export is wired in RAM and each GEMV runs off the
/// mapping without page faults — the RAM-resident ("record") mode for a box
/// whose memory holds the export (macOS in particular re-faults file-backed
/// pages it has already cached, which costs more than the GEMV itself).
/// Best-effort: the first expert the OS refuses to lock (`RLIMIT_MEMLOCK`,
/// wired-memory limit) is reported once and the rest stay page-cache backed.
pub(crate) fn pin_experts() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| super::env_flag("CASCADIA_INKLING_PIN_EXPERTS"))
}

/// [`pin_experts`] for one just-opened expert; returns the bytes wired.
fn maybe_pin(x: &AnyExpert, what: &str) -> usize {
    use std::sync::atomic::{AtomicBool, Ordering};
    static FAILED: AtomicBool = AtomicBool::new(false);
    if !pin_experts() || FAILED.load(Ordering::Relaxed) {
        return 0;
    }
    match x.as_mmap() {
        Some(m) => match m.pin() {
            Ok(()) => m.bin_len(),
            Err(e) => {
                if !FAILED.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "[inkling] CASCADIA_INKLING_PIN_EXPERTS: mlock of {what} failed ({e}); \
                         leaving the remaining experts to the page cache"
                    );
                }
                0
            }
        },
        None => 0,
    }
}

/// Open the expert bins of MoE layer `li` that `experts` owns, as
/// `(expert id, expert)` in ascending id order — routed ids `0..n_routed`
/// from `expert_EEE.bin`, shared ids `n_routed + s` from
/// `expert_shared{s}.bin`. `li` must be a MoE layer (not in `dense_layers`).
pub fn load_moe_experts(
    dir: &Path,
    m: &InklingManifest,
    li: usize,
    mode: ExpertsMode,
    experts: ExpertSet,
) -> Result<Vec<(usize, AnyExpert)>, LoadError> {
    if m.dense_layers.contains(&li) {
        return Err(LoadError::Manifest(format!(
            "layer {li} is dense; it has no MoE experts"
        )));
    }
    if let ExpertSet::Shard { index, count } = experts {
        if count == 0 || index >= count {
            return Err(LoadError::Manifest(format!(
                "expert shard index {index} of {count} is out of range"
            )));
        }
    }
    let (hidden, inter) = (m.hidden_size, m.moe_intermediate);
    let edir = dir.join("experts").join(format!("layer_{li:02}"));
    let mut out = Vec::new();
    let mut wired = 0usize;
    for e in 0..m.num_experts {
        if experts.owns(e) {
            let x = load_expert_bin(
                &edir.join(format!("expert_{e:03}.bin")),
                hidden,
                inter,
                mode,
            )?;
            wired += maybe_pin(&x, &format!("layer {li} expert {e}"));
            out.push((e, x));
        }
    }
    for s in 0..m.n_shared_experts {
        let id = m.num_experts + s;
        if experts.owns(id) {
            let x = load_expert_bin(
                &edir.join(format!("expert_shared{s}.bin")),
                hidden,
                inter,
                mode,
            )?;
            wired += maybe_pin(&x, &format!("layer {li} shared expert {s}"));
            out.push((id, x));
        }
    }
    if wired > 0 {
        eprintln!(
            "[inkling] layer {li}: mlock'd {} experts ({:.1} GB)",
            out.len(),
            wired as f64 / 1e9
        );
    }
    Ok(out)
}

/// The subset of `manifest.json` the Inkling shell needs (`PORT_SPEC.md` §1).
///
/// `route_scale`, `log_scaling_alpha` and `logits_mup_width_multiplier` are
/// REQUIRED: they scale the routed weights, the global layers' queries and the
/// final hidden, and a manifest silently defaulting them runs coherent-looking
/// garbage. `export_inkling.py` always writes them.
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
    /// Multiplies routed weights and shared gammas (8.0 on the released models).
    pub route_scale: f32,
    pub rms_norm_eps: f32,
    /// `None` → no log scaling on global layers.
    #[serde(default)]
    pub log_scaling_n_floor: Option<f32>,
    /// Log-scaling slope (only used with `log_scaling_n_floor`, but required
    /// so a manifest cannot silently zero it).
    pub log_scaling_alpha: f32,
    /// The final hidden is divided by this before the unembed (24.0 released).
    pub logits_mup_width_multiplier: f32,
    #[serde(default = "four")]
    pub conv_kernel_size: usize,
    #[serde(default)]
    pub eos_token_ids: Vec<u32>,
    #[serde(default)]
    pub has_mtp: bool,
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
    if m.moe_intermediate == 0 {
        return Err(LoadError::Manifest("moe_intermediate must be > 0".into()));
    }
    if !m.dense_layers.is_empty() && m.dense_intermediate == 0 {
        return Err(LoadError::Manifest(format!(
            "dense_intermediate must be > 0 when dense_layers is non-empty ({:?})",
            m.dense_layers
        )));
    }
    if m.logits_mup_width_multiplier.is_nan() || m.logits_mup_width_multiplier <= 0.0 {
        return Err(LoadError::Manifest(format!(
            "logits_mup_width_multiplier must be > 0, got {}",
            m.logits_mup_width_multiplier
        )));
    }
    if !m.hidden_size.is_multiple_of(32) || !m.moe_intermediate.is_multiple_of(32) {
        return Err(LoadError::Manifest(
            "hidden_size and moe_intermediate must be multiples of the int4 group (32)".into(),
        ));
    }
    if !m.dense_layers.is_empty() && !m.dense_intermediate.is_multiple_of(32) {
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
/// `experts` selects the MoE layers' expert bins: [`ExpertSet::All`] opens
/// every routed + shared expert, [`ExpertSet::None`] builds the MoE block
/// with its router only (the expert-parallel driver attaches an
/// [`super::ep::EpClient`] afterwards); [`ExpertSet::Shard`] is rejected —
/// a shell never holds a partial expert table (workers use
/// [`super::ep::load_expert_bank`], which opens no shells at all).
pub fn load_layer(
    dir: &Path,
    m: &InklingManifest,
    li: usize,
    max_seq: usize,
    mode: ExpertsMode,
    experts: ExpertSet,
) -> Result<Layer, LoadError> {
    let (hidden, eps, k) = (m.hidden_size, m.rms_norm_eps, m.conv_kernel_size);
    let st = StFile::open(&dir.join(format!("shells/layer_{li:02}.safetensors")))?;
    let g = |n: &str| st.f32(n).map(|t| t.1);
    // bf16 projections: the file payload as-is (no widen / narrow round trip).
    let gb = |n: &str| st.bf16_bits(n).map(|t| t.1);

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
        maybe_pin(&w, &format!("layer {li} dense MLP"));
        let gs = g("mlp.global_scale")?;
        LayerMlp::Dense(DenseMlp::new(
            w,
            m.dense_intermediate,
            gs.first().copied().unwrap_or(1.0),
        ))
    } else {
        let inter = m.moe_intermediate;
        let (experts, shared) = match experts {
            ExpertSet::All => {
                let mut routed: Vec<AnyExpert> = Vec::with_capacity(m.num_experts);
                let mut shared = Vec::with_capacity(m.n_shared_experts);
                for (id, x) in load_moe_experts(dir, m, li, mode, ExpertSet::All)? {
                    if id < m.num_experts {
                        routed.push(x);
                    } else {
                        shared.push(x);
                    }
                }
                (routed, shared)
            }
            ExpertSet::None => (Vec::new(), Vec::new()),
            ExpertSet::Shard { index, count } => {
                return Err(LoadError::Manifest(format!(
                    "layer {li}: a layer shell cannot hold expert shard {index}/{count}; \
                     expert workers load their shard with inkling::ep::load_expert_bank"
                )));
            }
        };
        let gs = g("mlp.gate.global_scale")?;
        let router_w = g("mlp.gate.weight")?;
        let router_bias = g("mlp.gate.bias")?;
        if router_bias.len() != m.num_experts
            || router_w.len() != (m.num_experts + m.n_shared_experts) * hidden
        {
            return Err(LoadError::Manifest(format!(
                "layer {li}: router shapes (bias {}, weight {}) do not match {} routed + {} shared experts × hidden {hidden}",
                router_bias.len(),
                router_w.len(),
                m.num_experts,
                m.n_shared_experts
            )));
        }
        let mw = MoeWeights {
            router_w,
            router_bias,
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
    pub head: Option<Head>,
    pub manifest: InklingManifest,
}

/// Read a `[vocab, hidden]` edge table as bf16 bits (the checkpoint dtype —
/// lossless, half the RAM of f32, and copied straight off the file payload).
fn wide_table(
    st: &StFile,
    name: &str,
    vocab: usize,
    hidden: usize,
) -> Result<WideTable, LoadError> {
    let (shape, v) = st.bf16_bits(name)?;
    if v.len() != vocab * hidden {
        return Err(LoadError::Manifest(format!(
            "{name}: shape {shape:?} != [{vocab}, {hidden}]"
        )));
    }
    Ok(WideTable::Bf16(v))
}

/// Load the layer slice `[lo, hi)`. Reads embed only when `first`, head only
/// when `last`. `max_seq` sizes the global layers' KV caches. `experts` is
/// passed to every [`load_layer`] (`All` for a self-contained stage, `None`
/// for an expert-parallel driver).
#[allow(clippy::too_many_arguments)]
pub fn load_stage(
    dir: &Path,
    max_seq: usize,
    lo: usize,
    hi: usize,
    first: bool,
    last: bool,
    mode: ExpertsMode,
    experts: ExpertSet,
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
        let norm = h.f32("norm.weight")?.1;
        if norm.len() != hidden {
            return Err(LoadError::Manifest(format!(
                "norm.weight has {} entries, expected hidden_size {hidden}",
                norm.len()
            )));
        }
        Some(Head::new(
            norm,
            wide_table(&h, "unembed.weight", vocab, hidden)?,
            m.rms_norm_eps,
            m.logits_mup_width_multiplier,
            m.unpadded_vocab(),
        ))
    } else {
        None
    };
    let mut layers = Vec::with_capacity(hi.saturating_sub(lo));
    for li in lo..hi {
        layers.push(load_layer(dir, &m, li, max_seq, mode, experts)?);
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
    let s = load_stage(
        dir,
        max_seq,
        0,
        m.num_layers,
        true,
        true,
        mode,
        ExpertSet::All,
    )?;
    let (embed, embed_norm) = s.embed.expect("full model has an embed");
    let Head { norm, unembed, .. } = s.head.expect("full model has a head");
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
