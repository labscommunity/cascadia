//! dsv4 loader — builds a [`DsV4Model`] from the layout emitted by
//! `tools/export_deepseek_v4.py`:
//!
//! ```text
//! <dir>/manifest.json
//! <dir>/embed.safetensors            embed.weight
//! <dir>/head.safetensors             head.weight, norm.weight, hc_head_*
//! <dir>/shells/layer_NN.safetensors  attn/gate/hc tensors (layer-relative names)
//! <dir>/experts/layer_NN/expert_{EEE,shared}.bin   int4_bin experts
//! ```
//!
//! Experts load in one of two [`ExpertsMode`]s: `Eager` (dequantized to f32
//! up front — tiny/dev only) or `Mmap` (int4 stays packed on disk, rows
//! dequantized per forward — required at real-model scale; see
//! `expert_mmap`).

use std::path::Path;

use half::bf16;
use serde::Deserialize;

use super::attn_layer::{AttentionLayer, AttnWeights};
use super::compress::{Compressor, CompressorWeights};
use super::expert_mmap::MmapExpert;
use super::indexer::{Indexer, IndexerWeights};
use super::model::{AnyExpert, DsV4Model, Expert, GateWeights, HcParams, Head, Layer, ModelCfg};
use super::rope::precompute_freqs;
use super::st::{StError, StFile};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest: {0}")]
    Manifest(String),
    #[error("safetensors: {0}")]
    St(#[from] StError),
    #[error("expert bin {0}: bad size {1}")]
    ExpertBin(String, usize),
}

/// The exporter's manifest contract (subset the shell needs).
#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub arch: String,
    pub num_layers: usize,
    pub exported_layers: Vec<usize>,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub eos_token_ids: Vec<i64>,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub window_size: usize,
    pub compress_ratios: Vec<usize>,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_seq_len: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub rms_norm_eps: f32,
    pub n_routed_experts: usize,
    pub top_k: usize,
    pub score_func: String,
    pub route_scale: f32,
    pub n_hash_layers: usize,
    pub swiglu_limit: f32,
    pub expert_intermediate: usize,
    pub experts_format: String,
}

/// Dequantize one int4_bin expert matrix section: packed nibbles
/// `[out, in/2]` (low nibble = even column, value = (nibble - 8) * scale)
/// followed by bf16-LE per-32-column-group scales `[out, in/32]`.
fn dequant_int4(
    buf: &[u8],
    off: &mut usize,
    out_dim: usize,
    in_dim: usize,
    name: &str,
) -> Result<Vec<f32>, LoadError> {
    const G: usize = 32;
    let packed_len = out_dim * in_dim / 2;
    let scale_len = out_dim * (in_dim / G) * 2;
    if buf.len() < *off + packed_len + scale_len {
        return Err(LoadError::ExpertBin(name.into(), buf.len()));
    }
    let packed = &buf[*off..*off + packed_len];
    let scales_raw = &buf[*off + packed_len..*off + packed_len + scale_len];
    *off += packed_len + scale_len;
    let ng = in_dim / G;
    let mut w = vec![0.0f32; out_dim * in_dim];
    for o in 0..out_dim {
        for g in 0..ng {
            let s = bf16::from_le_bytes([
                scales_raw[(o * ng + g) * 2],
                scales_raw[(o * ng + g) * 2 + 1],
            ])
            .to_f32();
            for i in 0..G / 2 {
                let byte = packed[o * in_dim / 2 + g * G / 2 + i];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                w[o * in_dim + g * G + 2 * i] = lo as f32 * s;
                w[o * in_dim + g * G + 2 * i + 1] = hi as f32 * s;
            }
        }
    }
    Ok(w)
}

fn load_expert_bin(path: &Path, dim: usize, inter: usize) -> Result<Expert, LoadError> {
    let buf = std::fs::read(path)?;
    let mut off = 0usize;
    let name = path.display().to_string();
    let w1 = dequant_int4(&buf, &mut off, inter, dim, &name)?; // gate
    let w3 = dequant_int4(&buf, &mut off, inter, dim, &name)?; // up
    let w2 = dequant_int4(&buf, &mut off, inter, dim, &name)?; // down
    Ok(Expert { w1, w3, w2, inter })
}

/// How expert weights are held in memory.
///
/// `Eager` dequantizes every expert to f32 at load — fast forwards, but
/// ~285 GB per rank at real-model scale. `Mmap` keeps the int4_bin files
/// memory-mapped and dequantizes rows on the fly — the only viable mode for
/// the 43-layer 257-expert model. Numerics are identical either way
/// (validated bitwise by `dsv4_expert_mmap.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertsMode {
    Eager,
    Mmap,
}

/// Load a full (single-stage) model from an export dir. `max_seq` sizes the
/// KV/compressed caches (runtime choice; the checkpoint supports up to 1M).
pub fn load_model(dir: &Path, max_seq: usize) -> Result<DsV4Model, LoadError> {
    let m = read_manifest(dir)?;
    let n = m.num_layers;
    load_stage_inner(dir, max_seq, m, 0, n, true, true, ExpertsMode::Eager)
}

/// Load one pipeline stage: layers `[lo, hi)` of the export, with embed on
/// the first stage and head on the last. Hash-gate layers must fall inside
/// the first stage (ids don't cross the wire) — enforced here.
pub fn load_stage(
    dir: &Path,
    max_seq: usize,
    lo: usize,
    hi: usize,
    first: bool,
    last: bool,
) -> Result<DsV4Model, LoadError> {
    load_stage_mode(dir, max_seq, lo, hi, first, last, ExpertsMode::Eager)
}

/// [`load_stage`] with an explicit [`ExpertsMode`].
pub fn load_stage_mode(
    dir: &Path,
    max_seq: usize,
    lo: usize,
    hi: usize,
    first: bool,
    last: bool,
    experts: ExpertsMode,
) -> Result<DsV4Model, LoadError> {
    let m = read_manifest(dir)?;
    if !first && lo < m.n_hash_layers {
        return Err(LoadError::Manifest(format!(
            "stage [{lo},{hi}) contains hash layers (< {}) but is not the first stage",
            m.n_hash_layers
        )));
    }
    load_stage_inner(dir, max_seq, m, lo, hi, first, last, experts)
}

fn read_manifest(dir: &Path) -> Result<Manifest, LoadError> {
    let m: Manifest = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)
        .map_err(|e| LoadError::Manifest(e.to_string()))?;
    if m.arch != "deepseek_v4" {
        return Err(LoadError::Manifest(format!(
            "arch {} != deepseek_v4",
            m.arch
        )));
    }
    if m.experts_format != "int4_bin" {
        return Err(LoadError::Manifest(format!(
            "experts_format {} unsupported by this loader",
            m.experts_format
        )));
    }
    // `compress_ratios` is indexed by absolute layer id in load_stage_inner; a
    // manifest whose array is shorter than its exported layers would panic
    // (index OOB) mid-load instead of failing cleanly here. Require it to cover
    // every exported layer.
    if let Some(&max_li) = m.exported_layers.iter().max() {
        if m.compress_ratios.len() <= max_li {
            return Err(LoadError::Manifest(format!(
                "compress_ratios has {} entries but exported layer id {max_li} needs at \
                 least {} — truncated or mismatched manifest?",
                m.compress_ratios.len(),
                max_li + 1
            )));
        }
    }
    Ok(m)
}

#[allow(clippy::too_many_arguments)]
fn load_stage_inner(
    dir: &Path,
    max_seq: usize,
    m: Manifest,
    lo: usize,
    hi: usize,
    first: bool,
    last: bool,
    experts_mode: ExpertsMode,
) -> Result<DsV4Model, LoadError> {
    let dim = m.hidden_size;

    let embed = if first {
        let embed_st = StFile::open(&dir.join("embed.safetensors"))?;
        Some(embed_st.f32("embed.weight")?.1)
    } else {
        None
    };
    let head = if last {
        let head_st = StFile::open(&dir.join("head.safetensors"))?;
        Some(Head {
            norm_w: head_st.f32("norm.weight")?.1,
            weight: head_st.f32("head.weight")?.1,
            hc_fn: head_st.f32("hc_head_fn")?.1,
            hc_base: head_st.f32("hc_head_base")?.1,
            hc_scale: head_st.f32("hc_head_scale")?.1,
        })
    } else {
        None
    };

    let stage_layers: Vec<usize> = m
        .exported_layers
        .iter()
        .copied()
        .filter(|&li| li >= lo && li < hi)
        .collect();
    // The stage MUST cover its whole [lo, hi) range. A manifest whose
    // exported_layers misses part of the range (e.g. a stale per-layer
    // manifest left behind by an incremental export) would otherwise load
    // silently with missing layers and generate deterministic garbage.
    let want: Vec<usize> = (lo..hi).collect();
    if stage_layers != want {
        return Err(LoadError::Manifest(format!(
            "stage [{lo},{hi}) needs layers {want:?} but manifest exported_layers \
             covers only {stage_layers:?} (full list: {:?}) — stale or partial manifest?",
            m.exported_layers
        )));
    }
    let mut layers = Vec::with_capacity(stage_layers.len());
    for &li in &stage_layers {
        let st = StFile::open(&dir.join(format!("shells/layer_{li:02}.safetensors")))?;
        let g = |n: &str| st.f32(n).map(|t| t.1);
        // Projection weights are stored bf16 in memory (half the streamed
        // bytes at batch=1); downcast the f32 tensor to bf16 bits on load.
        let gb = |n: &str| -> Result<Vec<u16>, LoadError> {
            Ok(g(n)?.iter().map(|v| bf16::from_f32(*v).to_bits()).collect())
        };
        let ratio = m.compress_ratios[li];
        let is_hash = li < m.n_hash_layers;

        let comp = if ratio > 0 {
            Some(Compressor::new(
                dim,
                m.head_dim,
                m.rope_head_dim,
                ratio,
                false,
                m.rms_norm_eps,
                CompressorWeights {
                    wkv: g("attn.compressor.wkv.weight")?,
                    wgate: g("attn.compressor.wgate.weight")?,
                    ape: g("attn.compressor.ape")?,
                    norm_w: g("attn.compressor.norm.weight")?,
                },
            ))
        } else {
            None
        };
        let idx = if ratio == 4 {
            let ic = Compressor::new(
                dim,
                m.index_head_dim,
                m.rope_head_dim,
                ratio,
                true,
                m.rms_norm_eps,
                CompressorWeights {
                    wkv: g("attn.indexer.compressor.wkv.weight")?,
                    wgate: g("attn.indexer.compressor.wgate.weight")?,
                    ape: g("attn.indexer.compressor.ape")?,
                    norm_w: g("attn.indexer.compressor.norm.weight")?,
                },
            );
            Some(Indexer::new(
                dim,
                m.q_lora_rank,
                m.index_n_heads,
                m.index_head_dim,
                m.rope_head_dim,
                m.index_topk,
                ratio,
                max_seq,
                IndexerWeights {
                    wq_b: g("attn.indexer.wq_b.weight")?,
                    weights_proj: g("attn.indexer.weights_proj.weight")?,
                },
                ic,
            ))
        } else {
            None
        };
        let theta = if ratio > 0 {
            m.compress_rope_theta
        } else {
            m.rope_theta
        };
        let orig = if ratio > 0 { m.original_seq_len } else { 0 };
        let freqs = precompute_freqs(
            m.rope_head_dim,
            max_seq,
            orig,
            theta,
            m.rope_factor,
            m.beta_fast,
            m.beta_slow,
        );
        let attn = AttentionLayer::new(
            dim,
            m.n_heads,
            m.head_dim,
            m.rope_head_dim,
            m.q_lora_rank,
            m.o_lora_rank,
            m.o_groups,
            m.window_size,
            ratio,
            max_seq,
            m.rms_norm_eps,
            AttnWeights {
                wq_a: gb("attn.wq_a.weight")?,
                q_norm_w: g("attn.q_norm.weight")?,
                wq_b: gb("attn.wq_b.weight")?,
                wkv: gb("attn.wkv.weight")?,
                kv_norm_w: g("attn.kv_norm.weight")?,
                wo_a: gb("attn.wo_a.weight")?,
                wo_b: gb("attn.wo_b.weight")?,
                sink: g("attn.attn_sink")?,
            },
            comp,
            idx,
            freqs,
        );

        let edir = dir.join(format!("experts/layer_{li:02}"));
        let load_one = |path: &Path| -> Result<AnyExpert, LoadError> {
            Ok(match experts_mode {
                ExpertsMode::Eager => {
                    AnyExpert::Eager(load_expert_bin(path, dim, m.expert_intermediate)?)
                }
                ExpertsMode::Mmap => {
                    AnyExpert::Mmap(MmapExpert::open(path, dim, m.expert_intermediate)?)
                }
            })
        };
        let experts: Vec<AnyExpert> = (0..m.n_routed_experts)
            .map(|e| load_one(&edir.join(format!("expert_{e:03}.bin"))))
            .collect::<Result<_, _>>()?;
        let shared = load_one(&edir.join("expert_shared.bin"))?;

        layers.push(Layer {
            attn,
            attn_norm_w: g("attn_norm.weight")?,
            ffn_norm_w: g("ffn_norm.weight")?,
            gate: GateWeights {
                weight: g("ffn.gate.weight")?,
                bias: if is_hash {
                    None
                } else {
                    Some(g("ffn.gate.bias")?)
                },
                tid2eid: if is_hash {
                    Some(st.i32("ffn.gate.tid2eid")?.1)
                } else {
                    None
                },
            },
            experts,
            shared,
            hc_attn: HcParams {
                fn_w: g("hc_attn_fn")?,
                base: g("hc_attn_base")?,
                scale: g("hc_attn_scale")?,
            },
            hc_ffn: HcParams {
                fn_w: g("hc_ffn_fn")?,
                base: g("hc_ffn_base")?,
                scale: g("hc_ffn_scale")?,
            },
        });
    }

    Ok(DsV4Model {
        cfg: ModelCfg {
            dim,
            vocab: m.vocab_size,
            hc: m.hc_mult,
            sinkhorn_iters: m.hc_sinkhorn_iters,
            hc_eps: m.hc_eps,
            norm_eps: m.rms_norm_eps,
            topk: m.top_k,
            route_scale: m.route_scale,
            swiglu_limit: m.swiglu_limit,
            n_hash_layers: m.n_hash_layers,
        },
        embed,
        layers,
        head,
        last_hiddens: Vec::new(),
        last_attn: Vec::new(),
        last_moe_in: Vec::new(),
        first_layer: lo,
        ov_experts: super::ov_expert::OvExperts::from_env(dir, dim),
    })
}
