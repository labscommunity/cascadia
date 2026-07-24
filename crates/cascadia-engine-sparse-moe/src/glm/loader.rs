//! GLM-5.2 loader — builds a full (single-stage) [`GlmModel`] from the layout
//! emitted by `tools/export_glm5.py`:
//!   <dir>/manifest.json                              (arch == "glm5")
//!   <dir>/embed.safetensors                          embed.weight (bf16)
//!   <dir>/head.safetensors                           head.weight (lm_head), norm.weight (final)
//!   <dir>/shells/layer_NN.safetensors                attn + norms (+ router on MoE layers), bf16
//!   <dir>/experts/layer_NN/expert_{EEE,shared}.bin   routed + shared experts (int4_bin)
//!   <dir>/experts/layer_NN/dense.bin                 dense-layer MLP (first_k_dense_replace)
//!
//! Shell weights are bf16 in the safetensors (downcast to bf16 bits on load,
//! like dsv4). Experts/dense are int4 → dequantized to f32 ([`AnyExpert::EagerF32`]):
//! int4 values are not exactly bf16-representable, so they stay f32 and run
//! through the f32-weight SwiGLU. Staged/pipeline loading is layered on later.

use std::path::Path;

use half::bf16;
use serde::Deserialize;

use super::attn::{AttentionLayer, AttnWeights};
use super::indexer::{Indexer, IndexerWeights};
use super::model::{GlmLayer, GlmModel, LayerMlp};
use super::moe::{AnyExpert, ExpertW, MoeLayer, MoeWeights};
use super::mtp::MtpHead;
use crate::dsv4::expert_mmap::MmapExpert;
use crate::dsv4::loader::{ExpertsMode, LoadError};
use crate::dsv4::rope::precompute_freqs;
use crate::dsv4::st::StFile;

/// int4 quant group (columns per bf16 scale) — the `cascadia-int4-gemm` /
/// `export_glm5.py` contract.
const G: usize = 32;

/// The subset of `manifest.json` the glm5 shell needs.
#[derive(Debug, Deserialize)]
pub struct GlmManifest {
    pub arch: String,
    pub num_layers: usize,
    pub dense_layers: Vec<usize>,
    pub num_experts: usize,
    pub top_k: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub vocab_size: usize,
    pub expert_intermediate: usize,
    pub dense_intermediate: usize,
    #[serde(default = "one")]
    pub n_shared_experts: usize,
    #[serde(default = "one_f32")]
    pub routed_scaling_factor: f32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub eos_token_ids: Vec<u32>,
    /// DSA lightning indexer. `index_n_heads == 0` (the default) means no indexer
    /// was exported → dense causal attention (correct for ctx ≤ index_topk).
    #[serde(default)]
    pub index_topk: usize,
    #[serde(default)]
    pub index_n_heads: usize,
    #[serde(default)]
    pub index_head_dim: usize,
    /// Per-layer IndexShare topology: `"full"` (owns an indexer) | `"shared"`
    /// (reuses the previous full layer's top-k). Empty = every layer is its own
    /// indexer (tiny/dev default).
    #[serde(default)]
    pub indexer_types: Vec<String>,
    /// Whether an MTP draft head was exported to `<dir>/mtp.safetensors` (enables
    /// speculative decode). Absent/false → single-token decode only.
    #[serde(default)]
    pub has_mtp: bool,
}

fn one() -> usize {
    1
}
fn one_f32() -> f32 {
    1.0
}

pub fn read_manifest(dir: &Path) -> Result<GlmManifest, LoadError> {
    let m: GlmManifest = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)
        .map_err(|e| LoadError::Manifest(e.to_string()))?;
    if m.arch != "glm5" {
        return Err(LoadError::Manifest(format!(
            "arch is {:?}, expected \"glm5\"",
            m.arch
        )));
    }
    Ok(m)
}

/// Dequantize one int4 section: `[out_dim, in_dim]` packed nibbles (2/byte,
/// low = even col, value = nibble-8) followed by bf16-LE per-`G` scales.
/// Advances `off`. Mirrors `export_glm5.py::_pack_int4_grouped`.
fn dequant_int4(
    buf: &[u8],
    off: &mut usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, LoadError> {
    if in_dim % G != 0 {
        return Err(LoadError::Manifest(format!(
            "int4 in_dim {in_dim} not divisible by group {G}"
        )));
    }
    let ng = in_dim / G;
    let packed_len = out_dim * in_dim / 2;
    let scale_len = out_dim * ng * 2;
    if *off + packed_len + scale_len > buf.len() {
        return Err(LoadError::ExpertBin(
            "int4 section truncated".into(),
            buf.len(),
        ));
    }
    let packed = &buf[*off..*off + packed_len];
    let scales_raw = &buf[*off + packed_len..*off + packed_len + scale_len];
    *off += packed_len + scale_len;
    let mut w = vec![0.0f32; out_dim * in_dim];
    for o in 0..out_dim {
        for gi in 0..ng {
            let s = bf16::from_le_bytes([
                scales_raw[(o * ng + gi) * 2],
                scales_raw[(o * ng + gi) * 2 + 1],
            ])
            .to_f32();
            for i in 0..G / 2 {
                let byte = packed[o * in_dim / 2 + gi * G / 2 + i];
                let lo = (byte & 0x0F) as i32 - 8;
                let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                w[o * in_dim + gi * G + 2 * i] = lo as f32 * s;
                w[o * in_dim + gi * G + 2 * i + 1] = hi as f32 * s;
            }
        }
    }
    Ok(w)
}

/// int3 packed group size in bytes (32 values × 3 bits = 96 bits = 12 bytes).
const INT3_GROUP_BYTES: usize = G * 3 / 8;

/// Dequantize one int3 section: `[out_dim, in_dim]` group-32, 3 bits/weight as an
/// LSB-first bitstream (32 vals → 12 bytes) then bf16-LE per-group scales.
/// Value = `(u3 − 4) · scale`. Mirrors `export_glm5.py::_pack_int3_grouped`.
///
/// The int3 format foundation (A2, cold-expert mixed precision). Not yet on the
/// load path: the format dispatch (`experts_format == "int3_bin"`) + the mmap
/// int3 GEMV are the remaining wiring; this + its test lock the pack↔unpack
/// contract first.
#[allow(dead_code)]
fn dequant_int3(
    buf: &[u8],
    off: &mut usize,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, LoadError> {
    if in_dim % G != 0 {
        return Err(LoadError::Manifest(format!(
            "int3 in_dim {in_dim} not divisible by group {G}"
        )));
    }
    let ng = in_dim / G;
    let packed_len = out_dim * ng * INT3_GROUP_BYTES;
    let scale_len = out_dim * ng * 2;
    if *off + packed_len + scale_len > buf.len() {
        return Err(LoadError::ExpertBin(
            "int3 section truncated".into(),
            buf.len(),
        ));
    }
    let packed = &buf[*off..*off + packed_len];
    let scales_raw = &buf[*off + packed_len..*off + packed_len + scale_len];
    *off += packed_len + scale_len;
    let mut w = vec![0.0f32; out_dim * in_dim];
    for o in 0..out_dim {
        for gi in 0..ng {
            let s = bf16::from_le_bytes([
                scales_raw[(o * ng + gi) * 2],
                scales_raw[(o * ng + gi) * 2 + 1],
            ])
            .to_f32();
            let gp =
                &packed[(o * ng + gi) * INT3_GROUP_BYTES..(o * ng + gi + 1) * INT3_GROUP_BYTES];
            for i in 0..G {
                let bit = i * 3;
                let b0 = gp[bit / 8] as u16;
                let b1 = if bit / 8 + 1 < INT3_GROUP_BYTES {
                    gp[bit / 8 + 1] as u16
                } else {
                    0
                };
                let u3 = (((b0 | (b1 << 8)) >> (bit % 8)) & 0x7) as i32;
                w[o * in_dim + gi * G + i] = (u3 - 4) as f32 * s;
            }
        }
    }
    Ok(w)
}

#[cfg(test)]
mod int3_tests {
    use super::*;

    // Pack 32 u3 values (0..7) LSB-first into 12 bytes, the exact layout
    // np.packbits(bitorder="little") produces in _pack_int3_grouped.
    fn pack_group(u3: &[u8; 32]) -> [u8; INT3_GROUP_BYTES] {
        let mut out = [0u8; INT3_GROUP_BYTES];
        for (i, &v) in u3.iter().enumerate() {
            for b in 0..3 {
                if (v >> b) & 1 == 1 {
                    let bit = i * 3 + b;
                    out[bit / 8] |= 1 << (bit % 8);
                }
            }
        }
        out
    }

    #[test]
    fn dequant_int3_matches_pack_layout() {
        // one row, one group; scale 1.0 (bf16 0x3f80). u3[i] = i % 8.
        let u3: [u8; 32] = std::array::from_fn(|i| (i % 8) as u8);
        let packed = pack_group(&u3);
        let mut buf = Vec::new();
        buf.extend_from_slice(&packed);
        buf.extend_from_slice(&bf16::from_f32(1.0).to_le_bytes());
        let mut off = 0;
        let w = dequant_int3(&buf, &mut off, 1, 32).unwrap();
        assert_eq!(off, buf.len(), "consumed the whole section");
        for i in 0..32 {
            assert_eq!(w[i], (u3[i] as i32 - 4) as f32, "value {i} mismatch");
        }
        // section byte size: 12 packed + 2 scale.
        assert_eq!(buf.len(), INT3_GROUP_BYTES + 2);
    }
}

/// Load one `expert_*.bin` / `dense.bin`: gate `[inter,hidden]`, up
/// `[inter,hidden]`, down `[hidden,inter]`. `Eager` dequantizes to f32 (fast,
/// tiny/dev); `Mmap` keeps it packed on disk and dequantizes rows on the fly
/// (the only mode that fits the real model).
fn load_expert_bin(
    path: &Path,
    hidden: usize,
    inter: usize,
    mode: ExpertsMode,
) -> Result<AnyExpert, LoadError> {
    match mode {
        ExpertsMode::Eager => {
            let buf = std::fs::read(path)?;
            let mut off = 0usize;
            let wg = dequant_int4(&buf, &mut off, inter, hidden)?;
            let wu = dequant_int4(&buf, &mut off, inter, hidden)?;
            let wd = dequant_int4(&buf, &mut off, hidden, inter)?;
            Ok(AnyExpert::EagerF32 { wg, wu, wd })
        }
        ExpertsMode::Mmap => Ok(AnyExpert::Mmap(MmapExpert::open(path, hidden, inter)?)),
    }
}

/// Build one transformer layer `li` from its shell safetensors + expert bins.
fn load_layer(
    dir: &Path,
    m: &GlmManifest,
    li: usize,
    max_seq: usize,
    mode: ExpertsMode,
    bf16_kv: bool,
) -> Result<GlmLayer, LoadError> {
    let (hidden, eps) = (m.hidden_size, m.rms_norm_eps);
    let (h, nope, rope) = (
        m.num_attention_heads,
        m.qk_nope_head_dim,
        m.qk_rope_head_dim,
    );
    let (vh, kvl, ql) = (m.v_head_dim, m.kv_lora_rank, m.q_lora_rank);
    let moe_inter = m.expert_intermediate;

    let st = StFile::open(&dir.join(format!("shells/layer_{li:02}.safetensors")))?;
    let g = |n: &str| st.f32(n).map(|t| t.1);
    // Projection weights held as bf16 bits (half the batch-1 traffic).
    let gb = |n: &str| -> Result<Vec<u16>, LoadError> {
        Ok(g(n)?.iter().map(|v| bf16::from_f32(*v).to_bits()).collect())
    };
    let aw = AttnWeights {
        wq_a: gb("self_attn.wq_a.weight")?,
        q_a_ln: g("self_attn.q_a_layernorm.weight")?,
        wq_b: gb("self_attn.wq_b.weight")?,
        wkv_a: gb("self_attn.wkv_a.weight")?,
        kv_a_ln: g("self_attn.kv_a_layernorm.weight")?,
        wkv_b: gb("self_attn.wkv_b.weight")?,
        wo: gb("self_attn.o_proj.weight")?,
    };
    let freqs = precompute_freqs(rope, max_seq, 0, m.rope_theta, 1.0, 32.0, 1.0);
    let mut attn = AttentionLayer::new(hidden, h, nope, rope, vh, kvl, ql, max_seq, aw, freqs);
    attn.set_kv_precision(bf16_kv);

    // DSA / IndexShare (when exported): a "full" layer owns an indexer and
    // publishes its top-`index_topk` selection; a "shared" layer reuses the
    // previous full layer's selection. `indexer_types` empty (tiny/dev) => every
    // layer is "full". The indexer ropes the qk_rope prefix at the attention's
    // positions, so it shares the attention rope config.
    if m.index_n_heads > 0 {
        let topk = if m.index_topk > 0 {
            m.index_topk
        } else {
            usize::MAX
        };
        let is_full = m.indexer_types.get(li).map(|t| t == "full").unwrap_or(true); // absent/short -> full
        if is_full {
            let iw = IndexerWeights {
                ix_wq: gb("self_attn.indexer.wq_b.weight")?,
                ix_wk: gb("self_attn.indexer.wk.weight")?,
                ix_wp: gb("self_attn.indexer.weights_proj.weight")?,
                k_norm_w: g("self_attn.indexer.k_norm.weight")?,
                k_norm_b: g("self_attn.indexer.k_norm.bias")?,
            };
            let ifreqs = precompute_freqs(rope, max_seq, 0, m.rope_theta, 1.0, 32.0, 1.0);
            let indexer = Indexer::new(
                hidden,
                ql,
                m.index_n_heads,
                m.index_head_dim,
                rope,
                max_seq,
                crate::glm::attn::MLA_LATENT_EPS,
                iw,
                ifreqs,
            );
            attn.attach_indexer(indexer, topk);
        } else {
            attn.mark_shared(topk);
        }
    }

    let edir = dir.join("experts").join(format!("layer_{li:02}"));
    let mlp = if m.dense_layers.contains(&li) {
        let w = load_expert_bin(&edir.join("dense.bin"), hidden, m.dense_intermediate, mode)?;
        LayerMlp::Dense {
            w,
            inter: m.dense_intermediate,
        }
    } else {
        let mut experts = Vec::with_capacity(m.num_experts);
        for e in 0..m.num_experts {
            experts.push(load_expert_bin(
                &edir.join(format!("expert_{e:03}.bin")),
                hidden,
                moe_inter,
                mode,
            )?);
        }
        let shared = load_expert_bin(&edir.join("expert_shared.bin"), hidden, moe_inter, mode)?;
        let mw = MoeWeights {
            router_w: g("mlp.gate.weight")?,
            router_bias: g("mlp.gate.e_score_correction_bias")?,
            experts,
            shared,
        };
        LayerMlp::Moe(MoeLayer::new(
            hidden,
            m.num_experts,
            m.top_k,
            moe_inter,
            moe_inter * m.n_shared_experts,
            m.routed_scaling_factor,
            mw,
        ))
    };
    Ok(GlmLayer::new(
        hidden,
        eps,
        g("input_layernorm.weight")?,
        g("post_attention_layernorm.weight")?,
        attn,
        mlp,
    ))
}

/// Build the MTP draft head from `<dir>/mtp.safetensors` (present iff
/// `manifest.has_mtp`). The block is a normal attn + MoE layer with NO indexer
/// (the draft window is far below `index_topk`, so attention is full causal);
/// its experts are stored bf16 in the safetensors ([`AnyExpert::Bf16`]).
/// `max_seq` sizes the block's KV cache (it resets and consumes ≤ g positions
/// per draft round).
fn load_mtp(
    dir: &Path,
    m: &GlmManifest,
    max_seq: usize,
    bf16_kv: bool,
) -> Result<MtpHead, LoadError> {
    let (hidden, eps) = (m.hidden_size, m.rms_norm_eps);
    let (h, nope, rope) = (
        m.num_attention_heads,
        m.qk_nope_head_dim,
        m.qk_rope_head_dim,
    );
    let (vh, kvl, ql) = (m.v_head_dim, m.kv_lora_rank, m.q_lora_rank);
    let moe_inter = m.expert_intermediate;

    let st = StFile::open(&dir.join("mtp.safetensors"))?;
    let g = |n: &str| st.f32(n).map(|t| t.1);
    let gb = |n: &str| -> Result<Vec<u16>, LoadError> {
        Ok(g(n)?.iter().map(|v| bf16::from_f32(*v).to_bits()).collect())
    };
    // bf16 expert (gate/up/down) held as bf16 bits — the shell's native dtype.
    let bexp = |p: &str| -> Result<AnyExpert, LoadError> {
        Ok(AnyExpert::Bf16(ExpertW {
            wg: gb(&format!("{p}.gate.weight"))?,
            wu: gb(&format!("{p}.up.weight"))?,
            wd: gb(&format!("{p}.down.weight"))?,
        }))
    };

    let aw = AttnWeights {
        wq_a: gb("mtp.block.self_attn.wq_a.weight")?,
        q_a_ln: g("mtp.block.self_attn.q_a_layernorm.weight")?,
        wq_b: gb("mtp.block.self_attn.wq_b.weight")?,
        wkv_a: gb("mtp.block.self_attn.wkv_a.weight")?,
        kv_a_ln: g("mtp.block.self_attn.kv_a_layernorm.weight")?,
        wkv_b: gb("mtp.block.self_attn.wkv_b.weight")?,
        wo: gb("mtp.block.self_attn.o_proj.weight")?,
    };
    let freqs = precompute_freqs(rope, max_seq, 0, m.rope_theta, 1.0, 32.0, 1.0);
    let mut attn = AttentionLayer::new(hidden, h, nope, rope, vh, kvl, ql, max_seq, aw, freqs);
    attn.set_kv_precision(bf16_kv);

    let mut experts = Vec::with_capacity(m.num_experts);
    for e in 0..m.num_experts {
        experts.push(bexp(&format!("mtp.block.mlp.experts.{e}"))?);
    }
    let mw = MoeWeights {
        router_w: g("mtp.block.mlp.gate.weight")?,
        router_bias: g("mtp.block.mlp.gate.e_score_correction_bias")?,
        experts,
        shared: bexp("mtp.block.mlp.shared")?,
    };
    let moe = MoeLayer::new(
        hidden,
        m.num_experts,
        m.top_k,
        moe_inter,
        moe_inter * m.n_shared_experts,
        m.routed_scaling_factor,
        mw,
    );
    let block = GlmLayer::new(
        hidden,
        eps,
        g("mtp.block.input_layernorm.weight")?,
        g("mtp.block.post_attention_layernorm.weight")?,
        attn,
        LayerMlp::Moe(moe),
    );

    Ok(MtpHead::new(
        hidden,
        m.vocab_size,
        eps,
        g("mtp.enorm.weight")?,
        g("mtp.hnorm.weight")?,
        g("mtp.norm.weight")?,
        gb("mtp.eh_proj.weight")?,
        block,
    ))
}

/// One pipeline stage: `embed` present iff this rank owns layer 0 (`first`),
/// `head` (final_norm, lm_head) present iff it owns the last layer (`last`),
/// and the layer slice `[lo, hi)`.
pub struct GlmStage {
    pub embed: Option<Vec<f32>>,
    pub layers: Vec<GlmLayer>,
    pub head: Option<(Vec<f32>, Vec<f32>)>,
    pub hidden: usize,
    pub vocab: usize,
    pub eps: f32,
    pub eos: Vec<u32>,
}

/// Load the layer slice `[lo, hi)` of a model. Reads embed only when `first`,
/// head only when `last`. `max_seq` sizes the attention KV caches.
pub fn load_stage(
    dir: &Path,
    max_seq: usize,
    lo: usize,
    hi: usize,
    first: bool,
    last: bool,
    mode: ExpertsMode,
    bf16_kv: bool,
) -> Result<GlmStage, LoadError> {
    let m = read_manifest(dir)?;
    let embed = if first {
        Some(
            StFile::open(&dir.join("embed.safetensors"))?
                .f32("embed.weight")?
                .1,
        )
    } else {
        None
    };
    let head = if last {
        let h = StFile::open(&dir.join("head.safetensors"))?;
        Some((h.f32("norm.weight")?.1, h.f32("head.weight")?.1))
    } else {
        None
    };
    let mut layers = Vec::with_capacity(hi - lo);
    for li in lo..hi {
        layers.push(load_layer(dir, &m, li, max_seq, mode, bf16_kv)?);
    }
    Ok(GlmStage {
        embed,
        layers,
        head,
        hidden: m.hidden_size,
        vocab: m.vocab_size,
        eps: m.rms_norm_eps,
        eos: m.eos_token_ids,
    })
}

/// Load a full single-stage model. `max_seq` sizes the attention KV caches.
pub fn load_model(dir: &Path, max_seq: usize) -> Result<GlmModel, LoadError> {
    load_model_with(dir, max_seq, ExpertsMode::Eager)
}

/// Load a full single-stage model with an explicit experts `mode`. `Eager`
/// dequantizes all int4 experts to f32 in RAM (tiny/dev); `Mmap` streams them
/// from disk (the only mode that fits the real 410 GB model). The MTP draft
/// head, when present, is always bf16 in RAM (mode-independent).
pub fn load_model_with(
    dir: &Path,
    max_seq: usize,
    mode: ExpertsMode,
) -> Result<GlmModel, LoadError> {
    let m = read_manifest(dir)?;
    // The single-process reference is always exact f32 KV (bf16_kv = false); it is
    // the oracle the staged runner is validated against.
    let s = load_stage(dir, max_seq, 0, m.num_layers, true, true, mode, false)?;
    let (final_norm, lm_head) = s.head.expect("full model has a head");
    let mut model = GlmModel::new(
        s.hidden,
        s.vocab,
        s.eps,
        s.embed.expect("full model has an embed"),
        s.layers,
        final_norm,
        lm_head,
    );
    // Attach the MTP draft head (speculative decode) when the export carries one.
    if m.has_mtp {
        model.set_mtp(load_mtp(dir, &m, max_seq, false)?);
    }
    Ok(model)
}
