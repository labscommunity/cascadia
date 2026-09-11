//! Inkling layer + model: parity tests on a random tiny model (prefill vs
//! per-token, truncate, prefix snapshot/restore, greedy determinism) and the
//! HF goldens (per-layer hidden states, final logits, greedy ids, MoE block)
//! built straight from the checkpoint-named tensors in
//! `tests/fixtures/inkling/fixtures.safetensors`.
//!
//! Regenerate fixtures:
//!   python tools/inkling_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/inkling

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::inkling::attn::{AttentionLayer, AttnDims, AttnWeights};
use cascadia_engine_sparse_moe::inkling::conv::ShortConv;
use cascadia_engine_sparse_moe::inkling::ffn::{deinterleave_w13, AnyExpert, ExpertW};
use cascadia_engine_sparse_moe::inkling::model::{Layer, LayerMlp, Model, WideTable};
use cascadia_engine_sparse_moe::inkling::moe::{DenseMlp, MoeLayer, MoeWeights};
use cascadia_engine_sparse_moe::inkling::relpos::RelPos;

fn fixtures() -> Option<StFile> {
    let p = match std::env::var_os("INKLING_FIXTURES") {
        Some(dir) => PathBuf::from(dir).join("fixtures.safetensors"),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/inkling/fixtures.safetensors"),
    };
    if !p.exists() {
        eprintln!(
            "inkling fixtures missing at {}; skipping golden",
            p.display()
        );
        return None;
    }
    Some(StFile::open(&p).expect("open inkling fixtures"))
}

fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut worst = (0usize, 0.0f32, 0.0f32, 0.0f32);
    let mut n_bad = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_mag = 0.0f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.is_finite() && w.is_finite(),
            "{name}: non-finite value at [{i}]: got {g} want {w}"
        );
        let d = (g - w).abs();
        max_abs = max_abs.max(d);
        max_mag = max_mag.max(w.abs());
        if d > atol + rtol * w.abs() {
            n_bad += 1;
            if d > worst.1 {
                worst = (i, d, g, w);
            }
        }
    }
    assert!(
        n_bad == 0,
        "{name}: {n_bad}/{} elements outside atol {atol} + rtol {rtol} (max |diff| {max_abs}, \
         max |want| {max_mag}); worst at [{}]: got {} want {} (diff {})",
        got.len(),
        worst.0,
        worst.2,
        worst.3,
        worst.1
    );
}

/// Scale-aware closeness for hidden states / logits, row by row (`cols` wide):
/// `|got - want| <= frac · max_j |want_row[j]| + rtol · |want|`. Every linear
/// rounds to bf16 at write-back, so the absolute error of a row tracks the
/// magnitude of the branch that produced it (route_scale-8 MoE outputs reach
/// several units), not the individual element: where the residual and the
/// branch cancel to ~0.5, that element still carries the branch's ~0.03 of
/// rounding. A per-element relative band cannot express that; 2% of the row
/// scale + 2% relative can. Prints the worst |diff| / row-scale so the margin
/// is visible under `--nocapture`.
fn assert_close_rows(name: &str, got: &[f32], want: &[f32], cols: usize, frac: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    assert_eq!(got.len() % cols, 0, "{name}: len % cols != 0");
    let mut worst = (0usize, 0.0f32, 0.0f32, 0.0f32, 0.0f32); // (i, diff/scale, got, want, scale)
    let mut n_bad = 0usize;
    for (r, (g_row, w_row)) in got
        .chunks_exact(cols)
        .zip(want.chunks_exact(cols))
        .enumerate()
    {
        let scale = w_row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        for (c, (&g, &w)) in g_row.iter().zip(w_row).enumerate() {
            assert!(
                g.is_finite() && w.is_finite(),
                "{name}: non-finite value at [{r},{c}]: got {g} want {w}"
            );
            let d = (g - w).abs();
            if d > frac * scale + rtol * w.abs() {
                n_bad += 1;
            }
            let scaled = d / scale.max(1e-12);
            if scaled > worst.1 {
                worst = (r * cols + c, scaled, g, w, scale);
            }
        }
    }
    eprintln!(
        "{name}: worst |diff|/row-scale {:.4} at [{}] (got {} want {} row scale {})",
        worst.1, worst.0, worst.2, worst.3, worst.4
    );
    assert!(
        n_bad == 0,
        "{name}: {n_bad}/{} elements outside frac {frac}·row-scale + rtol {rtol}·|want|; \
         worst |diff|/row-scale {} at [{}]: got {} want {} (row scale {})",
        got.len(),
        worst.1,
        worst.0,
        worst.2,
        worst.3,
        worst.4
    );
}

fn bits(f: &[f32]) -> Vec<u16> {
    f.iter()
        .map(|&v| half::bf16::from_f32(v).to_bits())
        .collect()
}

fn to_bf16(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| to_bf16(self.next() * scale)).collect()
    }
    fn norm(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 1.0 + 0.1 * self.next()).collect()
    }
}

// ---- random tiny model -------------------------------------------------------

struct Cfg {
    hidden: usize,
    vocab: usize,
    unpadded: usize,
    heads: usize,
    kv: usize,
    d: usize,
    d_rel: usize,
    window: usize,
    rel_extent: usize,
    dense_inter: usize,
    moe_inter: usize,
    n_routed: usize,
    top_k: usize,
    n_shared: usize,
    route_scale: f32,
    n_floor: f32,
    alpha: f32,
    mup: f32,
    eps: f32,
    max_seq: usize,
    rewind: usize,
}

fn cfg() -> Cfg {
    Cfg {
        hidden: 16,
        vocab: 24,
        unpadded: 20,
        heads: 2,
        kv: 1,
        d: 8,
        d_rel: 2,
        window: 3,
        rel_extent: 6,
        dense_inter: 32,
        moe_inter: 32,
        n_routed: 4,
        top_k: 2,
        n_shared: 2,
        route_scale: 8.0,
        n_floor: 2.0,
        alpha: 0.1,
        mup: 2.0,
        eps: 1e-6,
        max_seq: 48,
        rewind: 4,
    }
}

fn random_expert(rng: &mut Lcg, hidden: usize, inter: usize) -> AnyExpert {
    ExpertW {
        wg: bits(&rng.vec(inter * hidden, 0.3)),
        wu: bits(&rng.vec(inter * hidden, 0.3)),
        wd: bits(&rng.vec(hidden * inter, 0.3)),
    }
    .into()
}

fn random_attn(rng: &mut Lcg, c: &Cfg, sliding: bool) -> AttentionLayer {
    let (hd, hq, hkv, d, dr) = (c.hidden, c.heads, c.kv, c.d, c.d_rel);
    let dims = if sliding {
        AttnDims::sliding(hd, hq, hkv, d, dr, c.window, c.eps)
    } else {
        AttnDims::global(hd, hq, hkv, d, dr, c.max_seq, c.eps)
    }
    .with_log_scaling(c.n_floor, c.alpha)
    .with_rewind(c.rewind);
    let extent = if sliding { c.window } else { c.rel_extent };
    let w = AttnWeights {
        wq: bits(&rng.vec(hq * d * hd, 0.4)),
        wk: bits(&rng.vec(hkv * d * hd, 0.4)),
        wv: bits(&rng.vec(hkv * d * hd, 0.4)),
        wr: bits(&rng.vec(hq * dr * hd, 0.4)),
        wo: bits(&rng.vec(hd * hq * d, 0.4)),
        q_norm: rng.norm(d),
        k_norm: rng.norm(d),
    };
    let k_conv = ShortConv::with_rewind(rng.vec(hkv * d * 4, 0.3), hkv * d, 4, c.rewind);
    let v_conv = ShortConv::with_rewind(rng.vec(hkv * d * 4, 0.3), hkv * d, 4, c.rewind);
    let relpos = RelPos::new(rng.vec(dr * extent, 0.5), dr, extent);
    AttentionLayer::from_parts(dims, w, k_conv, v_conv, relpos)
}

/// Layer 0: dense + sliding; layer 1: MoE + global.
fn random_model(seed: u64) -> Model {
    let c = cfg();
    let mut rng = Lcg(seed);
    let mut layers = Vec::new();
    for li in 0..2 {
        let sliding = li == 0;
        let attn = random_attn(&mut rng, &c, sliding);
        let mlp = if li == 0 {
            LayerMlp::Dense(DenseMlp::new(
                random_expert(&mut rng, c.hidden, c.dense_inter),
                c.dense_inter,
                0.9,
            ))
        } else {
            let experts = (0..c.n_routed)
                .map(|_| random_expert(&mut rng, c.hidden, c.moe_inter))
                .collect();
            let shared = (0..c.n_shared)
                .map(|_| random_expert(&mut rng, c.hidden, c.moe_inter))
                .collect();
            let w = MoeWeights {
                router_w: rng.vec((c.n_routed + c.n_shared) * c.hidden, 0.5),
                router_bias: rng.vec(c.n_routed, 0.05),
                global_scale: 1.1,
                experts,
                shared,
            };
            LayerMlp::Moe(MoeLayer::new(
                c.hidden,
                c.moe_inter,
                c.top_k,
                c.route_scale,
                w,
            ))
        };
        layers.push(Layer::new(
            c.hidden,
            c.eps,
            rng.norm(c.hidden),
            attn,
            ShortConv::with_rewind(rng.vec(c.hidden * 4, 0.3), c.hidden, 4, c.rewind),
            rng.norm(c.hidden),
            mlp,
            ShortConv::with_rewind(rng.vec(c.hidden * 4, 0.3), c.hidden, 4, c.rewind),
        ));
    }
    Model::new(
        c.hidden,
        c.vocab,
        c.unpadded,
        c.eps,
        c.mup,
        WideTable::Bf16(bits(&rng.vec(c.vocab * c.hidden, 1.0))),
        rng.norm(c.hidden),
        layers,
        rng.norm(c.hidden),
        WideTable::Bf16(bits(&rng.vec(c.vocab * c.hidden, 1.0))),
    )
}

#[test]
fn prefill_is_bit_identical_to_per_token() {
    let prompt: Vec<u32> = vec![3, 7, 1, 19, 4, 4, 11, 0, 9];
    let mut m = random_model(1);
    m.reset();
    let mut per_token = Vec::new();
    for &t in &prompt {
        per_token = m.forward_token(t);
    }
    assert_eq!(m.len(), prompt.len());
    m.reset();
    let pre = m.prefill(&prompt);
    assert_eq!(pre, per_token, "prefill last logits vs per-token");
    assert_eq!(pre.len(), cfg().unpadded, "logits sliced to unpadded vocab");
}

#[test]
fn truncate_then_redecode_matches_uninterrupted() {
    let prompt: Vec<u32> = vec![5, 2, 8, 8, 1, 17, 3];
    let mut m = random_model(2);
    m.reset();
    let _ = m.prefill(&prompt);
    let a1 = m.forward_token(6);
    let a2 = m.forward_token(12);
    let a3 = m.forward_token(2);
    // Reject the last two decoded positions and redo them.
    m.truncate(prompt.len() + 1);
    let b2 = m.forward_token(12);
    let b3 = m.forward_token(2);
    assert_eq!(a2, b2, "first re-decoded logits");
    assert_eq!(a3, b3, "second re-decoded logits");
    let _ = a1;
}

#[test]
fn prefix_snapshot_restore_is_bit_exact_vs_full_prefill() {
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let k = 5usize;
    let mut m = random_model(3);
    m.reset();
    let full = m.prefill(&prompt);
    let full_step = m.forward_token(9);

    m.reset();
    let _ = m.prefill(&prompt[..k]);
    let snap = m.snapshot_prefix();
    assert_eq!(snap.len(), 2);
    assert_eq!(snap[0].len(), k);
    m.reset();
    m.restore_prefix(&snap);
    assert_eq!(m.len(), k);
    let reuse = m.prefill(&prompt[k..]);
    let reuse_step = m.forward_token(9);
    assert_eq!(reuse, full, "prefix-restored prefill logits");
    assert_eq!(reuse_step, full_step, "decode after prefix restore");
}

#[test]
fn greedy_is_deterministic_and_within_unpadded_vocab() {
    let prompt: Vec<u32> = vec![2, 9, 14];
    let mut m = random_model(4);
    let a = m.greedy(&prompt, 6);
    let b = m.greedy(&prompt, 6);
    assert_eq!(a, b);
    assert_eq!(a.len(), 6);
    assert!(a.iter().all(|&t| (t as usize) < cfg().unpadded));
    // greedy == argmax of the per-step logits from a manual loop
    m.reset();
    let mut logits = m.prefill(&prompt);
    for &want in &a {
        let got = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 {
                    (i, v)
                } else {
                    b
                }
            })
            .0 as u32;
        assert_eq!(got, want);
        logits = m.forward_token(want);
    }
}

#[test]
fn moe_batch_is_bit_identical_to_per_row() {
    let c = cfg();
    let mut rng = Lcg(9);
    let experts = (0..c.n_routed)
        .map(|_| random_expert(&mut rng, c.hidden, c.moe_inter))
        .collect();
    let shared = (0..c.n_shared)
        .map(|_| random_expert(&mut rng, c.hidden, c.moe_inter))
        .collect();
    let w = MoeWeights {
        router_w: rng.vec((c.n_routed + c.n_shared) * c.hidden, 0.5),
        router_bias: rng.vec(c.n_routed, 0.05),
        global_scale: 0.8,
        experts,
        shared,
    };
    let moe = MoeLayer::new(c.hidden, c.moe_inter, c.top_k, c.route_scale, w);
    let rows = 7;
    let xs = rng.vec(rows * c.hidden, 1.0);
    let batch = moe.forward_batch(&xs, rows);
    let mut per_row = Vec::new();
    for x in xs.chunks_exact(c.hidden) {
        per_row.extend(moe.forward(x));
    }
    assert_eq!(batch, per_row);
    // The output is a nontrivial mix (weights/gammas both nonzero).
    let g = moe.route(&xs[..c.hidden]);
    assert_eq!(g.idx.len(), c.top_k);
    assert_eq!(g.gammas.len(), c.n_shared);
    assert!(g.gammas.iter().all(|&v| v > 0.0));
}

// ---- HF goldens (tiny config, PORT_SPEC §4) ---------------------------------

const HIDDEN: usize = 64;
const N_LAYERS: usize = 4;
const HEADS: usize = 4;
const KV: usize = 2;
const HEAD_DIM: usize = 16;
const D_REL: usize = 4;
const WINDOW: usize = 4;
const SLIDING_LAYERS: [usize; 3] = [0, 1, 2];
const DENSE_LAYERS: [usize; 1] = [0];
const DENSE_INTER: usize = 64;
const MOE_INTER: usize = 32;
const N_ROUTED: usize = 8;
const TOP_K: usize = 2;
const N_SHARED: usize = 2;
const ROUTE_SCALE: f32 = 8.0;
const VOCAB: usize = 128;
const UNPADDED: usize = 120;
const N_FLOOR: f32 = 4.0;
const ALPHA: f32 = 0.1;
const MUP: f32 = 2.0;
const EPS: f32 = 1e-6;
const CONV_K: usize = 4;

fn f(fx: &StFile, name: &str) -> Vec<f32> {
    fx.f32(name).unwrap_or_else(|e| panic!("{name}: {e}")).1
}

/// First present tensor among `names` (the checkpoint name first, then the HF
/// module name the generator may have used instead).
fn f_any(fx: &StFile, names: &[&str]) -> Vec<f32> {
    for n in names {
        if let Ok(t) = fx.f32(n) {
            return t.1;
        }
    }
    panic!("none of {names:?} found in fixtures");
}

fn scalar(fx: &StFile, name: &str) -> f32 {
    let v = f(fx, name);
    assert_eq!(v.len(), 1, "{name}: expected a scalar");
    v[0]
}

/// gate/up from an interleaved `w13` slab + `w2` -> bf16 expert.
fn expert_from(w13: &[f32], w2: &[f32], inter: usize, hidden: usize) -> AnyExpert {
    let (g, u) = deinterleave_w13(w13, inter, hidden);
    assert_eq!(w2.len(), hidden * inter);
    ExpertW {
        wg: bits(&g),
        wu: bits(&u),
        wd: bits(w2),
    }
    .into()
}

fn attn_from_fixture(fx: &StFile, li: usize, max_seq: usize) -> AttentionLayer {
    let p = format!("model.llm.layers.{li}.attn");
    let sliding = SLIDING_LAYERS.contains(&li);
    let dims = if sliding {
        AttnDims::sliding(HIDDEN, HEADS, KV, HEAD_DIM, D_REL, WINDOW, EPS)
    } else {
        AttnDims::global(HIDDEN, HEADS, KV, HEAD_DIM, D_REL, max_seq, EPS)
    }
    .with_log_scaling(N_FLOOR, ALPHA);
    let w = AttnWeights {
        wq: bits(&f(fx, &format!("{p}.wq_du.weight"))),
        wk: bits(&f(fx, &format!("{p}.wk_dv.weight"))),
        wv: bits(&f(fx, &format!("{p}.wv_dv.weight"))),
        wr: bits(&f(fx, &format!("{p}.wr_du.weight"))),
        wo: bits(&f(fx, &format!("{p}.wo_ud.weight"))),
        q_norm: f(fx, &format!("{p}.q_norm.weight")),
        k_norm: f(fx, &format!("{p}.k_norm.weight")),
    };
    let c = KV * HEAD_DIM;
    let k_conv = ShortConv::new(f(fx, &format!("{p}.k_sconv.weight")), c, CONV_K);
    let v_conv = ShortConv::new(f(fx, &format!("{p}.v_sconv.weight")), c, CONV_K);
    let (pshape, proj) = fx
        .f32(&format!("{p}.rel_logits_proj.proj"))
        .expect("rel_logits_proj.proj");
    let relpos = RelPos::new(proj, pshape[0], pshape[1]);
    AttentionLayer::from_parts(dims, w, k_conv, v_conv, relpos)
}

fn moe_from_fixture(fx: &StFile, li: usize) -> MoeLayer {
    let p = format!("model.llm.layers.{li}.mlp");
    let w13 = f(fx, &format!("{p}.experts.w13_weight")); // [E, 2I, H] interleaved
    let w2 = f(fx, &format!("{p}.experts.w2_weight")); // [E, H, I]
    let (e13, e2) = (2 * MOE_INTER * HIDDEN, HIDDEN * MOE_INTER);
    assert_eq!(w13.len(), N_ROUTED * e13, "w13_weight shape");
    assert_eq!(w2.len(), N_ROUTED * e2, "w2_weight shape");
    let experts = (0..N_ROUTED)
        .map(|e| {
            expert_from(
                &w13[e * e13..(e + 1) * e13],
                &w2[e * e2..(e + 1) * e2],
                MOE_INTER,
                HIDDEN,
            )
        })
        .collect();
    let s13 = f(fx, &format!("{p}.shared_experts.shared_w13_weight")); // [2, 2I, H]
    let s2 = f(fx, &format!("{p}.shared_experts.shared_w2_weight")); // [2, H, I]
    assert_eq!(s13.len(), N_SHARED * e13, "shared_w13_weight shape");
    let shared = (0..N_SHARED)
        .map(|s| {
            expert_from(
                &s13[s * e13..(s + 1) * e13],
                &s2[s * e2..(s + 1) * e2],
                MOE_INTER,
                HIDDEN,
            )
        })
        .collect();
    let w = MoeWeights {
        router_w: f(fx, &format!("{p}.gate.weight")),
        router_bias: f_any(
            fx,
            &[
                &format!("{p}.gate.bias"),
                &format!("{p}.gate.e_score_correction_bias"),
            ],
        ),
        global_scale: scalar(fx, &format!("{p}.gate.global_scale")),
        experts,
        shared,
    };
    MoeLayer::new(HIDDEN, MOE_INTER, TOP_K, ROUTE_SCALE, w)
}

fn dense_from_fixture(fx: &StFile, li: usize) -> DenseMlp {
    let p = format!("model.llm.layers.{li}.mlp");
    let w13 = f(fx, &format!("{p}.w13_dn.weight")); // [2·Id, H] interleaved
    let w2 = f(fx, &format!("{p}.w2_md.weight")); // [H, Id]
    DenseMlp::new(
        expert_from(&w13, &w2, DENSE_INTER, HIDDEN),
        DENSE_INTER,
        scalar(fx, &format!("{p}.global_scale")),
    )
}

fn model_from_fixture(fx: &StFile, max_seq: usize) -> Model {
    let mut layers = Vec::with_capacity(N_LAYERS);
    for li in 0..N_LAYERS {
        let p = format!("model.llm.layers.{li}");
        let attn = attn_from_fixture(fx, li, max_seq);
        let mlp = if DENSE_LAYERS.contains(&li) {
            LayerMlp::Dense(dense_from_fixture(fx, li))
        } else {
            LayerMlp::Moe(moe_from_fixture(fx, li))
        };
        layers.push(Layer::new(
            HIDDEN,
            EPS,
            f(fx, &format!("{p}.attn_norm.weight")),
            attn,
            ShortConv::new(f(fx, &format!("{p}.attn_sconv.weight")), HIDDEN, CONV_K),
            f(fx, &format!("{p}.mlp_norm.weight")),
            mlp,
            ShortConv::new(f(fx, &format!("{p}.mlp_sconv.weight")), HIDDEN, CONV_K),
        ));
    }
    Model::new(
        HIDDEN,
        VOCAB,
        UNPADDED,
        EPS,
        MUP,
        WideTable::Bf16(bits(&f(fx, "model.llm.embed.weight"))),
        f(fx, "model.llm.embed_norm.weight"),
        layers,
        f(fx, "model.llm.norm.weight"),
        WideTable::Bf16(bits(&f(fx, "model.llm.unembed.weight"))),
    )
}

fn prompt_ids(fx: &StFile) -> Vec<u32> {
    fx.i32("prompt_ids")
        .expect("prompt_ids")
        .1
        .iter()
        .map(|&t| t as u32)
        .collect()
}

/// Per-layer hidden states over the prompt (prefill) and the final logits at
/// every position: 2% of the row scale + 2% relative (bf16 write-back on every
/// linear; see [`assert_close_rows`]).
#[test]
fn golden_layer_hidden_states_and_final_logits_match_hf() {
    let Some(fx) = fixtures() else { return };
    let prompt = prompt_ids(&fx);
    let rows = prompt.len();
    let mut m = model_from_fixture(&fx, rows + 16);
    m.reset();

    let mut xs = Vec::with_capacity(rows * HIDDEN);
    for &t in &prompt {
        xs.extend(m.embed_token(t));
    }
    for (li, layer) in m.layers_mut().iter_mut().enumerate() {
        xs = layer.forward_prefill(&xs, rows);
        let (shape, want) = fx.f32(&format!("layer{li}_out")).expect("layer_out");
        assert_eq!(shape, vec![rows, HIDDEN], "layer{li}_out shape");
        assert_close_rows(&format!("layer{li}_out"), &xs, &want, HIDDEN, 2e-2, 2e-2);
    }
    let (lshape, want) = fx.f32("final_logits").expect("final_logits");
    assert_eq!(lshape, vec![rows, UNPADDED], "final_logits shape");
    let mut got = Vec::with_capacity(rows * UNPADDED);
    for r in 0..rows {
        got.extend(m.head_logits(&xs[r * HIDDEN..(r + 1) * HIDDEN]));
    }
    assert_close_rows("final_logits", &got, &want, UNPADDED, 2e-2, 2e-2);
}

/// Greedy continuation: token ids must match HF `generate(do_sample=False)` EXACTLY.
#[test]
fn golden_greedy_ids_match_hf_exactly() {
    let Some(fx) = fixtures() else { return };
    let prompt = prompt_ids(&fx);
    let want: Vec<u32> = fx
        .i32("greedy_ids")
        .expect("greedy_ids")
        .1
        .iter()
        .map(|&t| t as u32)
        .collect();
    let mut m = model_from_fixture(&fx, prompt.len() + want.len() + 4);
    let got = m.greedy(&prompt, want.len());
    assert_eq!(got, want, "greedy token mismatch");
}

/// MoE block golden: `moe_x` `[1, 64]` -> `moe_out` through the first sparse
/// layer's (layer 1) MoE. Expert linears are bf16 write-back: rel 2e-2.
#[test]
fn golden_moe_block_matches_hf() {
    let Some(fx) = fixtures() else { return };
    let (_, x) = fx.f32("moe_x").expect("moe_x");
    let (_, want) = fx.f32("moe_out").expect("moe_out");
    assert_eq!(x.len(), HIDDEN);
    let moe = moe_from_fixture(&fx, 1);
    let got = moe.forward(&x);
    assert_close("moe_out", &got, &want, 2e-2, 2e-2);
}
