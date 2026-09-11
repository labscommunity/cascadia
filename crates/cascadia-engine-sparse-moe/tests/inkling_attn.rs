//! Inkling attention (`InklingAttention`): hand-derivable unit tests (sliding
//! window, relative bias, log scaling on q and bias, global-only), parity
//! (prefill vs per-token, truncate, snapshot/restore) and the HF goldens
//! (`attn_x` -> `attn_out` for layer 3 [global, log scaling] and
//! `attn_out_sliding` for layer 0 [window 4]).
//!
//! Regenerate fixtures:
//!   python tools/inkling_ref/gen_fixtures.py \
//!       --out crates/cascadia-engine-sparse-moe/tests/fixtures/inkling

use std::path::PathBuf;

use cascadia_engine_sparse_moe::dsv4::st::StFile;
use cascadia_engine_sparse_moe::inkling::attn::{AttentionLayer, AttnDims, AttnWeights};
use cascadia_engine_sparse_moe::inkling::conv::ShortConv;
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
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.is_finite() && w.is_finite(),
            "{name}: non-finite value at [{i}]: got {g} want {w}"
        );
        let d = (g - w).abs();
        if d > atol + rtol * w.abs() && d > worst.1 {
            worst = (i, d, g, w);
        }
    }
    assert!(
        worst.1 == 0.0,
        "{name}: worst diff {} at [{}]: got {} want {} (atol {atol} rtol {rtol})",
        worst.1,
        worst.0,
        worst.2,
        worst.3
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

fn eye(n: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
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
    /// bf16-valued random vector (so bf16 storage is lossless).
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| to_bf16(self.next() * scale)).collect()
    }
}

/// A 1-head, 1-kv-head, D == hidden layer with explicit weights: convs are
/// zero kernels (identity via the residual), head norms are ones.
struct Tiny {
    hidden: usize,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wr: Vec<f32>, // [d_rel, hidden]
    wo: Vec<f32>,
    proj: Vec<f32>, // [d_rel, extent]
    extent: usize,
}

impl Tiny {
    fn build(&self, dims: AttnDims) -> AttentionLayer {
        let d = self.hidden;
        let w = AttnWeights {
            wq: bits(&self.wq),
            wk: bits(&self.wk),
            wv: bits(&self.wv),
            wr: bits(&self.wr),
            wo: bits(&self.wo),
            q_norm: vec![1.0; d],
            k_norm: vec![1.0; d],
        };
        let k_conv = ShortConv::new(vec![0.0; d * 4], d, 4);
        let v_conv = ShortConv::new(vec![0.0; d * 4], d, 4);
        let relpos = RelPos::new(self.proj.clone(), dims.d_rel, self.extent);
        AttentionLayer::from_parts(dims, w, k_conv, v_conv, relpos)
    }
}

/// q = k = 0 (uniform softmax), v = h, Wo = I, no bias: the output is the mean
/// of the visible inputs.
fn uniform_tiny(hidden: usize) -> Tiny {
    Tiny {
        hidden,
        wq: vec![0.0; hidden * hidden],
        wk: vec![0.0; hidden * hidden],
        wv: eye(hidden),
        wr: vec![0.0; hidden], // d_rel 1
        wo: eye(hidden),
        proj: vec![0.0; 2],
        extent: 2,
    }
}

fn ramp_inputs(hidden: usize, t: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; t * hidden];
    for p in 0..t {
        h[p * hidden] = p as f32; // channel 0 carries the position
        h[p * hidden + hidden - 1] = 1.0; // last channel constant (drives r)
    }
    h
}

#[test]
fn sliding_window_attends_only_to_the_last_window_keys() {
    let hidden = 4;
    let tiny = uniform_tiny(hidden);
    let t = 5;
    let h = ramp_inputs(hidden, t);

    // window 2: position p sees keys p-1, p -> mean of the two positions.
    let mut sl = tiny.build(AttnDims::sliding(hidden, 1, 1, hidden, 1, 2, 1e-6));
    let out = sl.forward_prefill(&h, t);
    let ch0: Vec<f32> = (0..t).map(|p| out[p * hidden]).collect();
    assert_close("sliding mean", &ch0, &[0.0, 0.5, 1.5, 2.5, 3.5], 1e-6, 0.0);

    // global: mean of all positions so far.
    let mut gl = tiny.build(AttnDims::global(hidden, 1, 1, hidden, 1, 16, 1e-6));
    let out = gl.forward_prefill(&h, t);
    let ch0: Vec<f32> = (0..t).map(|p| out[p * hidden]).collect();
    assert_close("global mean", &ch0, &[0.0, 0.5, 1.0, 1.5, 2.0], 1e-6, 0.0);

    // window 1: only the current key.
    let mut w1 = tiny.build(AttnDims::sliding(hidden, 1, 1, hidden, 1, 1, 1e-6));
    let out = w1.forward_prefill(&h, t);
    let ch0: Vec<f32> = (0..t).map(|p| out[p * hidden]).collect();
    assert_close("window-1", &ch0, &[0.0, 1.0, 2.0, 3.0, 4.0], 1e-6, 0.0);
}

/// Expected channel-0 output at position p with bias-only scores (q = k = 0):
/// softmax over visible keys of `tau · bias(dist)`, applied to v[j] = j.
fn bias_only_expected(p: usize, window: Option<usize>, prof: &[f32], tau: f32) -> f32 {
    let j0 = window.map_or(0, |w| (p + 1).saturating_sub(w));
    let scores: Vec<f32> = (j0..=p)
        .map(|j| prof.get(p - j).copied().unwrap_or(0.0) * tau)
        .collect();
    let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
    let den: f32 = e.iter().sum();
    let mut acc = 0.0f32;
    for (i, j) in (j0..=p).enumerate() {
        acc += e[i] / den * j as f32;
    }
    to_bf16(acc)
}

#[test]
fn relative_bias_steers_the_softmax() {
    // r = h[last] = 1 for every token (d_rel 1), proj = [b0, b1]: bias(0)=b0,
    // bias(1)=b1, 0 beyond extent 2.
    let hidden = 4;
    let mut tiny = uniform_tiny(hidden);
    tiny.wr = vec![0.0, 0.0, 0.0, 1.0];
    tiny.proj = vec![1.0, -1.0];
    let t = 5;
    let h = ramp_inputs(hidden, t);
    let mut gl = tiny.build(AttnDims::global(hidden, 1, 1, hidden, 1, 16, 1e-6));
    let out = gl.forward_prefill(&h, t);
    for p in 0..t {
        let want = bias_only_expected(p, None, &tiny.proj, 1.0);
        assert!(
            (out[p * hidden] - want).abs() <= 1e-3 * want.abs().max(1.0),
            "p={p}: got {} want {want}",
            out[p * hidden]
        );
    }
    // Sanity: the bias must actually matter (differs from the uniform mean).
    assert!((out[2 * hidden] - 1.0).abs() > 1e-2);
}

#[test]
fn log_scaling_scales_the_bias_on_global_layers_only() {
    let hidden = 4;
    let mut tiny = uniform_tiny(hidden);
    tiny.wr = vec![0.0, 0.0, 0.0, 1.0];
    tiny.proj = vec![1.5, -0.5];
    let (n_floor, alpha) = (1.0f32, 0.5f32); // tau = 1 + 0.5·ln(p+1) from p = 1
    let t = 6;
    let h = ramp_inputs(hidden, t);

    let mut gl = tiny.build(
        AttnDims::global(hidden, 1, 1, hidden, 1, 16, 1e-6).with_log_scaling(n_floor, alpha),
    );
    let out = gl.forward_prefill(&h, t);
    for p in 0..t {
        let tau = 1.0 + alpha * ((p + 1) as f32 / n_floor).max(1.0).ln();
        let want = bias_only_expected(p, None, &tiny.proj, tau);
        assert!(
            (out[p * hidden] - want).abs() <= 1e-3 * want.abs().max(1.0),
            "global p={p}: got {} want {want}",
            out[p * hidden]
        );
        // and the unscaled value is genuinely different for p >= 1
        if p >= 1 {
            let unscaled = bias_only_expected(p, None, &tiny.proj, 1.0);
            assert!((want - unscaled).abs() > 1e-3, "tau must change p={p}");
        }
    }

    // The same n_floor on a sliding layer: NO log scaling.
    let mut sl = tiny.build(
        AttnDims::sliding(hidden, 1, 1, hidden, 1, 3, 1e-6).with_log_scaling(n_floor, alpha),
    );
    let out = sl.forward_prefill(&h, t);
    for p in 0..t {
        let want = bias_only_expected(p, Some(3), &tiny.proj, 1.0);
        assert!(
            (out[p * hidden] - want).abs() <= 1e-3 * want.abs().max(1.0),
            "sliding p={p}: got {} want {want}",
            out[p * hidden]
        );
    }
}

/// Wq = Wk = Wv = Wo = I, no bias: an in-test reference of the full score path
/// (per-head RMSNorm on q/k, 1/D scale, tau on q, f32 softmax, bf16 Wo output).
#[test]
fn log_scaling_scales_q_and_scale_is_one_over_d() {
    let hidden = 4;
    let eps = 1e-6f32;
    let (n_floor, alpha) = (2.0f32, 0.3f32);
    let tiny = Tiny {
        hidden,
        wq: eye(hidden),
        wk: eye(hidden),
        wv: eye(hidden),
        wr: vec![0.0; hidden],
        wo: eye(hidden),
        proj: vec![0.0; 2],
        extent: 2,
    };
    let t = 6;
    let mut rng = Lcg(3);
    let h = rng.vec(t * hidden, 2.0);

    let rms = |x: &[f32]| -> Vec<f32> {
        let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let r = 1.0 / (ms + eps).sqrt();
        x.iter().map(|v| v * r).collect()
    };
    let expect = |p: usize, tau: f32| -> Vec<f32> {
        let q: Vec<f32> = rms(&h[p * hidden..(p + 1) * hidden])
            .iter()
            .map(|v| v * tau)
            .collect();
        let scores: Vec<f32> = (0..=p)
            .map(|j| {
                let k = rms(&h[j * hidden..(j + 1) * hidden]);
                q.iter().zip(&k).map(|(a, b)| a * b).sum::<f32>() / hidden as f32
            })
            .collect();
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let den: f32 = e.iter().sum();
        let mut ctx = vec![0.0f32; hidden];
        for (j, ej) in e.iter().enumerate() {
            for (c, &v) in ctx.iter_mut().zip(&h[j * hidden..(j + 1) * hidden]) {
                *c += ej / den * v;
            }
        }
        ctx.iter().map(|&v| to_bf16(v)).collect()
    };

    // Without log scaling.
    let mut gl = tiny.build(AttnDims::global(hidden, 1, 1, hidden, 1, 16, eps));
    let out = gl.forward_prefill(&h, t);
    for p in 0..t {
        assert_close(
            &format!("no-tau p={p}"),
            &out[p * hidden..(p + 1) * hidden],
            &expect(p, 1.0),
            1e-3,
            1e-2,
        );
    }
    // With log scaling: tau = 1 + 0.3·ln((p+1)/2) once p+1 > 2.
    let mut gl = tiny
        .build(AttnDims::global(hidden, 1, 1, hidden, 1, 16, eps).with_log_scaling(n_floor, alpha));
    let out = gl.forward_prefill(&h, t);
    for p in 0..t {
        let tau = 1.0 + alpha * ((p + 1) as f32 / n_floor).max(1.0).ln();
        assert_close(
            &format!("tau p={p}"),
            &out[p * hidden..(p + 1) * hidden],
            &expect(p, tau),
            1e-3,
            1e-2,
        );
        if p + 1 > 2 {
            let a = expect(p, tau);
            let b = expect(p, 1.0);
            assert!(
                a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-3),
                "tau must change the output at p={p}"
            );
        }
    }
}

/// Random GQA layer (2 heads, 1 kv head) with real conv kernels and a
/// relative bias, for the parity tests.
fn random_layer(dims: AttnDims, seed: u64) -> AttentionLayer {
    let mut rng = Lcg(seed);
    let (hd, hq, hkv, d, dr) = (
        dims.hidden,
        dims.n_heads,
        dims.n_kv_heads,
        dims.head_dim,
        dims.d_rel,
    );
    let extent = dims.window.unwrap_or(5);
    let w = AttnWeights {
        wq: bits(&rng.vec(hq * d * hd, 0.5)),
        wk: bits(&rng.vec(hkv * d * hd, 0.5)),
        wv: bits(&rng.vec(hkv * d * hd, 0.5)),
        wr: bits(&rng.vec(hq * dr * hd, 0.5)),
        wo: bits(&rng.vec(hd * hq * d, 0.5)),
        q_norm: rng.vec(d, 1.0).iter().map(|v| 1.0 + 0.2 * v).collect(),
        k_norm: rng.vec(d, 1.0).iter().map(|v| 1.0 + 0.2 * v).collect(),
    };
    let k_conv = ShortConv::new(rng.vec(hkv * d * 4, 0.4), hkv * d, 4);
    let v_conv = ShortConv::new(rng.vec(hkv * d * 4, 0.4), hkv * d, 4);
    let relpos = RelPos::new(rng.vec(dr * extent, 0.5), dr, extent);
    AttentionLayer::from_parts(dims, w, k_conv, v_conv, relpos)
}

fn parity_dims() -> Vec<AttnDims> {
    vec![
        AttnDims::sliding(8, 2, 1, 4, 2, 3, 1e-6).with_rewind(4),
        AttnDims::global(8, 2, 1, 4, 2, 32, 1e-6)
            .with_log_scaling(2.0, 0.1)
            .with_rewind(4),
    ]
}

#[test]
fn prefill_is_bit_identical_to_per_token() {
    for dims in parity_dims() {
        let t = 9;
        let h = Lcg(42).vec(t * dims.hidden, 1.0);
        let mut a = random_layer(dims.clone(), 7);
        let pre = a.forward_prefill(&h, t);
        let mut b = random_layer(dims.clone(), 7);
        let mut seq = Vec::new();
        for row in h.chunks_exact(dims.hidden) {
            seq.extend(b.forward_token(row));
        }
        assert_eq!(pre, seq, "window {:?}: prefill vs per-token", dims.window);
        // Chunked prefill continuing a history.
        let mut c = random_layer(dims.clone(), 7);
        let mut chunked = c.forward_prefill(&h[..4 * dims.hidden], 4);
        chunked.extend(c.forward_prefill(&h[4 * dims.hidden..], t - 4));
        assert_eq!(pre, chunked, "window {:?}: chunked prefill", dims.window);
    }
}

#[test]
fn truncate_then_redecode_matches_uninterrupted() {
    for dims in parity_dims() {
        let t = 10;
        let h = Lcg(8).vec(t * dims.hidden, 1.0);
        let mut layer = random_layer(dims.clone(), 99);
        let full = layer.forward_prefill(&h, t);
        // Reject the last 3 positions (<= rewind 4) and re-feed them.
        layer.truncate(t - 3);
        assert_eq!(layer.len(), t - 3);
        let redo = layer.forward_prefill(&h[(t - 3) * dims.hidden..], 3);
        assert_eq!(
            redo,
            full[(t - 3) * dims.hidden..].to_vec(),
            "window {:?}: re-decode after truncate",
            dims.window
        );
    }
}

#[test]
fn snapshot_restore_roundtrip_then_rewind() {
    for dims in parity_dims() {
        let t = 12;
        let h = Lcg(15).vec(t * dims.hidden, 1.0);
        let mut layer = random_layer(dims.clone(), 5);
        let full = layer.forward_prefill(&h, t);

        let mut layer = random_layer(dims.clone(), 5);
        let _ = layer.forward_prefill(&h[..7 * dims.hidden], 7);
        let snap = layer.snapshot();
        assert_eq!(snap.len(), 7);
        layer.reset();
        assert_eq!(layer.len(), 0);
        layer.restore(&snap);
        assert_eq!(layer.len(), 7);
        let tail = layer.forward_prefill(&h[7 * dims.hidden..], t - 7);
        assert_eq!(
            tail,
            full[7 * dims.hidden..].to_vec(),
            "window {:?}: decode after restore",
            dims.window
        );
        // Rewind 2 (within slack) after the restore path, re-feed.
        layer.truncate(t - 2);
        let redo = layer.forward_prefill(&h[(t - 2) * dims.hidden..], 2);
        assert_eq!(
            redo,
            full[(t - 2) * dims.hidden..].to_vec(),
            "window {:?}: rewind after restore",
            dims.window
        );
    }
}

#[test]
#[should_panic(expected = "rewind slack")]
fn sliding_truncate_beyond_the_ring_slack_panics() {
    let dims = AttnDims::sliding(8, 2, 1, 4, 2, 3, 1e-6).with_rewind(2);
    let mut layer = random_layer(dims.clone(), 1);
    let h = Lcg(1).vec(8 * dims.hidden, 1.0);
    let _ = layer.forward_prefill(&h, 8);
    layer.truncate(5); // 3 > 2
}

// ---- HF goldens -----------------------------------------------------------

/// Tiny config (PORT_SPEC §4).
const HIDDEN: usize = 64;
const HEADS: usize = 4;
const KV: usize = 2;
const HEAD_DIM: usize = 16;
const D_REL: usize = 4;
const WINDOW: usize = 4;
const N_FLOOR: f32 = 4.0;
const ALPHA: f32 = 0.1;
const EPS: f32 = 1e-6;

fn f(fx: &StFile, name: &str) -> Vec<f32> {
    fx.f32(name).unwrap_or_else(|e| panic!("{name}: {e}")).1
}

/// Build layer `li`'s attention from its checkpoint-named fixture tensors.
fn attn_from_fixture(fx: &StFile, li: usize, dims: AttnDims) -> AttentionLayer {
    let p = format!("model.llm.layers.{li}.attn");
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
    let k_conv = ShortConv::new(f(fx, &format!("{p}.k_sconv.weight")), c, 4);
    let v_conv = ShortConv::new(f(fx, &format!("{p}.v_sconv.weight")), c, 4);
    let (pshape, proj) = fx
        .f32(&format!("{p}.rel_logits_proj.proj"))
        .expect("rel_logits_proj.proj");
    let relpos = RelPos::new(proj, pshape[0], pshape[1]);
    AttentionLayer::from_parts(dims, w, k_conv, v_conv, relpos)
}

/// Layer 3 (global, log scaling): `attn_x` `[T, 64]` -> `attn_out` `[T, 64]`
/// (the o_proj output, before `attn_sconv`). bf16 write-back path: rel 2e-2.
#[test]
fn golden_global_attention_layer3_matches_hf() {
    let Some(fx) = fixtures() else { return };
    let (xshape, x) = fx.f32("attn_x").expect("attn_x");
    let (_, want) = fx.f32("attn_out").expect("attn_out");
    let t = xshape[0];
    assert_eq!(xshape[1], HIDDEN);
    let dims = AttnDims::global(HIDDEN, HEADS, KV, HEAD_DIM, D_REL, t + 8, EPS)
        .with_log_scaling(N_FLOOR, ALPHA);
    let mut layer = attn_from_fixture(&fx, 3, dims);
    let got = layer.forward_prefill(&x, t);
    assert_close("attn_out (layer 3, global)", &got, &want, 1e-2, 2e-2);
}

/// Layer 0 (sliding, window 4): `attn_x` -> `attn_out_sliding`.
#[test]
fn golden_sliding_attention_layer0_matches_hf() {
    let Some(fx) = fixtures() else { return };
    let (xshape, x) = fx.f32("attn_x").expect("attn_x");
    let (_, want) = fx.f32("attn_out_sliding").expect("attn_out_sliding");
    let t = xshape[0];
    let dims = AttnDims::sliding(HIDDEN, HEADS, KV, HEAD_DIM, D_REL, WINDOW, EPS)
        .with_log_scaling(N_FLOOR, ALPHA); // ignored on sliding layers
    let mut layer = attn_from_fixture(&fx, 0, dims);
    let got = layer.forward_prefill(&x, t);
    assert_close("attn_out_sliding (layer 0)", &got, &want, 1e-2, 2e-2);
}
