//! LatentMoE block — K3's routed experts run in a 3584-dim latent, not in the
//! 7168 residual stream.
//!
//! ```text
//! idx, w = gate(x)                    // gate reads HIDDEN, not the latent
//! x_lat  = routed_expert_down_proj(x) // 7168 -> 3584
//! y      = sum_k w_k * expert_k(x_lat)
//! y      = routed_expert_norm(y)      // RMSNorm on the COMBINED output
//! y      = routed_expert_up_proj(y)   // 3584 -> 7168
//! out    = y + shared_experts(x)      // shared run on HIDDEN, inter = moe_inter * n_shared
//! ```
//!
//! The router is [`crate::glm::gate::moe_gate`] unchanged — K3's `KimiMoEGate`
//! has identical semantics (sigmoid scoring, `noaux_tc` selection bias,
//! norm-topk, then `* routed_scaling_factor`, which is 1.0 for K3).

use crate::dsv4::math::{linear_bf16_w, rmsnorm, to_bf16};
use crate::glm::gate::moe_gate;
use crate::k3::expert_fp4;
use crate::k3::prof;
use crate::k3::situ::situ;

/// Shape contract for one LatentMoE layer.
#[derive(Clone, Copy, Debug)]
pub struct MoeDims {
    /// Global layer index — only used to attribute profiler routing records.
    pub layer: u32,
    pub hidden: usize,
    pub latent: usize,
    pub inter: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub n_shared: usize,
    pub scale: f32,
    pub renormalize: bool,
    pub situ_beta: f32,
    pub situ_linear_beta: Option<f32>,
    pub eps: f32,
    pub use_norm: bool,
}

/// Per-layer non-expert weights. Experts live in their own fp4 blobs.
pub struct MoeWeights {
    pub gate: Vec<f32>,
    pub e_score_correction_bias: Vec<f32>,
    pub down_proj: Vec<u16>,
    pub up_proj: Vec<u16>,
    pub norm: Vec<f32>,
    pub shared_w1: Vec<u16>,
    pub shared_w3: Vec<u16>,
    pub shared_w2: Vec<u16>,
}

/// Source of one expert's packed fp4 bytes (`w1`, `w3`, `w2` back to back).
pub trait ExpertSource {
    fn expert_bytes(&self, expert: usize) -> &[u8];
    /// Bytes per expert — what one routed selection streams on a cache miss.
    fn stride(&self) -> usize;
}

/// A flat in-memory expert set — `n_experts * expert_bytes(latent, inter)`.
/// Fine for tests and tiny models; a real layer is ~15.7 GB, so the runtime
/// uses [`MmapExperts`] instead.
pub struct FlatExperts {
    pub data: Vec<u8>,
    pub stride: usize,
}

impl ExpertSource for FlatExperts {
    fn expert_bytes(&self, expert: usize) -> &[u8] {
        &self.data[expert * self.stride..(expert + 1) * self.stride]
    }
    fn stride(&self) -> usize {
        self.stride
    }
}

/// One layer's experts, memory-mapped and demand-paged by the OS.
///
/// This is the only viable form at real scale: a layer holds 896 experts of
/// ~17.5 MB (~15.7 GB), and the full model is ~1.45 TB — far past what can be
/// read into RAM. Mapping lets the page cache hold whatever fits and stream the
/// rest from NVMe, which is also what makes throughput residency-bound rather
/// than compute-bound.
///
/// One mapping per LAYER, not per expert: 896 x 92 per-expert mappings would
/// exceed Linux's default `vm.max_map_count` (65,530) outright.
pub struct MmapExperts {
    mmap: memmap2::Mmap,
    stride: usize,
    n: usize,
}

impl MmapExperts {
    /// Map `path`, which must hold exactly `n * stride` bytes.
    pub fn open(path: &std::path::Path, stride: usize, n: usize) -> std::io::Result<Self> {
        let f = std::fs::File::open(path)?;
        let len = f.metadata()?.len() as usize;
        if len != n * stride {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{}: expected {} bytes ({n} experts x {stride}), got {len}",
                    path.display(),
                    n * stride
                ),
            ));
        }
        // SAFETY: the file is opened read-only and the mapping is never written.
        let mmap = unsafe { memmap2::Mmap::map(&f)? };
        Ok(Self { mmap, stride, n })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Bytes this layer streams if nothing is resident — the per-token IO floor
    /// when every routed expert misses the page cache.
    pub fn bytes(&self) -> usize {
        self.n * self.stride
    }

    /// Sampled page residency `(resident, probed)` of this layer's map.
    ///
    /// The real expert-cache hit signal: mmap faults are not billed to the
    /// process read counter, so an I/O-counter delta reports a bogus 100%.
    pub fn resident_pages_sampled(&self, samples: usize) -> (usize, usize) {
        crate::dsv4::expert_mmap::resident_pages_sampled_ptr(
            self.mmap.as_ptr() as usize,
            self.mmap.len(),
            samples,
        )
    }
}

impl ExpertSource for MmapExperts {
    fn expert_bytes(&self, expert: usize) -> &[u8] {
        &self.mmap[expert * self.stride..(expert + 1) * self.stride]
    }
    fn stride(&self) -> usize {
        self.stride
    }
}

/// SiTU FFN over fp4-packed weights: `w2(SiTU(w1(x), w3(x)))`.
fn fp4_expert_forward(bytes: &[u8], x: &[f32], d: MoeDims, out: &mut [f32]) {
    let sec_gate = expert_fp4::section_bytes(d.inter, d.latent);
    let sec_down = expert_fp4::section_bytes(d.latent, d.inter);
    debug_assert_eq!(bytes.len(), 2 * sec_gate + sec_down);

    let mut g = vec![0.0f32; d.inter];
    let mut u = vec![0.0f32; d.inter];
    expert_fp4::gemv(&bytes[..sec_gate], d.inter, d.latent, x, &mut g);
    expert_fp4::gemv(&bytes[sec_gate..2 * sec_gate], d.inter, d.latent, x, &mut u);

    let mut h = vec![0.0f32; d.inter];
    situ(&g, &u, &mut h, d.situ_beta, d.situ_linear_beta);
    for v in h.iter_mut() {
        *v = to_bf16(*v);
    }
    expert_fp4::gemv(&bytes[2 * sec_gate..], d.latent, d.inter, &h, out);
}

/// One token through the LatentMoE block. `x`, `out`: `[hidden]`.
pub fn moe_forward<E: ExpertSource>(
    x: &[f32],
    w: &MoeWeights,
    d: MoeDims,
    experts: &E,
    out: &mut [f32],
) {
    let _t0 = std::time::Instant::now();
    // router reads the HIDDEN stream
    let mut logits = vec![0.0f32; d.n_experts];
    for (lg, row) in logits.iter_mut().zip(w.gate.chunks_exact(d.hidden)) {
        *lg = row.iter().zip(x).map(|(&a, &b)| a * b).sum();
    }
    let sel = moe_gate(
        &logits,
        &w.e_score_correction_bias,
        d.top_k,
        d.scale,
        d.renormalize,
    );
    prof::add(prof::ROUTER, _t0);
    prof::record_routing(d.layer, &sel.idx, experts.stride());

    let _t1 = std::time::Instant::now();
    // down-project once, then accumulate the selected experts in latent space
    let mut x_lat = vec![0.0f32; d.latent];
    linear_bf16_w(x, &w.down_proj, d.latent, d.hidden, &mut x_lat);

    let mut acc = vec![0.0f32; d.latent];
    let mut eo = vec![0.0f32; d.latent];
    for (i, &e) in sel.idx.iter().enumerate() {
        fp4_expert_forward(experts.expert_bytes(e as usize), &x_lat, d, &mut eo);
        let wt = sel.weight[i];
        for (a, &v) in acc.iter_mut().zip(eo.iter()) {
            *a += wt * v;
        }
    }
    for v in acc.iter_mut() {
        *v = to_bf16(*v);
    }
    if d.use_norm {
        rmsnorm(&mut acc, &w.norm, d.eps);
    }
    linear_bf16_w(&acc, &w.up_proj, d.hidden, d.latent, out);

    // shared experts run on the HIDDEN stream, width moe_inter * n_shared
    if d.n_shared > 0 {
        let si = d.inter * d.n_shared;
        let mut g = vec![0.0f32; si];
        let mut u = vec![0.0f32; si];
        linear_bf16_w(x, &w.shared_w1, si, d.hidden, &mut g);
        linear_bf16_w(x, &w.shared_w3, si, d.hidden, &mut u);
        let mut h = vec![0.0f32; si];
        situ(&g, &u, &mut h, d.situ_beta, d.situ_linear_beta);
        for v in h.iter_mut() {
            *v = to_bf16(*v);
        }
        let mut sh = vec![0.0f32; d.hidden];
        linear_bf16_w(&h, &w.shared_w2, d.hidden, si, &mut sh);
        for (o, &v) in out.iter_mut().zip(sh.iter()) {
            *o = to_bf16(*o + v);
        }
    }
    prof::add(prof::EXPERTS, _t1);
}

/// Batch-union MoE: `rows` tokens through the block, loading each distinct
/// expert's bytes ONCE instead of once per row.
///
/// This is what makes prefill affordable. Per-token prefill re-streams the full
/// active set at every position — at low residency a few-thousand-token prompt
/// reads tens of TB. A batched pass touches each expert once, so a long prompt
/// costs about one sweep of the layer's expert set.
///
/// Bit-exact against looping [`moe_forward`]: each expert's contribution is
/// staged at its gate slot and the slots are summed in GATE order, because
/// float addition is not associative and expert-id order would reassociate it.
///
/// `xs`, `outs`: `[rows * hidden]`.
pub fn moe_forward_batch<E: ExpertSource>(
    xs: &[f32],
    w: &MoeWeights,
    d: MoeDims,
    experts: &E,
    rows: usize,
    outs: &mut [f32],
) {
    debug_assert_eq!(xs.len(), rows * d.hidden);
    debug_assert_eq!(outs.len(), rows * d.hidden);

    // gate every row, and down-project every row into the latent
    let mut sel = Vec::with_capacity(rows);
    let mut lat = vec![0.0f32; rows * d.latent];
    let mut logits = vec![0.0f32; d.n_experts];
    for r in 0..rows {
        let x = &xs[r * d.hidden..(r + 1) * d.hidden];
        for (lg, row) in logits.iter_mut().zip(w.gate.chunks_exact(d.hidden)) {
            *lg = row.iter().zip(x).map(|(&a, &b)| a * b).sum();
        }
        sel.push(moe_gate(
            &logits,
            &w.e_score_correction_bias,
            d.top_k,
            d.scale,
            d.renormalize,
        ));
        linear_bf16_w(
            x,
            &w.down_proj,
            d.latent,
            d.hidden,
            &mut lat[r * d.latent..(r + 1) * d.latent],
        );
    }

    for s in &sel {
        prof::record_routing(d.layer, &s.idx, experts.stride());
    }

    // invert the selection: expert -> the (row, slot) pairs that chose it
    let mut want: Vec<Vec<(usize, usize)>> = vec![Vec::new(); d.n_experts];
    for (r, s) in sel.iter().enumerate() {
        for (k, &e) in s.idx.iter().enumerate() {
            want[e as usize].push((r, k));
        }
    }

    // one pass per distinct expert; stage results at their gate slots
    let mut slots = vec![0.0f32; rows * d.top_k * d.latent];
    let mut eo = vec![0.0f32; d.latent];
    for (e, hits) in want.iter().enumerate() {
        if hits.is_empty() {
            continue;
        }
        let bytes = experts.expert_bytes(e);
        for &(r, k) in hits {
            fp4_expert_forward(bytes, &lat[r * d.latent..(r + 1) * d.latent], d, &mut eo);
            let base = (r * d.top_k + k) * d.latent;
            slots[base..base + d.latent].copy_from_slice(&eo);
        }
    }

    // reduce in gate order, then norm / up-proj / shared per row
    let mut acc = vec![0.0f32; d.latent];
    let si = d.inter * d.n_shared;
    let (mut g, mut u, mut hid, mut sh) = (
        vec![0.0f32; si],
        vec![0.0f32; si],
        vec![0.0f32; si],
        vec![0.0f32; d.hidden],
    );
    for r in 0..rows {
        acc.fill(0.0);
        for k in 0..d.top_k {
            let wt = sel[r].weight[k];
            let base = (r * d.top_k + k) * d.latent;
            for (a, &v) in acc.iter_mut().zip(&slots[base..base + d.latent]) {
                *a += wt * v;
            }
        }
        for v in acc.iter_mut() {
            *v = to_bf16(*v);
        }
        if d.use_norm {
            rmsnorm(&mut acc, &w.norm, d.eps);
        }
        let out = &mut outs[r * d.hidden..(r + 1) * d.hidden];
        linear_bf16_w(&acc, &w.up_proj, d.hidden, d.latent, out);

        if d.n_shared > 0 {
            let x = &xs[r * d.hidden..(r + 1) * d.hidden];
            linear_bf16_w(x, &w.shared_w1, si, d.hidden, &mut g);
            linear_bf16_w(x, &w.shared_w3, si, d.hidden, &mut u);
            situ(&g, &u, &mut hid, d.situ_beta, d.situ_linear_beta);
            for v in hid.iter_mut() {
                *v = to_bf16(*v);
            }
            linear_bf16_w(&hid, &w.shared_w2, d.hidden, si, &mut sh);
            for (o, &v) in out.iter_mut().zip(sh.iter()) {
                *o = to_bf16(*o + v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> MoeDims {
        MoeDims {
            layer: 0,
            hidden: 8,
            latent: 32,
            inter: 32,
            n_experts: 4,
            top_k: 2,
            n_shared: 1,
            scale: 1.0,
            renormalize: true,
            situ_beta: 4.0,
            situ_linear_beta: Some(25.0),
            eps: 1e-5,
            use_norm: true,
        }
    }

    fn bf(n: usize, k: f32) -> Vec<u16> {
        (0..n)
            .map(|i| half::bf16::from_f32(((i as f32) * k).sin() * 0.2).to_bits())
            .collect()
    }

    fn weights(d: MoeDims) -> MoeWeights {
        MoeWeights {
            gate: (0..d.n_experts * d.hidden)
                .map(|i| ((i as f32) * 0.37).sin())
                .collect(),
            e_score_correction_bias: vec![0.0; d.n_experts],
            down_proj: bf(d.latent * d.hidden, 0.11),
            up_proj: bf(d.hidden * d.latent, 0.13),
            norm: vec![1.0; d.latent],
            shared_w1: bf(d.inter * d.n_shared * d.hidden, 0.17),
            shared_w3: bf(d.inter * d.n_shared * d.hidden, 0.19),
            shared_w2: bf(d.hidden * d.inter * d.n_shared, 0.23),
        }
    }

    fn experts(d: MoeDims) -> FlatExperts {
        let stride = expert_fp4::expert_bytes(d.latent, d.inter);
        let mut data = vec![0u8; stride * d.n_experts];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i * 31 % 251) as u8;
        }
        // keep every E8M0 scale near 2^0 so the test values stay in range
        for e in 0..d.n_experts {
            let base = e * stride;
            let mut off = base;
            for (o, inn) in [
                (d.inter, d.latent),
                (d.inter, d.latent),
                (d.latent, d.inter),
            ] {
                let nib = o * inn / 2;
                for b in data[off + nib..off + expert_fp4::section_bytes(o, inn)].iter_mut() {
                    *b = 127;
                }
                off += expert_fp4::section_bytes(o, inn);
            }
        }
        FlatExperts { data, stride }
    }

    /// Counts how many times each expert's bytes are fetched.
    struct Counting {
        inner: FlatExperts,
        hits: std::cell::RefCell<Vec<usize>>,
    }

    impl ExpertSource for Counting {
        fn expert_bytes(&self, expert: usize) -> &[u8] {
            self.hits.borrow_mut()[expert] += 1;
            self.inner.expert_bytes(expert)
        }
        fn stride(&self) -> usize {
            self.inner.stride()
        }
    }

    #[test]
    fn batch_union_loads_each_expert_once_and_matches_per_token() {
        let d = dims();
        let w = weights(d);
        let rows = 8usize;
        let xs: Vec<f32> = (0..rows * d.hidden)
            .map(|i| ((i as f32) * 0.19).cos())
            .collect();

        // per-token reference
        let ex = experts(d);
        let mut want = vec![0.0f32; rows * d.hidden];
        for r in 0..rows {
            moe_forward(
                &xs[r * d.hidden..(r + 1) * d.hidden],
                &w,
                d,
                &ex,
                &mut want[r * d.hidden..(r + 1) * d.hidden],
            );
        }

        // batched, counting fetches
        let c = Counting {
            inner: experts(d),
            hits: std::cell::RefCell::new(vec![0; d.n_experts]),
        };
        let mut got = vec![0.0f32; rows * d.hidden];
        moe_forward_batch(&xs, &w, d, &c, rows, &mut got);

        assert_eq!(got, want, "batch-union must be bit-exact vs per-token");

        let hits = c.hits.borrow();
        let touched = hits.iter().filter(|&&h| h > 0).count();
        // each DISTINCT expert is fetched exactly once, however many rows chose it
        assert!(
            hits.iter().all(|&h| h <= 1),
            "an expert was fetched more than once: {hits:?}"
        );
        // and per-token would have fetched rows * top_k times in total
        assert!(
            touched <= d.n_experts && touched < rows * d.top_k,
            "no dedup: touched {touched} of {} experts for {} per-token fetches",
            d.n_experts,
            rows * d.top_k
        );
    }

    #[test]
    fn output_is_finite_and_shaped() {
        let d = dims();
        let (w, ex) = (weights(d), experts(d));
        let x: Vec<f32> = (0..d.hidden).map(|i| (i as f32 * 0.29).cos()).collect();
        let mut out = vec![0.0f32; d.hidden];
        moe_forward(&x, &w, d, &ex, &mut out);
        assert!(out.iter().all(|v| v.is_finite()), "non-finite: {out:?}");
    }

    #[test]
    fn dropping_shared_experts_changes_the_output() {
        // Guards against the shared branch being silently skipped.
        let d = dims();
        let (w, ex) = (weights(d), experts(d));
        let x: Vec<f32> = (0..d.hidden).map(|i| (i as f32 * 0.29).cos()).collect();
        let mut with = vec![0.0f32; d.hidden];
        moe_forward(&x, &w, d, &ex, &mut with);
        let mut without = vec![0.0f32; d.hidden];
        let d0 = MoeDims { n_shared: 0, ..d };
        moe_forward(&x, &w, d0, &ex, &mut without);
        assert_ne!(with, without, "shared expert contribution went missing");
    }

    #[test]
    fn top_k_selection_bias_steers_the_router() {
        // A large bias on one expert must force it into the selection.
        let d = dims();
        let (mut w, ex) = (weights(d), experts(d));
        w.e_score_correction_bias = vec![0.0, 0.0, 0.0, 100.0];
        let x: Vec<f32> = (0..d.hidden).map(|i| (i as f32 * 0.29).cos()).collect();
        let mut logits = vec![0.0f32; d.n_experts];
        for (lg, row) in logits.iter_mut().zip(w.gate.chunks_exact(d.hidden)) {
            *lg = row.iter().zip(&x).map(|(&a, &b)| a * b).sum();
        }
        let sel = moe_gate(&logits, &w.e_score_correction_bias, d.top_k, d.scale, true);
        assert!(
            sel.idx.contains(&3),
            "biased expert not selected: {:?}",
            sel.idx
        );
        let mut out = vec![0.0f32; d.hidden];
        moe_forward(&x, &w, d, &ex, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
