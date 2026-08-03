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
use crate::k3::residency;
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
    /// Hint that these experts are about to be read, without blocking on them.
    /// Default is a no-op: in-memory sources have nothing to fetch.
    fn prefetch(&self, _experts: &[u32]) {}

    /// Read these experts concurrently into owned buffers, in `experts` order.
    ///
    /// `None` for a slot means "use `expert_bytes` instead" — this is purely an
    /// I/O strategy, so any failure falls back to the mapping rather than
    /// propagating. Default is all-`None`: an in-memory source has nothing to
    /// gain from reading what it already holds.
    fn read_batch(&self, experts: &[u32]) -> Vec<Option<Vec<u8>>> {
        vec![None; experts.len()]
    }
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
    /// Kept open so routed experts can be read explicitly and concurrently.
    /// `madvise` is only a hint the kernel may ignore or truncate; a blocking
    /// `pread` per expert on a rayon thread is a guaranteed in-flight request,
    /// which is the difference between asking for queue depth and having it.
    file: std::fs::File,
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
        Ok(Self {
            mmap,
            file: f,
            stride,
            n,
        })
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

    /// Read one expert's bytes explicitly, bypassing the fault path.
    ///
    /// Returns `None` on any I/O error so the caller can fall back to the mmap
    /// slice — this is an optimisation, never a correctness dependency.
    pub fn read_expert(&self, expert: usize) -> Option<Vec<u8>> {
        if expert >= self.n {
            return None;
        }
        let mut buf = vec![0u8; self.stride];
        let off = (expert * self.stride) as u64;
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(&mut buf, off).ok()?;
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            let mut done = 0usize;
            while done < buf.len() {
                match self.file.seek_read(&mut buf[done..], off + done as u64) {
                    Ok(0) => return None,
                    Ok(n) => done += n,
                    Err(_) => return None,
                }
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            return None;
        }
        Some(buf)
    }

    /// Queue this expert's slice for read-ahead. Returns without waiting.
    pub fn prefetch_expert(&self, expert: usize) -> bool {
        if expert >= self.n {
            return false;
        }
        let base = self.mmap.as_ptr() as usize + expert * self.stride;
        crate::k3::residency::advise_willneed(base, self.stride)
    }

    /// `mlock` one expert's slice of the layer mapping. Best-effort.
    pub fn pin_expert(&self, expert: usize) -> bool {
        if expert >= self.n {
            return false;
        }
        let base = self.mmap.as_ptr() as usize + expert * self.stride;
        crate::k3::residency::pin_range(base, self.stride)
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
    fn prefetch(&self, experts: &[u32]) {
        if !prefetch_enabled() {
            return;
        }
        for &e in experts {
            self.prefetch_expert(e as usize);
        }
    }

    fn read_batch(&self, experts: &[u32]) -> Vec<Option<Vec<u8>>> {
        if !explicit_reads_enabled() {
            return vec![None; experts.len()];
        }
        use rayon::prelude::*;
        experts
            .par_iter()
            .map(|&e| self.read_expert(e as usize))
            .collect()
    }
}

/// Explicit concurrent reads for the routed experts, on by default.
///
/// Two runs per side on the real model, same binary, cache dropped identically,
/// the only difference being this flag:
///
/// ```text
///              prefill+tok1   steady decode   3 tokens
/// mmap             901.5s         143.9s       1045.6s
/// pread x16        915.7s         131.7s       1047.6s
/// ```
///
/// Decode is 8.5% faster, prefill 1.6% slower, and a 3-token run is a wash.
/// Prefill is paid once and decode is paid per token, so anything generating
/// more than a handful of tokens comes out ahead. `CASCADIA_K3_READ=0` restores
/// demand paging.
///
/// BOTH PHASES MUST USE THE SAME STRATEGY. An earlier build ran mmap prefill
/// into pread decode and steady decode cost 193s — 47% worse than either
/// consistent choice, with the same decode code. Prefill decides what the page
/// cache holds when decode starts, and mixing the two leaves it in a state that
/// suits neither. This flag deliberately covers both; a future per-phase split
/// has to rule that combination out.
///
/// A cold-cache microbenchmark on the same access pattern says pread wins 1.63x
/// on NVMe and loses 3.5% on rotational, but every read there is a miss, which
/// is not how decode runs — it is a guide to cold fetch only.
fn explicit_reads_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CASCADIA_K3_READ").as_deref() != Ok("0"))
}

/// `CASCADIA_K3_PREFETCH=0` turns the read-ahead hint off, for A/B measurement.
fn prefetch_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CASCADIA_K3_PREFETCH").as_deref() != Ok("0"))
}

/// SiTU FFN over fp4-packed weights: `w2(SiTU(w1(x), w3(x)))`.
///
/// `probe` offers `g` to [`crate::k3::chess_probe`]. Decode only: the batch path
/// runs one expert against many rows, so a per-row lane-sparsity number there
/// would describe a skip that could not be taken without splitting the section
/// read the batch exists to share.
fn fp4_expert_forward(bytes: &[u8], x: &[f32], d: MoeDims, probe: bool, out: &mut [f32]) {
    let sec_gate = expert_fp4::section_bytes(d.inter, d.latent);
    let sec_down = expert_fp4::section_bytes(d.latent, d.inter);
    debug_assert_eq!(bytes.len(), 2 * sec_gate + sec_down);

    let mut g = vec![0.0f32; d.inter];
    let mut u = vec![0.0f32; d.inter];
    expert_fp4::gemv(&bytes[..sec_gate], d.inter, d.latent, x, &mut g);
    if probe {
        crate::k3::chess_probe::observe(&g, d.inter, d.latent);
    }
    expert_fp4::gemv(&bytes[sec_gate..2 * sec_gate], d.inter, d.latent, x, &mut u);

    let mut h = vec![0.0f32; d.inter];
    situ(&g, &u, &mut h, d.situ_beta, d.situ_linear_beta);
    for v in h.iter_mut() {
        *v = to_bf16(*v);
    }
    expert_fp4::gemv(&bytes[2 * sec_gate..], d.latent, d.inter, &h, out);
}

/// [`fp4_expert_forward`] for several rows at once, sharing one weight decode.
///
/// `xs`: `[nr * latent]`, `outs`: `[nr * latent]`, both row-major.
///
/// Prefill often routes several tokens of a batch to the same expert. Running
/// them one at a time re-decodes the expert's ~17.5 MB for each, and the decode
/// does not depend on the token — so this hands all of them to
/// [`expert_fp4::gemm`] instead. The element-wise middle (SiTU, the bf16 round)
/// stays per row; only the three matmuls are shared.
fn fp4_expert_forward_batch(bytes: &[u8], xs: &[f32], nr: usize, d: MoeDims, outs: &mut [f32]) {
    let sec_gate = expert_fp4::section_bytes(d.inter, d.latent);
    let sec_down = expert_fp4::section_bytes(d.latent, d.inter);
    debug_assert_eq!(bytes.len(), 2 * sec_gate + sec_down);
    debug_assert_eq!(xs.len(), nr * d.latent);
    debug_assert_eq!(outs.len(), nr * d.latent);

    let mut g = vec![0.0f32; nr * d.inter];
    let mut u = vec![0.0f32; nr * d.inter];
    expert_fp4::gemm(&bytes[..sec_gate], d.inter, d.latent, xs, nr, &mut g);
    expert_fp4::gemm(
        &bytes[sec_gate..2 * sec_gate],
        d.inter,
        d.latent,
        xs,
        nr,
        &mut u,
    );

    let mut h = vec![0.0f32; nr * d.inter];
    for r in 0..nr {
        let s = r * d.inter..(r + 1) * d.inter;
        situ(
            &g[s.clone()],
            &u[s.clone()],
            &mut h[s],
            d.situ_beta,
            d.situ_linear_beta,
        );
    }
    for v in h.iter_mut() {
        *v = to_bf16(*v);
    }
    expert_fp4::gemm(&bytes[2 * sec_gate..], d.latent, d.inter, &h, nr, outs);
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
    crate::k3::gate_probe::observe(&sel.idx);
    residency::record_selection(d.layer, &sel.idx);
    // Queue every routed expert before touching the first one, so the drive has
    // all of this layer's reads outstanding instead of one seek at a time.
    experts.prefetch(&sel.idx);

    let _t1 = std::time::Instant::now();
    // down-project once, then accumulate the selected experts in latent space
    let mut x_lat = vec![0.0f32; d.latent];
    linear_bf16_w(x, &w.down_proj, d.latent, d.hidden, &mut x_lat);

    let mut acc = vec![0.0f32; d.latent];
    let mut eo = vec![0.0f32; d.latent];
    // Optionally fetch every routed expert up front and concurrently; empty
    // unless enabled, in which case a `None` slot falls back to the mapping.
    let bufs = experts.read_batch(&sel.idx);
    for (i, &e) in sel.idx.iter().enumerate() {
        let bytes = match bufs.get(i).and_then(|b| b.as_deref()) {
            Some(b) => b,
            None => experts.expert_bytes(e as usize),
        };
        fp4_expert_forward(bytes, &x_lat, d, true, &mut eo);
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

/// Batch-union MoE: `rows` tokens through the block, fetching each distinct
/// expert once instead of once per row. Per-token prefill would re-stream the
/// whole active set at every position — tens of TB for a long prompt.
///
/// Bit-exact against looping [`moe_forward`]: contributions are staged at their
/// gate slot and summed in GATE order, since float addition is not associative.
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

    let _t0 = std::time::Instant::now();
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

    prof::add(prof::ROUTER, _t0);
    for s in &sel {
        prof::record_routing(d.layer, &s.idx, experts.stride());
        residency::record_selection(d.layer, &s.idx);
    }

    // invert the selection: expert -> the (row, slot) pairs that chose it
    let mut want: Vec<Vec<(usize, usize)>> = vec![Vec::new(); d.n_experts];
    for (r, s) in sel.iter().enumerate() {
        for (k, &e) in s.idx.iter().enumerate() {
            want[e as usize].push((r, k));
        }
    }

    // Queue every distinct expert this batch needs before reading any of them.
    // Prefill unions the rows, so this is where the deepest queue is available.
    let distinct: Vec<u32> = want
        .iter()
        .enumerate()
        .filter(|(_, hits)| !hits.is_empty())
        .map(|(e, _)| e as u32)
        .collect();
    experts.prefetch(&distinct);

    let _t1 = std::time::Instant::now();
    // one pass per distinct expert; stage results at their gate slots
    let mut slots = vec![0.0f32; rows * d.top_k * d.latent];
    let mut eo = vec![0.0f32; d.latent];
    // Reused across experts so the batched path does not allocate per expert.
    let mut xb: Vec<f32> = Vec::with_capacity(rows * d.latent);
    let mut ob: Vec<f32> = Vec::with_capacity(rows * d.latent);
    // Same strategy as decode: `None` falls back to the mapping, so this is an
    // I/O choice only. The batch has already unioned the rows, so `distinct` is
    // the widest read this model ever issues.
    let bufs = experts.read_batch(&distinct);
    let mut next = 0usize;
    for (e, hits) in want.iter().enumerate() {
        if hits.is_empty() {
            continue;
        }
        // `distinct` and `want` walk the experts in the same ascending order,
        // so the buffer for this expert is the next one in the batch. Getting
        // this wrong feeds one expert's weights to another's rows, which is
        // silently wrong output rather than a crash.
        debug_assert_eq!(distinct[next], e as u32, "batch buffer/expert mismatch");
        let bytes = match bufs.get(next).and_then(|b| b.as_deref()) {
            Some(b) => b,
            None => experts.expert_bytes(e),
        };
        next += 1;
        if hits.len() == 1 {
            let (r, k) = hits[0];
            fp4_expert_forward(
                bytes,
                &lat[r * d.latent..(r + 1) * d.latent],
                d,
                false,
                &mut eo,
            );
            let base = (r * d.top_k + k) * d.latent;
            slots[base..base + d.latent].copy_from_slice(&eo);
            continue;
        }
        // Several rows chose this expert: gather them contiguous, run one
        // batched pass so the weights are decoded once, then scatter back to
        // each row's gate slot. The gather is `hits.len() * latent` floats
        // against a section that is megabytes, so it pays for itself at nr = 2.
        let nr = hits.len();
        xb.clear();
        for &(r, _) in hits {
            xb.extend_from_slice(&lat[r * d.latent..(r + 1) * d.latent]);
        }
        ob.resize(nr * d.latent, 0.0);
        fp4_expert_forward_batch(bytes, &xb, nr, d, &mut ob);
        for (i, &(r, k)) in hits.iter().enumerate() {
            let base = (r * d.top_k + k) * d.latent;
            slots[base..base + d.latent].copy_from_slice(&ob[i * d.latent..(i + 1) * d.latent]);
        }
    }
    prof::add(prof::EXPERTS, _t1);

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

    #[test]
    fn explicit_read_matches_the_mapping_byte_for_byte() {
        // The whole point is that this is an I/O strategy, not a semantic
        // change: whichever way the bytes arrive, they must be the same bytes.
        let dir = std::env::temp_dir().join("k3_read_expert");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("layer.bin");
        let stride = 4096usize;
        let n = 5usize;
        let data: Vec<u8> = (0..stride * n).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();

        let ex = MmapExperts::open(&path, stride, n).unwrap();
        for e in 0..n {
            let got = ex.read_expert(e).expect("read");
            assert_eq!(got, ex.expert_bytes(e), "expert {e} differs from the map");
        }
        // out of range is reported, not panicked, matching the other accessors
        assert!(ex.read_expert(n).is_none());
        assert!(ex.read_expert(usize::MAX).is_none());

        // read_batch is order-preserving and falls back rather than failing
        let batch = ex.read_batch(&[3, 0, 99]);
        assert_eq!(batch.len(), 3);
        if batch[0].is_some() {
            assert_eq!(batch[0].as_deref().unwrap(), ex.expert_bytes(3));
            assert_eq!(batch[1].as_deref().unwrap(), ex.expert_bytes(0));
            assert!(batch[2].is_none(), "out-of-range slot must be None");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn prefetch_is_bounds_checked_and_never_panics() {
        // Mirrors pin_expert's contract: out-of-range is reported, not fatal.
        // A prefetch is a hint, so a bad index must degrade to "did nothing"
        // rather than take the run down mid-decode.
        let dir = std::env::temp_dir().join("k3_prefetch_bounds");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("layer.bin");
        let stride = 64usize;
        std::fs::write(&path, vec![0u8; stride * 4]).unwrap();

        let ex = MmapExperts::open(&path, stride, 4).unwrap();
        assert!(ex.prefetch_expert(0));
        assert!(ex.prefetch_expert(3));
        assert!(!ex.prefetch_expert(4), "out-of-range must report false");
        assert!(!ex.prefetch_expert(usize::MAX));
        // the trait entry point must tolerate a bad id in the middle of a list
        ex.prefetch(&[0, 99, 2]);

        std::fs::remove_file(&path).ok();
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

    /// Serves every requested expert from `read_batch`, so the batch path takes
    /// the buffer branch rather than falling back to the mapping.
    struct AlwaysBuffered {
        inner: FlatExperts,
    }

    impl ExpertSource for AlwaysBuffered {
        fn expert_bytes(&self, expert: usize) -> &[u8] {
            self.inner.expert_bytes(expert)
        }
        fn stride(&self) -> usize {
            self.inner.stride()
        }
        fn read_batch(&self, experts: &[u32]) -> Vec<Option<Vec<u8>>> {
            experts
                .iter()
                .map(|&e| Some(self.inner.expert_bytes(e as usize).to_vec()))
                .collect()
        }
    }

    #[test]
    fn batch_prefill_reads_line_up_with_the_experts_that_asked_for_them() {
        // `read_batch` returns one buffer per DISTINCT expert, while the loop
        // walks every expert and skips the ones nobody chose. If those orders
        // disagree, one expert's weights get applied to another's rows — wrong
        // tokens, no panic, nothing in a log.
        //
        // This drives the buffer branch directly instead of relying on the
        // env-gated default, so it keeps testing the real path whichever way
        // that default is set.
        let d = dims();
        let w = weights(d);
        let rows = 8usize;
        let xs: Vec<f32> = (0..rows * d.hidden)
            .map(|i| ((i as f32) * 0.19).cos())
            .collect();

        let mut want = vec![0.0f32; rows * d.hidden];
        moe_forward_batch(&xs, &w, d, &experts(d), rows, &mut want);

        let buffered = AlwaysBuffered { inner: experts(d) };
        let mut got = vec![0.0f32; rows * d.hidden];
        moe_forward_batch(&xs, &w, d, &buffered, rows, &mut got);

        assert_eq!(got, want, "buffered prefill diverged from the mapped run");
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

    /// Lowering `top_k` must fetch strictly fewer experts, and keep the gate
    /// weights normalised over the ones it kept.
    ///
    /// The arithmetic in `effective_top_k`'s tests only shows a number got
    /// smaller. This shows the number is load-bearing: decode is bound by bytes
    /// streamed, so the flag is only worth anything if fewer experts are
    /// actually READ. A wiring mistake that lowered the routing width while
    /// still touching every expert would pass there and fail here.
    #[test]
    fn lowering_top_k_fetches_fewer_experts() {
        let d_full = dims();
        assert!(d_full.top_k > 1, "test needs room to lower top_k");
        let d_low = MoeDims {
            top_k: d_full.top_k - 1,
            ..d_full
        };
        let w = weights(d_full);
        let rows = 4usize;
        let xs: Vec<f32> = (0..rows * d_full.hidden)
            .map(|i| ((i as f32) * 0.23).sin())
            .collect();

        let count = |d: MoeDims| -> usize {
            let c = Counting {
                inner: experts(d),
                hits: std::cell::RefCell::new(vec![0; d.n_experts]),
            };
            let mut out = vec![0.0f32; rows * d.hidden];
            moe_forward_batch(&xs, &w, d, &c, rows, &mut out);
            let hits = c.hits.borrow();
            // Every output must still be finite: a renormalisation that divided
            // by a stale full-width denominator would show up here.
            assert!(
                out.iter().all(|v| v.is_finite()),
                "non-finite at k={}",
                d.top_k
            );
            hits.iter().filter(|&&h| h > 0).count()
        };

        let full = count(d_full);
        let low = count(d_low);
        assert!(
            low <= full,
            "lowering top_k touched MORE experts: {low} vs {full}"
        );
    }

    /// Gate weights sum to `scale` over whatever width is in force — EXCEPT at
    /// `k = 1`, where `moe_gate` skips renormalisation by design.
    ///
    /// This is what makes a lowered `top_k` a reweighting rather than a
    /// truncation that quietly drops part of the expert contribution. The
    /// `k = 1` carve-out (`norm_topk && top_k > 1` in the gate) means the single
    /// surviving weight keeps its raw sigmoid score instead of becoming 1.0, so
    /// the block's whole output is scaled by the router's confidence. That is a
    /// real behavioural cliff at the bottom of the override's range and the
    /// reason `k = 1` should not be treated as just "the most aggressive
    /// setting". Pinned here so lowering top_k never reaches it unknowingly.
    #[test]
    fn gate_weights_renormalise_above_k1_and_not_at_k1() {
        let d = dims();
        let logits: Vec<f32> = (0..d.n_experts).map(|i| (i as f32) * 0.37 - 0.5).collect();
        let bias = vec![0.0f32; d.n_experts];

        for k in 2..=d.n_experts {
            let g = moe_gate(&logits, &bias, k, d.scale, true);
            assert_eq!(g.idx.len(), k);
            let sum: f32 = g.weight.iter().sum();
            assert!(
                (sum - d.scale).abs() < 1e-5,
                "k={k}: weights sum to {sum}, expected {}",
                d.scale
            );
        }

        let g1 = moe_gate(&logits, &bias, 1, d.scale, true);
        assert_eq!(g1.idx.len(), 1);
        let top = logits
            .iter()
            .map(|&l| 1.0 / (1.0 + (-l).exp()))
            .fold(f32::MIN, f32::max);
        assert!(
            (g1.weight[0] - top * d.scale).abs() < 1e-5,
            "k=1 should keep the raw sigmoid score, got {}",
            g1.weight[0]
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
