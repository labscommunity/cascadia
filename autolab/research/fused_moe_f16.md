# Fused MoE at f16: where it overflows, and a fix that needs no IR regeneration

Scope: read-only study of the worktree at `autolab/inkling-fleet-perf` (2026-09-20). Nothing here was run on
hardware. Paths are repo-relative; `ENG` = `crates/cascadia-engine-sparse-moe/src/inkling`.

## 0. TL;DR

- The repo already contains a diagnosed, fixed instance of this exact bug on the expert-parallel (EP) path
  (commits c0d70c67, 450ba2c4). Two distinct f16 overflow sites were captured with replay frames:
  1. layer 8, shared expert 256: the **unweighted down-projection output** reached about -94,909 (> 65504);
  2. layer 40, expert 15: unweighted outputs were fine (max ~1,873) but **routing weights of 92-105** made the
     **weighted value ~173,194**. Routing weights grow with depth because they carry `mlp.gate.global_scale`.
  Site 2 is systematic on deep layers (any expert output element above ~650 overflows when w ~ 100) and is the
  best explanation for "layer 2 fine, deeper ranks NaN". Site 1 is sporadic (outlier channel, input dependent).
- The pipeline backend (`ENG/ov_moe.rs`) has neither of the EP path's two mitigations. Port both:
  (A) attenuate the `up` dequantisation scales by 2^-4, restore x16 on the host in f32;
  (B) divide each row's routing weights by a power of two so that sum|w| <= 1, restore on the host in f32.
- Neither needs the on-disk IR touched. (B) is pure Rust. (A) can be done in memory inside the shim's existing
  `materialize_constants` copy (`crates/cascadia-ov-genai-shim/cpp/shim.cpp:759-768`), which already duplicates
  every constant before compile. Both ship inside the `cascadia` binary; the switch is one line in
  `fleet-overrides.env`.
- Add a host-side non-finite check that returns `None` (the existing CPU fallback), so a residual overflow costs
  one slow call instead of poisoning a stream's KV with NaN.
- Open contradiction to resolve first: two places in the repo say the f32 hint does NOT compile this kernel
  (section 6). If that also holds on the fleet, "f32 is correct but slower" was a silent CPU fallback.

## 1. How the fused IR is built (`tools/inkling_moe_layer_ov.py`)

Invoked by `deploy/inkling-fleet/install.sh:198`:
`PYTHONPATH=$PYLIB python3 $HERE/tools/inkling_moe_layer_ov.py --src $PREFIX/model --layers $FUSED --layout u4zp --pad-experts 4 --skip-existing`.
Note `$HERE` is the SSD: the generator is **not copied** under `/opt/cascadia-inkling` (nothing in install.sh
installs `tools/`). What stays on the box is `$PREFIX/pylib` (numpy + openvino wheels, install.sh:174-181), the
expert bins, and `moe_ov/layer_NN/openvino_model.{xml,bin}`.

| piece | construction | dtype | lines |
|---|---|---|---|
| weights (gate, up, down) | bins' nibbles verbatim, `Constant [E, out, in/32, 32]` | `u4` | 106 |
| zero point | `Constant [E, out, in/32, 1]`, every nibble 8 | `u4` | 107 |
| scales | bins' bf16 group-32 scales converted via f32 to f16, `Constant [E, out, in/32, 1]` | `f16` | 58-60, 101 |
| decompression | `(Convert(w,f16) - Convert(zp,f16)) * scale -> Reshape [E,out,in] -> Convert(f32)` | f16 then f32 | 108-111 |
| inputs | `x [1,?,6144]` f32, `topk_indices [?,8]` i32, `routing_weights [?,8]` f32 | | 116-121 |
| expert body | `Reshape -> Tile [E*T,H] -> Reshape [E,T,H]`; `MatMul(xt, gate^T)`, `MatMul(xt, up^T)`; opset4 single-input `Swish`; `Multiply`; `MatMul(h, down^T)` | f32 in the IR | 123-131 |
| routing | zeros `[T,E]` (f32 const 0.0) -> `ScatterElementsUpdate(axis 1, idx, rw)` -> `Transpose` -> `Unsqueeze [E,T,1]` | f32 | 132-139 |
| **routing weights applied** | `ReduceSum(Multiply(down_out, routing), axis 0)`, i.e. **after** the down projection | f32 | 140 |
| shared experts | not special: ids 256, 257 of the stack, always selected, their gammas are the last two routing weights; `num_shared_expert` = 0 | | 147-152, docstring 15-17 |
| dummies | 4 experts, nibble 0x88 (value 0), scale 0; only needed for the offload path's 1% floor | | 157-167 |
| save | `ov.save_model(..., compress_to_fp16=False)`; no `rt_info` precision marks anywhere | | 252 |

E = 262, so each scale constant is 262 x 3072 x 192 x 2 B = 309 MB (gate, up) or 262 x 6144 x 96 x 2 B (down,
same size). Gate and up scale constants have identical shape; down is unique by shape.

Precision under `INFERENCE_PRECISION_HINT=f16`: the plugin converts every f32 activation in the table to f16,
with an f32->f16 reorder on `x` and `routing_weights` and f16->f32 on `y`. So `x`, gate/up outputs, the SwiGLU
product, the down output, the routing weights, the weighted product and the reduction are all f16. The generator
docstring (24-25) and `docs/architectures/inkling.md:383` both say the kernel is f16-only.
`--validate` (174-218) cannot see overflow: it feeds `x = 0.5*N(0,1)` and weights in U(0.2, 2.0) (194-199). The
docstring's claim that `--validate` reports subnormal scales is not implemented.

## 2. The math, and where f16 breaks

CPU path (never overflows; bf16 has f32's range):
- `ENG/model.rs:369-379` (decode), `400-415` (prefill), `452-470` (multi-stream rows):
  `h2 = rmsnorm(x1, mlp_norm)` -> `mlp(h2)` -> `mlp_sconv` -> added to the residual. `rmsnorm_f32` is
  `v * (1/sqrt(mean(v^2)+eps)) * w` (`ENG/mod.rs:55-66`).
- Router: f32 GEMV `[258, 6144]` (`ENG/moe.rs:508-521`), then `ENG/gate.rs:81-90`:
  `w_i = s_i/den * route_scale * global_scale`, same for the gammas, with `den` = sum of the 6 selected + 2 shared
  sigmoid scores. **Hence sum of the 8 weights = 8 * global_scale exactly, all positive.**
- Expert: `glm/ffn.rs:33-41`, bf16 rounding after each linear, `silu(g)*u` in f32. Sum in gate order, f32
  (`ENG/moe.rs:559` onwards).

Candidate overflow sites, ranked by the evidence in the repo:

| site | verdict | basis |
|---|---|---|
| weighted product / sum (`Multiply`+`ReduceSum`, line 140) | **most likely cause of the fleet NaN; systematic with depth** | layer-40 capture: w 92-105, weighted value ~173,194 (doc text added by 450ba2c4). w ~ 100 with sum(w) = 8*gs implies `global_scale` of order 10^2 at layer 40; the repo has no weight values, so the per-layer profile is unknown |
| unweighted down output | **proven, sporadic** | layer-8 capture: -94,909 on shared expert 256, exactly two non-finite outputs of 6144 (c0d70c67). Two bad outputs, not all, shows the product `silu(g)*u` was finite: an inf there would contaminate nearly every down output |
| SwiGLU product | possible, no evidence | would need \|g\|*\|u\| > 65504; if the kernel forms per-group integer dots before scaling, \|h\| > ~256 could already overflow a partial sum (speculation about kernel internals) |
| gate / up outputs | unlikely | O(1) input, int4 scales ~1e-2 |
| input `x` | very unlikely | post-rmsnorm, \|x_i\| <= \|w_i\| * sqrt(6144) = 78 \|w_i\|, so a norm weight > 835 would be needed. `mlp_norm` values are not in the repo; the layer-8 frames were finite and `ENG/f16_reference.rs:66` rounds them to f16 without issue |

One inf in `y` becomes NaN for the whole row at the next rmsnorm (inf/inf), hence all-NaN logits and token 0.
The true MoE output itself can exceed 65504 on deep layers (173,194 above), so **any** fix must also keep the
f16 output attenuated; fixing the intermediates alone is not enough.

## 3. Fixes, ranked

| # | fix | changes | IR regen | numerical risk | verdict |
|---|---|---|---|---|---|
| 1 | **(b) row-wise power-of-two weight normalisation**: send `w/F`, `F = 2^ceil(log2(max(sum\|w\|,1)))`, multiply the row of `y` by `F` in f32 | Rust only (`ov_moe.rs::forward`); `output_scaling` already exists at `ENG/ep_fused.rs:159-176` | no | none from scaling (powers of two are exact). `w/F` stays in ~[1e-3, 1]. GPU then holds a sub-convex combination, so \|y'\| <= max_i \|E_i(x)\|. Also robust if the kernel applies weights before `down` | **do first**; fixes site 2 alone |
| 2 | **(a) attenuate `up` scales by 2^-n (n=4), restore 2^n on host** | scale constant + Rust restore | no, if done in the shim (section 4) | exact while the scale stays normal (>= 6.1e-5 after division). Typical group scales ~4e-3..8e-3 give ~3e-4..5e-4 at n=4: safe. Groups below 9.8e-4 go subnormal and lose bits. The EP exporter guards at 1e-4 relative weight RMS (`tools/inkling_ep_fused_export.py:54-55`) and n=4 passed on the real model there. n=8 would push typical scales subnormal: do not. More headroom: split, e.g. up 2^-4 and down 2^-4 | **do second**; fixes site 1, shrinks product and down by 16x |
| 3 | host non-finite check -> `return None` -> CPU path (`ENG/moe.rs:543-547`, `710-714`) | 3 lines in `ov_moe.rs::forward`; EP has it at `ep_fused.rs:466` | no | none; a fallback call reads 8 experts from NVMe (~255 MB) | **always on**, with a per-layer counter |
| 4 | K=1 "compact" graph, all weighting on host in f32 (`ep_fused.rs:156-163`, exporter `compact()` 302-328: XML dims 8->1, hardlinked bin) | Rust writes a second XML + hardlink; 8x rows of `x` and `y` | no | best numerics | only if fix 1's f16 weighting proves too coarse |
| 5 | (c) `disable_fp16_compression` rt_info on chosen ops | generator | **yes** | unknown | **not recommended**: the kernel is f16-only, the matched block includes the routing `Multiply`/`ReduceSum`, and an unfused fallback would need 262 x 3 dense matrices (~30 GB at f16). Whether the matcher survives the marks cannot be determined from this repo |
| 6 | (d) bf16 | env only: the hint string is passed verbatim (`ov_moe.rs:214-216`) | no | - | nothing in `ov_moe.rs` mentions bf16; `docs/PERFORMANCE.md:39` lists it as a generic hint, but no GPU measurement exists in the repo and qwen36.rs:515-521 says this kernel has an f16-only layout. Expect a compile error -> graceful CPU fallback. A free experiment, low prior |
| 7 | (e) clamping | in-IR clamp breaks the matched pattern; host clamp is too late (inf already produced) and would cut a real 94,909 to 65,504 on exactly the massive-activation channels | - | high | rejected; fix 3 is the right safety net |

Combined effect of 1+2: everything on the GPU is bounded by max\|E_i(x)\|/16; the observed worst case 94,909
becomes 5,932 (11x headroom). Host restore factor is `F * 16`, exact. Output elements with true \|y\| below
~6.1e-5 * F * 16 (about 1.0 for F = 1024) are f16-subnormal on the GPU: absolute error <= ~5e-4 against a residual
stream of order 10^3..10^4, negligible. What is NOT fixed: f16's inherent ~2e-3 relative RMS per fused layer
(`docs/architectures/inkling.md:415`); the EP run with all 64 layers on GPU drifted to 0.05 relative RMS in a
residual with the first token identical (450ba2c4 doc text). Expect plausible but not CPU-identical text.

## 4. Applying it without regenerating IRs

| option | assessment |
|---|---|
| **A. shim rescales in memory while materialising (recommended)** | The fleet runs the materialised path (`ov_moe.rs:229`; rank.env sets no `_OFFLOAD`). `materialize_constants` already copies every constant. Add pseudo-property `CASCADIA_MOE_UP_SCALE_SHIFT=n` (erased like `CASCADIA_MATERIALIZE_CONSTANTS`, shim.cpp:780-787). Identify the `up` scale by topology, not order: f16 rank-4 constant with last dim 1 whose chain `Multiply -> Reshape -> Convert -> MatMul` feeds a `Multiply` (gate's MatMul feeds `Swish`; down's MatMul has a different scale shape). Rewrite the copy through a 65536-entry LUT built with `ov::float16(float(h)/2^n)` (round to nearest even, the same as `scale_table`, exporter 24-37), accumulate the relative RMS error like `scale_fp16` (40-56), fail the compile above 1e-4 or unless exactly one constant matched (-> `mark_failed` -> CPU path). Cost: 0 disk, 0 extra RAM, ~0.2 s per layer. The shim is compiled into `cascadia` by `cc` (shim `build.rs:70-81`), so this ships with the binary. One struct field in `OvMoe` drives both the property and the host restore factor, so they cannot desynchronise. Must be a constructor argument, not a global env read: `ep_fused.rs:421` also builds `OvMoe` over shards that may already be attenuated on disk. Refuse when `_OFFLOAD` is set (the plugin then streams scales from the .bin) |
| B. Rust patches the .bin | Feasible: the XML carries `offset`/`size` per Const and `tools/inkling_ep_rebalance.py:76-83,106-116` is a working in-place patcher (takes the 2nd of the three f16 rank-4 constants, journals, fsyncs). In place = 309 MB rewrite per layer, ~1 s, but a crash between blob and marker gives outputs wrong by 16x, which is silent garbage, not NaN, so it needs the journal and a marker file the engine reads (never an env var). Patched copy = 8.4 GB per layer, 25 GB per box, fine on 1.9 TB, ~10-20 s per layer. Only worth it if the offload path is ever used |
| C. runtime scaling through inputs | Routing weights are already an input, so fix 1 is exactly this. The `up` branch cannot be scaled this way: an extra node breaks the matched pattern and a non-constant scale defeats compressed-weight fusion |
| D. run.sh regenerates | The generator is not on the boxes (section 1); run.sh would have to carry it as a heredoc plus a new flag, run 25-50 s and rewrite 8.4 GB per layer under `Restart=always`, with a ~17-25 GB numpy/OV peak before the worker starts (fits in 61 GiB). A stdlib-only heredoc patcher in the style of `inkling_ep_rebalance.py` (no numpy/openvino needed, ~10-15 s per layer in pure Python) is the saner variant, with the same marker discipline as B. Strictly worse than A |

Keep the on-disk IR canonical (true weights). Do not add an attenuation flag to the generator while A exists,
or a regenerated box would be attenuated twice.

## 5. Plan and validation

1. `ov_moe.rs::forward`: fix 1 + fix 3, plus a max\|y'\| (pre-restore) high-water mark per layer in `OvMoeStats` and
   a one-line log of `global_scale` per layer at load. Both give headroom telemetry with no shell access.
2. Shim option A with `n` from `CASCADIA_INKLING_OV_MOE_UP_SHIFT` (default 4 when precision is f16, 0 at f32).
3. Unit tests (no hardware): LUT vs `half::f16::from_f32(h.to_f32()/16.0)` for all 65536 patterns; weight
   normalisation: `F*w' == w` bit-exact, sum\|w'\| <= 1, zero pad rows give F = 1. `ENG/f16_reference.rs` already
   models up/16 with restore (28-37, 73) and can serve as a CPU oracle of the f16 graph.
4. Rank 0 (layers 2-4, has sudo): `inkling_layer_dump` fused f16 with shift 4 against CPU. **Compare relative RMS
   or norm ratio, not cosine**: cosine is scale invariant and passes a missing x16 restore. Teeth: shift 4 with the
   restore disabled must fail at norm ratio 1/16.
5. Deep ranks (no shell): `fleet-overrides.env` is sourced as shell with `RANK` set (`fleet/run.sh:5-7`), so enable
   f16 on one deep rank at a time and compare greedy tokens with the CPU-path run of the same prompt, plus the
   fallback counter. Teeth: shift 0 and normalisation off on that rank must reproduce NaN or fallbacks > 0.
6. Then all ranks, 16 streams, long prompt, and only then re-measure tok/s. The ceiling is modest: 3 of 6
   layers per box are fused.

## 6. `ov_moe.rs` costs for multi-row batches and prefill

- **Rows > 1**: one call, `x [1, prow, 6144]`, `prow = bucket_rows(rows)`: 2, then multiples of 8 to 32, then
  multiples of 32 (`ov_moe.rs:76-82`). Pad rows copy the last row's x and ids with zero weights (405-426), so the
  kernel still computes them: 9 rows cost 16, 33 cost 64. Pointing pad rows at a dummy expert id (as
  `ep_fused.rs:111-112` does) would cut that to one zero-scale group. No upper chunking; the shim caps a tensor
  at 256 MiB (~10.9k rows).
- **Copies per call**: in: pad `to_vec` (412-414) -> fresh `ov::Tensor` + `memcpy` + `set_tensor` per input per
  call (shim.cpp:1450-1456) -> whatever the plugin does with a non-USM host pointer (not determinable here). Out:
  five FFI calls each doing `get_output_tensor`, a zero-filled `Vec<u8>` + `memcpy` (lib.rs:1551-1581), a
  byte-to-f32 `collect` (ov_moe.rs:492-495), then a truncating `to_vec` when padded (509-513). ~6 passes over
  12.6 MB for a 512-row prefill, roughly 5-10 ms per layer; trivial for decode. There is **no** remote tensor or
  USM use anywhere in the shim. Writing into the request's own input tensor and reading the output in place
  removes an allocation and two copies each way.
- **Infer request**: one per layer, created at compile (shim.cpp:798-799) and reused; `infer()` is synchronous
  (1467). `ENG/stage.rs:239-241, 257-259, 381-384, 406-408` walk the layers serially under `&mut self`, so the CPU
  idles during GPU layers and the iGPU idles during CPU layers; no overlap inside a rank.
- **Locks**: per-layer `Mutex<Runtime>` (428) is uncontended in that flow. The global `layers` table lock is held
  **across compile** (305-336), which only matters at first touch.
- **No warm-up in the worker**: `warm_ov_backends` has no call site outside examples, and `warm()` covers only
  buckets 2 and 8 (369-374). The first request compiles every fused layer lazily (8.3 GB read + materialise +
  compile each), and each new bucket (16, 24, 32, 64, ...) pays the plugin's first-shape cost (55-140 ms, ~1 s at
  32) mid-request. Prompts of varied length hit up to 32 distinct buckets at MAX_SEQ 1024. Warm at load, and
  consider fixed 128-row prefill chunks (`inkling.md:380-381`: 128 rows in 34 ms).
- **Routing is serial per row**: `forward_ov_moe` loops rows (`ENG/moe.rs:381-387`), each a rayon fork-join over
  258 tiny dot products (`dsv4/math.rs:461-470`). For hundreds of rows that is hundreds of sequential fork-joins
  before the GPU call; parallelise over rows instead.
- **pad-experts**: dead weight on the materialised path: 4 x 31.85 MB = 127 MB of unified memory per layer.

## 7. Could not be determined from the repo

- Whether the fleet's f32 run really executed on the GPU. `crates/cascadia-engine-openvino/src/qwen36.rs:515-521`
  ("No layout format available ... data_type: f32") and the c0d70c67 doc text both say the f32 hint fails to
  compile this kernel (Windows driver). A failed compile is `mark_failed` -> CPU path, reported once on stderr
  (`ov_moe.rs:526-536`). Check rank 0's journal for `fused-MoE layer N unusable` before trusting either
  "f32 is correct" or "f32 is slower than CPU".
- Per-layer `global_scale` and `mlp_norm` values, and the largest \|E_i(x)\| on layers 41-65. n=4 is supported only
  by the EP run (45 positions, all 64 layers, tokens matching CPU). Step 1's telemetry answers this.
- Where inside `moe_3gemm_fused_compressed` the routing weight is applied, whether GEMM partial sums are f16, and
  whether the plugin zero-copies host tensors on this iGPU.
