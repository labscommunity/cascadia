# Selective recomputation — investigation, not shipped

**Branch:** `perf/selective-recomp-082`
**Verdict:** Investigation only. Implementation **skipped** as
unjustified for K2.6 on the current target hardware (Xeon Gold 6252
miner; Lunar Lake AI PC pair). KV memory is not the bottleneck; the
disk-bound expert dispatch is.

## What "selective recomputation" means here

Instead of caching `K`/`V` for every transformer layer, cache only
every `N`-th layer. For the others, recompute K/V from the previous
cached layer's hidden state during attention. Trades roughly `2×`
attention compute on uncached layers for `1/N` the KV memory. The
pattern shows up in vLLM and Megatron-LM as "every-2nd-layer" or
"every-4th-layer" KV caching for memory-constrained long-context
serving on GPUs.

## What this investigation asked

1. How much KV memory does K2.6 actually hold per token today?
2. Is KV memory anywhere near the binding constraint on the only two
   hardware targets we have measurements for?
3. If we burned ~`2×` SDPA compute per uncached layer, would the
   savings show up at end-to-end tok/s?

## Numbers

K2.6 MLA per-layer-per-token KV footprint (from `crates/tahoma-int4-gemm/src/shell.rs`):

| Constant      | Value                  |
|---------------|------------------------|
| `NUM_HEADS`   | 64                     |
| `QK_HEAD_DIM` | 192 (= 128 nope + 64 rope) |
| `V_HEAD_DIM`  | 128                    |
| Layers        | 60 (K2.6)              |

```
K per layer per token = 64 * 192 = 12,288 floats
V per layer per token = 64 * 128 =  8,192 floats
                                  -------
                       per layer = 20,480 floats per token

Across 60 layers:
  f32  cache = 20,480 * 60 * 4 = 4,915,200 bytes/token = ~4.8 MB/tok
  bf16 cache = 20,480 * 60 * 2 = 2,457,600 bytes/token = ~2.4 MB/tok
```

`crates/tahoma-engine-sparse-moe/src/runner.rs` on `main` today
holds `past_k` / `past_v` as `Vec<f32>` → **~4.8 MB/token** in the
shells, plus a matching layer-0 cache. The branch
`perf/a8-kv-bf16-029` (which became iter 032 on `autolab/k26-perf`)
converted these to `Vec<u16>` (bf16) → **~2.4 MB/token**, but that
change has **not yet landed on `main`**. The task brief assumes the
bf16 number.

## Where the cost actually lives

Three independent measurements from the `autolab/k26-perf` research
log say the same thing:

1. **iter 044 (compound spec-decode) root-cause:** "~94% of shell
   cost is expert dispatch which can't batch across tokens."
   Multi-token kernel speedup capped end-to-end win at +19.7%
   because everything else is dominated by demand-paging cold expert
   weights off disk.
2. **iter 064 (native bf16 SDPA):** SDPA itself is "~3% of decode
   time at `past_seq_len`~64." The entire attention block this
   investigation would speed up by *paying more compute* is a tiny
   slice of one slice of total step time.
3. **iter 062 (int4 KV cache, NEGATIVE):** scalar int4 SDPA was
   "5-9% slower than bf16 at every realistic `past_seq_len`,
   despite reading 3.55× fewer KV bytes." The per-element dequant
   cost (nibble extract + scale fmul + cvt-to-f32) ate the
   bandwidth saving. **This is the same trade selective
   recomputation makes**, only worse: instead of paying nanoseconds
   per dequanted KV row, you pay milliseconds per recomputed layer
   (`q_a_proj` + `q_b_proj` + `kv_a_proj` + `kv_b_proj` + RoPE +
   two RMSNorms — i.e. most of the non-expert non-SDPA shell
   forward).

## Memory budget reality check

K2.6 on the miner (133 GB RAM, 553 GB model on disk):

| Component                                | Resident bytes |
|------------------------------------------|----------------|
| Sparse-MoE experts (active hot set)      | up to RAM cap  |
| Int4 quantized shells (60 layers)        | ~5 GB          |
| KV at 1024 tokens (bf16, post iter 032)  | ~2.4 GB        |
| KV at 1024 tokens (f32, current main)    | ~4.8 GB        |
| KV at 256 tokens (default `max_tokens`)  | ~0.6-1.2 GB    |

At the default `max_tokens=256` (`crates/tahoma-api/src/lib.rs:208`)
the KV cache is **0.5-1% of the 133 GB RAM budget**. Even at 4096
tokens (`~10-20 GB`) it's still smaller than the expert hot set.
Cutting KV in half by recomputing every other layer saves a few GB
of RAM that the OS would otherwise spend on more cached expert
pages — but the experiment that matters (more expert pages cached)
has already been run repeatedly under different framings (cache-
aware dispatch, hot-expert buffer, expert pinning) and the win
ceiling is small because the working set is much larger than RAM.

## Why selective recomputation would lose at end-to-end tok/s

Per uncached layer per decode token, we'd run (in addition to SDPA):

- `q_a_proj` int4 GEMV  (HIDDEN=7168 → Q_LORA_RANK=1536)
- `q_b_proj` int4 GEMV  (Q_LORA_RANK → NUM_HEADS * QK_HEAD_DIM = 12288)
- `kv_a_proj` int4 GEMV (HIDDEN → KV_LORA_RANK + QK_ROPE_HEAD_DIM = 576)
- `kv_b_proj` int4 GEMV (KV_LORA_RANK → NUM_HEADS *
  (QK_NOPE_HEAD_DIM + V_HEAD_DIM) = 16384)
- RoPE on K
- two RMSNorms

That's the entire pre-SDPA shell projection chain. The decode-time
profile says this chain is non-trivial fraction of the ~6% of step
time that isn't expert dispatch. Doubling it on half the layers
(`N=2`) plausibly adds a few percent to step time. The "savings"
(less KV bandwidth into SDPA) targets a ~3%-of-decode operation and
saves a small fraction of it. **The net is almost certainly
negative on this hardware**, exactly like iter 062.

## Where selective recomputation *might* be worth re-asking

This investigation is conditional, not absolute. Selective recomp
becomes interesting when:

1. **Memory is the binding constraint, not disk bandwidth.** That
   means experts have been moved out of the demand-paging regime
   — pure pipeline-parallel across enough Intel AI PCs that each
   box holds its share of experts fully resident in RAM, with
   meaningful headroom *only* eaten by KV at long context.
2. **Context length is `>= 16K tokens`.** At 16K bf16 KV is
   `~40 GB`; on a 32 GB Lunar Lake box it would not fit even after
   pipeline-parallel sharding. That's where vLLM and Megatron-LM
   actually use this pattern.
3. **SDPA cost is no longer ~3%.** That happens when the rest of
   the per-step cost has been driven down enough (e.g. iter 051
   keystone breaker for expert batching ships and converts the 94%
   expert ceiling into something smaller). Once SDPA is 20-30% of
   step time, KV-bandwidth tradeoffs start to bite.

None of those three conditions hold on the current target hardware
or for the current default 256-token decode workload. Re-open this
question after pipeline-parallel hits an in-RAM working set and the
expert-dispatch ceiling has been broken.

## What this branch ships

- This document.
- No code change to `Runner`, `LayerState`, `shell_int4`,
  `layer0_int4`, or the C FFI.
- No change to KV memory layout, attention kernel, or wire format.

That is the entire deliverable. Shipping a skeleton that wouldn't
get exercised would just be code rot.

## Pointers

- `crates/tahoma-engine-sparse-moe/src/runner.rs` — `LayerState`
  with `past_k`/`past_v`, the structs that would have grown a
  "cached every Nth layer" variant.
- `crates/tahoma-int4-gemm/src/shell_int4.rs` —
  `shell_forward_decode_int4_with_capacity` SDPA loop that reads
  the cache.
- iter 032 (KV bf16, branch `perf/a8-kv-bf16-029`) — needs to land
  on `main` before any future KV work, gives us 2× headroom for
  free.
- iter 044, 062, 064 entries in `autolab/JOURNAL.md` on
  `autolab/k26-perf` — the bottleneck-attribution measurements
  cited above.
- iter 051 (expert dispatch batching) — the keystone that would
  change the 94%-expert-dispatch ceiling and reopen the question.
