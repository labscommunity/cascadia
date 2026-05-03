# Moonshot proposals — session 2

After session 1 (11 campaigns, [SUMMARY.md](SUMMARY.md)), the conclusion is hard: distributed 2-stage PP for Llama 3.1 8B INT4 on alpha+charlie/TB4 single-user single-token has a **structural ceiling at ~17-18 tok/s vs single-node alpha's 23 tok/s** ([DISCOVERIES.md D2](DISCOVERIES.md)). The bar is unreachable on this model+hardware.

What remains are TRUE moonshots — none are quick. Listed in order of expected impact × tractability.

## M1 — distributed wins for models too big to fit single-node

**Goal:** Demonstrate distributed inference for Mixtral 8x7B INT4 (~12 GB) or Llama 3.3 70B INT4 (~35 GB), where alpha alone OOMs and distributed is the *only* option. No need to "beat" single-node — single-node fails.

**Why it's a moonshot:** flips the value prop entirely. Tahoma stops being a "make 8B faster" tool and becomes "run 70B on a pair of consumer Intel laptops" tool. Real differentiator.

**What's blocking it now:** model weights. Mixtral OV-int4 download stalls on cas-bridge (HF Xet redirect) — same blocker as the prior python autolab d6. Llama 70B INT4 not downloaded.

**Plan:**
1. Schedule the Mixtral download via raw HTTP (curl/aria2 over wifi, retrying on failure) instead of huggingface_hub which gives up on cas-bridge slowness. Or use `hf_xet` env vars to disable Xet and force the legacy LFS endpoint.
2. As fallback: try `OpenVINO/Llama-3.3-70B-Instruct-int4-ov` if it exists; smaller alternatives like `OpenVINO/codestral-22b-int4-ov`.
3. Re-export as 2-stage shards via rainier's exporter (the v3/v5 scripts are model-agnostic enough to handle Mistral/Mixtral architectures with config tweaks). For MoE (Mixtral), all 8 experts need to land in one stage's IR — single-stage OOM is exactly the use case.
4. Run distributed; demonstrate single-node OOM via `--device GPU` on alpha alone and capture the OOM error.
5. Bench distributed tok/s. Even 5 tok/s is a milestone if single-node can't run at all.

**Engineering:** mostly export-pipeline work and config tuning. ~1-2 days.

## M2 — within-host hardware concurrency: alpha NPU + alpha GPU running different attention heads of the same forward in parallel

**Goal:** True tensor parallelism *within a single host*. Alpha B390 GPU and alpha NPU compute disjoint subsets of the attention heads (and MLP slices) for the same token, with all-reduce happening over the shared host memory bus (~80 GB/s on Lunar Lake) instead of TB4 (8.75 Gbps = 1.1 GB/s, ~70× slower). Same for charlie. This breaks the per-token serialization tax that bounds 2-stage PP — both compute units are doing useful work on the same token simultaneously.

**Why it's a moonshot:** The python autolab tested cross-device on Llama 3.2 1B (D4: NPU at 91% of GPU speed). Within-host TP is the structural fix for the bottleneck D2 identifies.

**What's blocking it now:**
- The OV plugin model is per-device. To run a single attention op across GPU+NPU, the model must be exported with the heads partitioned across two sub-models, and the engine must sync host-memory tensors between them.
- NPU requires static-shape models. Per-token forward at variable seq_len would need re-compile on each shape, which is too expensive. Workaround: bucketed seq_lens (e.g. 1, 4, 16, 64, 256) with a separate NPU compile per bucket.
- Each device's intermediate KV must stay coherent across forward passes. NPU's stateful KV is supported but the cross-device sync for partial KV is not a documented OV pattern.

**Plan:**
1. Re-export stage_0 as TWO sub-IRs: `stage_0_gpu` (heads 0-15) and `stage_0_npu` (heads 16-31). Each contains its own ReadValue/Assign for its half of the KV.
2. Engine code: per-token forward = (alpha GPU sub-stage, alpha NPU sub-stage) launched concurrently → both produce partial hidden states → host-memory all-reduce-sum → send combined hidden to charlie.
3. Same for charlie.
4. Validate quality on a 256-tok creative prompt (no degradation expected — TP is mathematically equivalent to non-TP).
5. Bench.

**Engineering:** Multi-week. Requires shim primitives for cross-device tensor sharing, possibly NPU-specific compile, custom all-reduce.

**Expected gain:** Per-token compute halves on each device (since each does 1/2 the heads). Per-token = max(gpu_half, npu_half) + small all-reduce. If GPU at 25 ms / NPU at 35 ms (per-device), per-token ~35 ms/device + 1 ms host-mem all-reduce = 36 ms/device. With charlie similar: total per-token = 36 (alpha) + 1 (TB) + 36 (charlie) = 73 ms ⇒ **13.7 tok/s** — actually slightly worse than current 15.81 because both halves still serialize at the host level.

Hmm — within-host TP only helps if alpha GPU and alpha NPU are FASTER together than alpha GPU alone. Need to measure NPU's actual throughput on this workload to confirm.

→ Probable MEDIUM moonshot. Worth investigating but not a guaranteed win.

## M3 — early-exit pseudo-head speculation

**Goal:** Use the embedding matrix (already loaded as part of stage_0 on alpha) as a pseudo-LM head. Project stage_0's hidden state through embed → pseudo_token. Pipeline next-round stage_0 with pseudo_token while charlie verifies real_token via stage_1's lm_head. If pseudo == real, pipelined work was free.

**Why it's a moonshot:** Could 1.5-2× distributed throughput IF pseudo agreement rate is high. The agreement rate is unknown — needs an offline experiment first (~2 hours of Python). If agreement is ≥50%, the engineering payoff justifies multi-day work.

**Step 1 (cheap, 2 hours):** Write a Python script that:
- Loads embed_tokens.weight from `C:\cascadia\models\llama-3.1-8b-src\` safetensors
- For a sample prompt, runs ov-genai LLMPipeline single-step at a time, captures stage_0 (layer-16) hidden state OR uses a pre-stage_0-cut intermediate snapshot
- Projects hidden through embed.T → pseudo_logits
- Compares pseudo_token = argmax(pseudo) to real_token = argmax(LLMPipeline.last_logits)
- Reports agreement rate over 256 tokens for both creative and factual prompts

**Step 2 (multi-day, only if step 1 shows agreement ≥40%):** Engineering:
- Load embed matrix on alpha at engine init
- Add a small OV graph that does `hidden @ embed.T` on alpha GPU
- Refactor dist_spec spec_decode to start next-round stage_0 with pseudo_token
- Reconciliation logic for mismatches

**Risk:** If pseudo agreement is <30%, the speculation is too unreliable to be net-positive. Prior literature on early-exit (e.g. "ChunkAttention", "DeepFusion") suggests 16-layer of 32 has high but not perfect agreement on intermediate prediction.

## M4 — change the workload: focus on multi-pass generation patterns

**Goal:** Pivot the bar definition. Instead of "single-user single-prompt", consider workloads where distributed has natural advantages even at single-user:
- **Reasoning chains**: prompt → (output1) → reuse output1 as new prompt → (output2) → ... Each round is a SEPARATE single-user request, but they reuse KV. PP can pipeline rounds asynchronously.
- **Multi-turn chat with long history**: prefix-cached KV. Each turn is short output. Distributed can prefill prefix once and amortize.
- **Long context generation (16K+)**: per-token compute scales with sequence length²; distributed amortizes the prefill better than single-node when prefill dominates.

**Why it's a moonshot:** the user-facing perf bar should match real chat patterns, not synthetic 256-token continuation.

**Plan:** Define new workload classes (multi-turn chat, long-context Q&A), measure single-node + distributed on each, find the workload where distributed actually wins. Communicate the win clearly (e.g. "distributed gives lower TTFT on long context").

## Recommended order

1. **M1 first** (1-2 days) — biggest, cleanest impact. The download blocker is the entire path; if Mixtral arrives, the rest is straightforward.
2. **M3 step 1** in parallel (2 hours) — cheap experiment that gates the multi-day commitment.
3. **M4** as a parallel framing exercise (no engineering) — re-tells the value prop honestly.
4. **M2** only if M1 doesn't pan out and M3 step 1 looks negative — multi-week committed effort.
