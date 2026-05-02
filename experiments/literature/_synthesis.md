# Literature synthesis: top-5 highest-leverage hypotheses for Intel GPUs

**Compiled:** 2026-05-02
**Scope:** 23 algorithmic / kernel-level arxiv papers reviewed (2022-2026) across speculative decoding, attention, quantization, batching, MoE, parallelism, and CPU+GPU hybrid inference. See sibling `<arxiv-id>-*.md` files for per-paper detail.
**Sister doc:** `_intel_synthesis.md` covers OpenVINO *configuration knobs* and runtime feature adoption (e.g. enable_prefix_caching, U8 KV cache). This file covers *novel research bets* — algorithmic ideas not yet wired into our worker.
**Anchored against:** baselines.md last measure 2026-05-02. alpha B390 baseline 8.85 tok/s greedy; alpha+charlie distributed-spec 17.59 tok/s. Stack OV 2026.2.0, optimum-intel 1.27, NNCF 3.1.

## Ranking method

Each candidate scored on `expected_speedup × probability_of_working_on_Intel × inverse_implementation_cost`. Speedup is the *additional* multiplier on top of current measured baseline. Probability is conditional on OV 2026.x state. Cost is engineer-weeks at tahoma's current codebase maturity.

Implementation-cost mapping (used to compute the inverse):
- TRIVIAL = 1.0 (config flag, exporter switch)
- SMALL = 0.6 (≤ 1 week)
- MEDIUM = 0.25 (~ 2-4 weeks)
- LARGE = 0.1 (~ months)

Note: Many of the OpenVINO release-note files in this directory show that Intel has *already shipped* a number of these techniques (PagedAttention, prefix caching, U8 KV cache, dynamic INT8 quant, prompt-lookup decoding). The synthesis below excludes things already adopted as baseline configuration and ranks only the *novel* algorithmic bets that aren't yet wired into our worker.

## Top 5

### 1. Sequoia hardware-aware draft-tree optimizer on top of EAGLE/Medusa
**Source papers:** [2402.12374-sequoia.md], [2401.15077-eagle-1.md], [2401.10774-medusa.md]
**Hypothesis:** A device-profiled DP solver picks a different optimal draft-tree shape on Intel iGPU vs Battlemage vs NVIDIA. Our current measured `accept=0.50` on alpha B390 (K=4 static tree) implies the tree is not Pareto-optimal — Sequoia's optimizer historically finds 1.2-1.4x extra speedup on top of any underlying speculative method.
**Expected speedup:** 1.3x on top of current 13.83 tok/s on alpha → **~18 tok/s**. Up to 1.5x on charlie or distributed where verify cost dominates differently.
**Probability of working on Intel:** 0.85. The DP solver and the verify-cost profiling are pure CPU code; the only Intel-specific risk is whether OV's tree-attention mask handling has cliffs at unusual tree shapes (likely not).
**Implementation cost:** SMALL. ~1 week. The cost-profiler runs a small (width × depth) grid against `Engine.measure_verify_cost(...)` once at startup; the DP runs in milliseconds at every speculative step (or once at session start with cached output).
**Score:** 1.3 × 0.85 × 0.6 = **0.66**
**Why this is #1:** It stacks on whatever speculative method we're already running — composes with EAGLE-1, Medusa-1, prompt-lookup, REST. The measured K=4 static-tree result is leaving 25-40% on the table by the published EAGLE-2/Sequoia evidence. The hardware-aware variant is *uniquely* well-suited to a heterogeneous Intel fleet where static tree shapes from NVIDIA reference configs are obviously wrong.

---

### 2. SmoothAttention + W4A8KV4 (QServe-style) for the Battlemage path
**Source papers:** [2405.04532-qserve.md], [2211.10438-smoothquant.md], [2404.00456-quarot.md], [2402.02750-kivi.md]
**Hypothesis:** On charlie B390 (Battlemage XMX engines have a fast INT4 weight × INT8 activation path), the optimal precision recipe is W4A8KV4 with SmoothQuant on weights and SmoothAttention on the QK path — *not* W4A16 (current default). KV4 also unlocks ~3-4x serving batch size at the same memory budget.
**Expected speedup:** 1.4-1.7x decode tok/s vs current W4A16 baseline (8.85 tok/s on alpha B390 → ~12-15 tok/s purely from KV bandwidth + activation memory traffic reduction). Bigger gains at batch >1.
**Probability of working on Intel:** 0.7. OV 2025.2+ added INT8 dynamic quant on GPU; OV 2025.4 added more multi-token kernels. The W4A8KV4 stack is composable from existing OV pieces, but full validation that XMX fully utilizes the W4 × A8 fast path at meaningful batch sizes hasn't been published. KV4 is the riskiest piece (per-channel U8 KV is what's natively supported in 2025.3+; KV4 is more exploratory).
**Implementation cost:** MEDIUM. ~2-3 weeks. SmoothQuant calibration via NNCF (already exists), QoQ-style progressive group quant (write a small calibration loop), KV4 layout (custom OV op or an INT4-pack/unpack pre/post the SDPA call). The XMX kernel itself is OV's responsibility.
**Score:** 1.55 × 0.7 × 0.25 = **0.27**
**Why this is #2:** Highest expected speedup of any *single* technique that doesn't require new kernel work, but riskier than #1 because we depend on OV's INT4×INT8 fast path actually being on the critical path of our model. Stacks with #1 but is *substitutable* with the existing W4A16 path.

---

### 3. Intel HETERO + sparse-FFN hot/cold split (PowerInfer-style on Lunar Lake)
**Source papers:** [2312.12456-powerinfer.md], [2412.11053-nitro-intel-npu.md]
**Hypothesis:** On Lunar Lake (charlie) the Llama-3 FFN has enough activation sparsity (post-SwiGLU) that a Top-K hot-neuron split can run ~70% of FFN compute on iGPU and ~30% on CPU AMX *in parallel*, hiding CPU latency behind iGPU work. This is the OpenVINO HETERO plugin's intended use case but has not been demonstrated on LLMs.
**Expected speedup:** 1.4-1.8x decode tok/s (10.33 tok/s baseline on charlie → ~15-18 tok/s) — *if* sparsity is high enough. Possibly higher because Intel CPU AMX is competitive with iGPU on small matmul.
**Probability of working on Intel:** 0.45. Two failure modes: (a) modern SwiGLU activations may not be sparse enough (PowerInfer's data was on ReLU-family models), (b) CPU↔iGPU memory hand-off via OneAPI shared-memory buffers may have larger overhead than the compute savings.
**Implementation cost:** LARGE. ~6-8 weeks. Need: offline neuron-importance profiling, a tiny predictor net, runtime split orchestration via HETERO, and a custom sparse FFN op. Interesting research but not a fast win.
**Score:** 1.6 × 0.45 × 0.1 = **0.07** (raw)
**Why this is #3 despite low score:** The score under-weights the *strategic* value. This is the unique-to-Intel technique — every other vendor (NVIDIA, AMD, Apple) has weaker CPU paths and can't credibly do hybrid inference. If it works it's a defensible Tahoma-only feature. Worth a small de-risking experiment first (just measure FFN sparsity in deployed Qwen/Llama on alpha) before committing real engineering.

---

### 4. MLA retrofit (TransMLA) for KV-bandwidth-bound Lunar Lake
**Source papers:** [2412.19437-deepseek-v3-mtp.md], [2305.13245-gqa.md], [2402.02750-kivi.md]
**Hypothesis:** Llama-3-8B has 32 attention heads with GQA factor 4 (8 KV heads). On alpha (Lunar Lake unified memory, ~70 GB/s LPDDR5X) the KV-cache traffic is the dominant decode-step cost. Retrofitting MLA via TransMLA (arXiv:2502.07864) to reduce KV traffic by another 4-6x should give 1.3-1.6x decode tok/s on alpha.
**Expected speedup:** 1.4x at the bandwidth-bound iGPU regime — alpha 8.85 → ~12.4 tok/s; charlie 10.33 → ~14.5 tok/s. Stacks with #1 (spec-decode) and #2 (KV4) somewhat: KV-traffic gains compose multiplicatively up to a memory-bandwidth-vs-compute crossover.
**Probability of working on Intel:** 0.6. TransMLA is a published recipe; the model itself is a one-time retrofit. The *runtime* support is the question — OV's attention op needs to handle the latent-projection matrices of MLA. Likely workable as a custom subgraph but not a single OV op today.
**Implementation cost:** MEDIUM. ~3 weeks total: ~1 week for a TransMLA conversion run on Llama-3-8B (one GPU-day of fine-tuning per the paper), ~2 weeks for the OV IR exporter + custom subgraph for the MLA attention block, validation.
**Score:** 1.4 × 0.6 × 0.25 = **0.21**
**Why this is #4:** MLA is the structural change with the biggest *ceiling* on memory-bound iGPU, but requires modifying the model itself (not just the runtime). Justified only if we commit to maintaining a tahoma-curated set of converted models — fine if that aligns with the model-registry plan.

---

### 5. Sarathi-Serve chunked-prefill scheduler from day-one
**Source papers:** [2403.02310-sarathi-serve.md], [2401.09670-distserve.md]
**Hypothesis:** When tahoma reaches multi-tenant serving (real workloads with concurrent prefills + decodes), naive batching will leave ≥40% of iGPU compute idle during decode-only batches. Adopting Sarathi-Serve's chunked-prefill + stall-free batching from day one prevents a future rewrite and gives ~1.5-2.0x serving capacity at the same TPOT SLO.
**Expected speedup:** 1.7x on aggregate requests/sec under realistic mixed prefill+decode load. Doesn't change single-stream tok/s.
**Probability of working on Intel:** 0.85. The technique is purely a scheduler change; OV PagedAttention + the batching primitives in OV-GenAI's `ContinuousBatchingPipeline` already support variable-shape batches.
**Implementation cost:** MEDIUM. ~2-3 weeks for the master/scheduler design — but most of that work has to happen *anyway* to support multi-tenant API serving. Adopting Sarathi-Serve's architecture upfront means it's an architectural choice, not a retrofit.
**Score:** 1.7 × 0.85 × 0.25 = **0.36**
**Why this is #5:** Different leverage axis than #1-#4. Doesn't move single-stream tok/s, but moves the *serving capacity* metric that matters once tahoma exposes the API to multiple users. Best treated as an architecture invariant: the master *must* be a continuous-batching loop that mixes prefill chunks with decode steps. Cost is "do it now correctly" vs "do it twice."

## Honorable mentions / not in top 5 but worth tracking

- **REST (retrieval-based spec decode)**: Score ~0.35 — *might* outrank #5 for code completion specifically. Underweight in synthesis because it's domain-specific (code/RAG only, not general chat).
- **Hydra over Medusa**: Score ~0.18 — strictly dominates Medusa; should be the default head architecture if we adopt Medusa-style spec-decode. Subsumed under #1's framing.
- **EAGLE-3**: Score ~0.30 — better than EAGLE-1 but requires re-training heads with multi-layer feature fusion. Re-evaluate after #1 is wired.
- **Mooncake-style KV spillover to system DRAM**: Underrated for Lunar Lake's unified memory architecture — KV spill is essentially free there. Worth a 1-week prototype.
- **FlashAttention-3 Hadamard rotation pre-quant**: Standalone gain on top of any quantization — small but cheap.
- **DistServe-style prefill/decode disaggregation across alpha+charlie**: Tahoma's heterogeneous fleet is the natural fit. Score lowered because the cross-node KV transfer over Wi-Fi/Thunderbolt is the bottleneck, not GPU compute.

## What we deliberately did *not* rank

- Generic "use FlashAttention-2/3" — already in OV SDPA.
- Generic "use AWQ for INT4 quantization" — already in `optimum-intel`.
- Generic "use PagedAttention" — already default on GPU since OV 2025.1.
- Generic "use prefix caching" — added in OV 2025.4, just needs `enable_prefix_caching=True`.
- Ring Attention — only valuable for ≥32K contexts; not a current-quarter priority.
- QuIP# 2-bit / OmniQuant W2A16 — interesting *if* the iGPU 8GB memory limit is binding, but loses to MLA-retrofit on the same constraint.
- DeepSeek-V3 / MoE patterns — out of scope until Tahoma supports MoE models (post-MVP).

## Implications for the next 3 sprints

| Sprint | Focus | Why |
|---|---|---|
| Now | #5 (Sarathi-Serve scheduler architecture) | Architectural invariant — cheaper to do right than to retrofit |
| Next | #1 (Sequoia HW-aware spec-decode tree) | Highest score, lowest risk, stacks on existing 13.83 tok/s baseline |
| After | #2 (W4A8KV4 QServe-style) and #4 (MLA retrofit) in parallel | Both are bandwidth-traffic optimizations, naturally stack |

#3 (PowerInfer HETERO) is a research bet — gate it on a small de-risking experiment (measure activation sparsity on deployed model) before committing to the full pipeline.
