# Campaign index

Chronological. Status legend: ✓ WIN (≥20%), ⚠ NEUTRAL (<10%), ✗ LOSS, ◌ ERROR/INCOMPLETE, 📊 BASELINE.

| Iter | Campaign | ID | Hypothesis | HW | Result | Δ vs baseline |
|------|----------|----|------------|----|--------|---------------|
| 1 | e0-baseline-single-node | e0 | re-measure single-node ov-genai + FastDraft K=5 baseline | alpha | 📊 23.01 tok/s (median of 5, 256-tok creative) | bar = ×1.20 = 27.6 tok/s |
| 2 | e1-baseline-distributed | e1 | re-measure distributed ov-dist-spec K=3 + FastDraft baseline | alpha+charlie | 📊 ✗ 9.88 tok/s (median of 5, accept=5.4%) | -57% vs e0; -64% vs bar |
| 3 | e2-k-sweep | e2 | sweep K∈{1,2,4,5,6} on creative workload | alpha+charlie | ⚠ K=1 wins at 11.78 tok/s (+19% over K=3) | -49% vs e0; -57% vs bar |
| 4 | e3-no-spec-distributed | e3 | pure PP without spec decode (ov-runtime) on creative | alpha+charlie | ⚠ 12.15 tok/s (slightly better than K=1 spec) | -47% vs e0; -56% vs bar |
| 5 | e4-layer-rebalance | e4 | 22/10 alpha-heavy split (bottleneck=charlie hypothesis) | alpha+charlie | ✗ 12.23 tok/s (in noise vs 16/16) — bottleneck is per-step OV overhead | -47% vs e0; -56% vs bar |
| 6 | e5-profile | e5 | instrument ov-runtime engine for per-task alpha_ms / wire_ms | alpha+charlie | 📊 alpha=34ms/tok, wire=55ms/tok, 11.19 tok/s — wire (charlie+net) is 62% of total | (instrumentation) |
| 7 | e6-u8kv | e6 | --ov-kv-precision u8 + --ov-dyn-quant-group 32 plugin config | alpha+charlie | ✗ 10.56 tok/s (-13%) — u8 KV regresses without PA in the IR | -54% vs e0; -62% vs bar |
| 8 | e7-factual-baseline | e7 | factual workload baseline single-node + distributed (single trial) | alpha + alpha+charlie | 📊 single 23.30 / dist 15.30 (accept 0.205) | dist 66% of single |
| 9 | e8-factual-k-sweep | e8 | K-sweep on factual workload | alpha+charlie | ✓ K=1 wins at 15.81 tok/s (+3% over K=3) | dist 68% of single, -43% vs bar |
| 10 | e9-paged-attention | e9 | PA re-export to engage LLMPipeline runtime optimizations | (export failure) | ✗ PA transform requires optimum-cli-shape IR; un-retrofittable to per-stage trace export — see DISCOVERIES D1 | (no perf data) |
| 11 | e10-stage-breakdown | e10 | full per-stage timing of dist_spec K=1 factual | alpha+charlie | 📊 alpha 27 / wire 43 / draft 17 ms/round → optimal-overlap ceiling 17.9 tok/s; bar (28) is structurally unreachable on this model+hardware — see D2 | n/a |
| 12 | m3-pseudohead-feasibility | M3.1 | embed-projection of stage_0 hidden as speculation source | alpha | ✗ 0/32 = 0% agreement — layer-16 hidden encodes input token, not next; M3 moonshot DEAD — see D3 | n/a |
| 13 | (M1 bootstrap) | M1.1 | Mixtral 8x7B INT4 single-node on alpha (expecting OOM) | alpha | 📊 0.54 tok/s — ran via shared-system-memory spillage; M1 needs more lift than scoped — see D4 | (real pain point baseline) |
| 14 | m3-pseudohead-feasibility | M3.1 v2 | re-test pseudo-head with REAL lm_head + RMSNorm at layer 16 / 22 | alpha | ✓ layer 22 = 12.5% top-1 / 34.4% top-5 (was 0% with broken embed); D3 REVISED — M3 alive at deeper cuts | (gates M3 engineering) |
| 15 | q2-pa-compile-time | Q2 | apply SDPAToPagedAttention at compile-time on v5 IR | alpha | ⚠ compile works, GPU OOM during decode (need num_kv_blocks cap); D5 — PA gives no per-token speedup for single-user, only enables tree-spec | n/a |
| 16 | q4-npu-gpu-concurrent | Q4 | NPU+GPU concurrent on charlie Lunar Lake | charlie | ✗ NPU dynamic-shape blocker; charlie GPU 8B = 19-21 tok/s LLMPipeline (was assumed 1.6× slower than alpha) | (D2 reframe data point) |
| 17 | q3-better-draft | Q3.X | swap dist_spec draft from FastDraft 150M to Llama 3.2 1B INT4 + K-sweep | alpha+charlie | ✓ **K=3 = 18.42 tok/s (+16%)** — accept rate jumped to 63%, tok/round to 2.88 | NEW HIGH; gates async overlap projection |
