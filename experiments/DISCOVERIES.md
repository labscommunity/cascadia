# Discoveries

Novel / surprising / undocumented findings worth saving forever. Format: cite the experiment that produced the evidence, give the surprise plainly, explain why it's saveable.

## D1 — OV 2026.1 PagedAttention transform requires optimum-cli-shaped IRs and is not retrofittable to per-stage trace exports

**Setup:** OpenVINO 2026.1.0 + openvino-genai 2026.1.0 on alpha (Battlemage Arc B390). Llama 3.1 8B Instruct INT4, exported per-stage via rainier's `scripts/export_cached_shards_v5.py` (torch.jit.trace + nncf + apply_make_stateful_transformation). Tried to apply `openvino._offline_transformations.paged_attention_transformation` at the end of the export to engage LLMPipeline-class GPU-plugin optimizations on multi-stage IRs.

**Finding:** Two stacked failures:

1. The transformation's first check is `!model->get_variables().empty()` — it requires a stateful model. The v5 script branched away from `apply_make_stateful_transformation` for the PA path (commented as "PA expects pre-stateful"). That branch was wrong for OV 2026.1.
2. After patching to apply stateful first, the next check is `unregistered_parameters.str().empty()` — the transform deletes `attention_mask` from the parameter list (PA absorbs masking) but leaves a graph reference, failing validation.

The OV error suggests `optimum-cli export openvino --task text-generation-with-past`. optimum-cli's exporter produces a model shape PA accepts — but it is monolithic, no per-stage support exists.

**Why this matters:** D7 in the prior python autolab ([branch autolab/intel-gpu-perf](https://github.com/labscommunity/tahoma/tree/autolab/intel-gpu-perf)) hoped PA was a config flag. It is not. Reclaiming the LLMPipeline per-stage win on multi-stage requires either custom torch.export flows or forking optimum-cli — multi-week. **For the current Llama-style trace-based per-stage export, runtime PA optimizations are unreachable.** This re-prioritizes the moonshot stack toward async pipelining and per-host TP, away from per-stage compute optimization via PA.

**Source experiments:** `experiments/e9-paged-attention/`.

---

## D2 — 2-stage PP on Llama 8B INT4 over alpha+charlie/TB4 has a STRUCTURAL upper bound at ~17-18 tok/s, vs single-node alpha at 23 tok/s. Beating single-node for single-user sequential decode is impossible without changing model, hardware concurrency, or speculation regime.

**Setup:** alpha (Battlemage Arc B390 dGPU) + charlie (Lunar Lake 140V iGPU) over direct Thunderbolt 4 (8.75 Gbps, 0.142 ms RTT). Llama 3.1 8B Instruct INT4. v3/v5 per-stage shards from rainier's exporter. Engines: `ov-runtime`, `ov-dist-spec` with FastDraft 150M companion. Workloads: 256-tok creative + 256-tok technical/factual.

**Finding:** Detailed per-stage instrumentation (e10) on the K=1 factual config — the leaderboard high at 15.81 tok/s — shows per-round costs:
- alpha stage_0 (16 INT4 layers on B390): 27 ms
- wire (network + charlie stage_1 16 layers on 140V iGPU): 43 ms (wait dominates over network's 0.142 ms)
- draft.feed (alpha 150M FastDraft incl. rejection rewinds): 17 ms

Even with **optimal async overlap** (drafts during charlie wait), the lower bound per round is `alpha_stage_0 + max(draft, wire) + reconcile = 27 + 43 + 7 = 77 ms` ⇒ **17.9 tok/s**.

Single-node alpha with `ov-genai + FastDraft K=5` does the same workload at 23.30 tok/s (e0/e7).

**Why distributed cannot beat single-node here:**

1. Distributed adds a serialization: per-token cost = alpha_compute + wire + charlie_compute (~70 ms minimum even with charlie-side optimizations). Single-node = alpha_compute_full (~43 ms).
2. The "extra" cost (~27 ms) cannot be hidden because alpha's stage_0 must complete BEFORE charlie's stage_1 starts (autoregressive dependency), and the next token needs charlie's output before alpha can start its next stage_0.
3. Speculation breaks the dependency only when the draft accepts. FastDraft 150M accepts 5% on creative and 38% on factual — not enough amortization to overcome the serialization tax.
4. Charlie's iGPU is ~1.6× slower per layer than alpha's dGPU. Layer rebalance (e4) doesn't help — the bottleneck just moves to alpha.
5. Per-stage paged-attention re-export is blocked at the export pipeline level (D1) — the LLMPipeline-class GPU optimizations are not reachable.

**Implication:** The autolab perf bar (single-node × 1.20 = ~28 tok/s for distributed Llama 3.1 8B INT4 on this hardware) is structurally unreachable. To beat the bar, the campaign must shift to:

- **Models too big for single-node** (Mixtral 8x7B INT4 ~12 GB doesn't fit alpha B390's 12 GB GPU memory comfortably with KV cache; Llama 3.3 70B INT4 ~35 GB definitely doesn't). Distributed wins by default because single-node OOMs.
- **Within-host hardware concurrency** (alpha GPU + alpha NPU running different attention heads of the same forward in parallel) — major engineering.
- **Speculation against an in-graph draft head** that runs on alpha's stage_0 hidden state via the embedding matrix (early-exit pseudo-head) — multi-week engineering, quality loss.

**Source experiments:** `experiments/e10-stage-breakdown/`, with supporting evidence in e0/e2/e3/e7/e8.

---

## D3 — early-exit pseudo-head via embed-matrix projection at layer 16/32 has 0% agreement with full forward — kills the M3 moonshot at validation

**Setup:** alpha B390 dGPU. Llama 3.1 8B INT4 v3 stage_0 (16 layers + embed) compared against full 32-layer Llama 3.1 8B INT4 single-step decode. After each token, project stage_0's last hidden state through `embed_tokens.weight^T` (bf16 → f16) to get pseudo-logits; argmax; compare to argmax of full-model output. 10-token prompt + 32 generated tokens.

**Finding:** Agreement rate = **0/32 = 0.0%**. Inspection shows pseudo_token at position N typically equals real_token at position **N-1** — the layer-16 residual stream is still primarily encoding the *input* token, not yet evolved to predict the next one.

**Why this matters:** the M3 moonshot proposed using stage_0's hidden state as a free speculation source so the pipeline could overlap stage_0 of token N+1 with stage_1 of token N. For that to work, the speculation accept rate needs to be at least 40% — it is 0%. The multi-day engineering is non-viable.

This is consistent with published interpretability work showing the "next-token prediction" head structure in Llama-class models lives in the upper layers (typically the last 25% — layers 24-32 for Llama 8B). Layer 16 hidden is still in the "feature elaboration" regime.

**Variants that might work** (deferred — none are quick):
- Pseudo-head at a DEEPER cut (e.g. layer 24) — would require a different shard split (24/8) and re-export. Probably 30-40% agreement based on pattern. Would need re-test.
- Learned linear head trained to predict next token from layer-16 hidden — multi-week training.
- Use a small companion model (Llama 3.2 1B) as the speculator on alpha NPU — d4 in the prior python autolab saw -75% due to cross-device sync overhead.

**Why this is worth saving:** the obvious-sounding "use embed as pseudo-head" trick is a TRAP for low-effort distributed-PP speculation. Anyone trying to revive this idea on Llama-class models should use deeper-layer cuts — and even then, accept rates will trade off against per-stage compute time.

**Source experiments:** `experiments/m3-pseudohead-feasibility/`.

---

## D4 — Mixtral 8x7B INT4 (28 GB OV format) RUNS on alpha B390 12 GB GPU at 0.54 tok/s via implicit shared-system-memory spillage; proves M1's "doesn't fit single-node" premise needs more nuance

**Setup:** alpha (Battlemage Arc B390 dGPU, 12 GB GPU memory, 32 GB system RAM). Model `OpenVINO/Mixtral-8x7B-Instruct-v0.1-int4-ov`, `openvino_model.bin` size 28.7 GB (larger than the headline "INT4" suggests because of unpacked scales + zero points + tokenizer artifacts). Loaded via tahoma `ov-genai` engine (LLMPipeline path).

**Finding:** **Single-node monolithic loaded successfully and ran 34-token generation in 63.4 seconds = 0.54 tok/s.** Did NOT OOM. The OV GPU plugin transparently spills weights between GPU memory and shared system memory; spillage is functional but ~25-50× slower than fully-on-GPU inference for a model of this scale.

**Why this matters:**

1. The M1 moonshot ("distributed wins because single-node OOMs") is **partially wrong** — single-node doesn't OOM, it just gets ~0.5 tok/s. The real value prop is "distributed gives **usable** tok/s for big models that single-node makes unusably slow."
2. To beat 0.54 tok/s with distributed Mixtral is trivially easy — anything above 1 tok/s is a 2× win. The real bar should be "make Mixtral feel responsive": ~5-10 tok/s would unlock a chat-quality experience.
3. The blocker for landing this win is the **export pipeline**: rainier's `export_cached_shards_v6_mixtral.py` requires raw HF safetensors (~90 GB to download), and our OV-format INT4 .bin can't be split with our existing tooling. Either: (a) download HF weights and rebuild via rainier (~hours over wifi), or (b) write a custom splitter for the existing OV monolithic IR (multi-day, the python autolab d6 attempted this and didn't complete).

**Implication for the moonshot stack:** M1 is **viable and high-value** (current 0.54 tok/s is a real pain point on this hardware) but **larger lift than initially scoped**. The cleanest path is downloading HF Mixtral safetensors and using rainier v6 to produce 2-stage shards, then running ov-runtime distributed.

**Source experiments:** session-1 close, M1 bootstrap log at `experiments/MOONSHOT_M1_BOOTSTRAP_RESULT.log`.
