# Autolab — Distributed Pipeline-Parallel Inference Performance

**Branch:** `autolab/distributed-perf`
**Mission:** Make distributed pipeline-parallel inference *faster than monolithic single-node* for Llama-class models on Intel hardware. Stretch goal: enough headroom to absorb network latency (TB4 / wifi) and still win.
**Hardware:**
- alpha — Battlemage Arc B390 dGPU, NPU (Meteor Lake-class), 32 GB RAM
- charlie — Lunar Lake 140V iGPU, NPU 4 (~48 TOPS), 32 GB RAM
- alpha ↔ charlie via direct Thunderbolt 4 (8.75 Gbps measured, 0.142 ms RTT)
- beta — third Intel AI PC, wifi-connected (no TB to alpha/charlie)
**Software:** OpenVINO 2026.1.0 + openvino-genai 2026.1.0 + Rust port (post-PR-3)
**Primary baseline:** single-node `ov-genai + FastDraft K=5` (re-measured in e0; see leaderboard)
**Bar to beat:** single-node best × 1.20, sustained, on the same workload
**Workload assumption:** **single-user sequential requests.** Micro-batching / continuous batching wins do **not** count toward the bar — the primary user of this codebase issues prompts one at a time. Multi-tenant strategies (CB, request fan-in) are out of scope for the perf bar; they may still appear as informational side notes if discovered.

## What this autolab is allowed to change

Everything load-bearing in pipeline-parallel inference:

- **Wire format / shard format**: free to break the dist-spec frame protocol, write new shard versions (`v8`, `v9`, ...), retire formats that don't pull weight.
- **Rainier export scripts**: free to write `export_cached_shards_v8_*.py` variants. Only land the ones that prove out.
- **NPU support**: free to extend the OV shim + Rust engines with `--device NPU` if it boosts throughput (e.g. NPU draft + GPU target topology).
- **Topologies**: PP, TP, hybrid PP+TP, MoE-style, 3-node — all on the table.

## Operating mode

- Long-running autonomous loop. Silent until interrupted.
- Each campaign: hypothesis → setup → bench → measure → record. Results live under `experiments/eXX-name/`.
- After every campaign, update `INDEX.md` + `LEADERBOARD.md` + (if applicable) `DISCOVERIES.md`.
- Push every ~10 commits. Early draft PR to `main` for visibility.
- Negative results are valuable — record them.

## Definitions

- **Monolithic** = single-node config that gives the best tok/s for a single user (currently `ov-genai + FastDraft K=5`).
- **Distributed** = ≥2 ranks, model split across nodes by layer (PP), heads (TP), experts (EP), or hybrid.
- **tok/s** = actual generated tokens / wall-clock time. Counted via tokenizer, NOT max_tokens cap (lesson from prior autolab — see python branch's SUMMARY).
- **Win** = ≥20% improvement over current campaign's baseline. Anything <10% is noise (per prior variance measurements).

## Living docs

- [INDEX.md](INDEX.md) — chronological list of every campaign with one-line result
- [LEADERBOARD.md](LEADERBOARD.md) — best-of for each (model × workload × topology) tuple
- [DISCOVERIES.md](DISCOVERIES.md) — novel / surprising findings worth saving forever

## At a glance — what to read first

If you're picking this up cold:
1. **DISCOVERIES.md D2** — the central finding: distributed PP for 8B INT4 is structurally bounded below single-node on this hardware.
2. **MOONSHOT_PROPOSALS.md M1** — the cleanest moonshot remaining: distributed Mixtral 8x7B (won't fit single-node alpha B390 12GB GPU).
3. **LEADERBOARD.md** — final perf numbers across (model × workload × topology).

In progress at session-1 close: Mixtral 8x7B INT4 OV-format download via raw curl on alpha (HF huggingface_hub + cas-bridge stalled at 16 MB; curl is at ~18 GB / ~28 GB total at session close, ETA 15 more minutes). Once complete, the M1 experiment is bench-ready — see `MOONSHOT_PROPOSALS.md` and `/tmp/m1_mixtral_singlenode.sh` for the entry-point script.

## Session-1 final (2026-05-03, ~6.5 hours, 15 commits, 12 campaigns, 3 discoveries)

The headline story: **distributed pipeline-parallel inference of Llama 3.1 8B INT4 on alpha (B390 dGPU) + charlie (LL 140V iGPU) over Thunderbolt 4 cannot beat single-node alpha for single-user sequential decode**, regardless of layer split, K-tuning, plugin config, NPU experiments, or speculation regime tested.

The structural ceiling is ~17-18 tok/s vs single-node 23 tok/s (e0/e10/D2). The 1.20× perf bar is unreachable on this model+hardware combination without changing one of:

1. **Model size** — distributed wins by default for models too big for one node (Mixtral 8x7B INT4, Llama 70B INT4). Blocked this session on Mixtral cas-bridge download stalls.
2. **Hardware concurrency** — within-host TP via alpha NPU + alpha GPU on the same forward. Multi-week engineering, big potential.
3. **Speculation regime** — needs reliable next-token prediction from a fast local source. Pseudo-head via embed projection at layers 16/22 was tested and FAILED (0% agreement, D3) — the residual stream still encodes the input token, not the next.

## Discoveries

- **D1** — OV 2026.1 PA transformation requires optimum-cli-shape IRs and is not retrofittable to per-stage trace-based exports (e9).
- **D2** — 2-stage PP on 8B INT4 over alpha+charlie has a structural upper bound at ~17-18 tok/s; beating single-node for single-user sequential decode is impossible without changing model, hardware concurrency, or speculation (e0/e10).
- **D3** — Embed-matrix projection of intermediate hidden state (layer 16 or 22 of 32) gives 0% agreement with next-token argmax — the fast-speculation shortcut M3 is dead at validation (m3-pseudohead-feasibility).

## Session-1 status (2026-05-03)

After 11 campaigns the central finding is **D2** in DISCOVERIES.md: for Llama 3.1 8B INT4 on alpha (B390 dGPU) + charlie (LL 140V iGPU) over TB4, single-user sequential 2-stage PP is structurally bounded at ~17-18 tok/s vs single-node alpha at 23 tok/s. The bar (28 tok/s) is unreachable on this model+hardware without changing one of: model size, hardware concurrency model, speculation regime.

Levers tried + results:
- spec K-sweep on creative + factual workloads (e2, e8): K=1 wins; higher K wastes target compute when accept rate is low. Drop default K from 5 → 1 for the dist_spec engine — modest free win.
- Layer rebalance 22/10 (e4): no-op — bottleneck shifts but doesn't shrink.
- U8 KV cache plugin config (e6): -13% regression without paged-attention in the IR.
- NPU stage_1 (e7-side): can't compile — dynamic shape rejected.
- NPU draft (e9-side): -67% — cross-device sync too expensive.
- Paged-attention re-export (e9): blocked at the OV transformation level — D1 in DISCOVERIES.

Levers NOT yet tried (in scope for session 2):
- **Models that don't fit single-node** (Mixtral 8x7B INT4 — download stalled on cas-bridge / Llama 70B INT4 — not downloaded). Cleanest moonshot: distributed wins by default because single-node OOMs.
- **Within-host hardware concurrency**: alpha NPU + alpha GPU running different attention heads of the same forward in parallel. Multi-week engineering, biggest potential.
- **Early-exit pseudo-head speculation**: project stage_0 hidden via embed matrix → speculative pseudo_token → pipeline next stage_0 while charlie verifies. Multi-day, quality risk.
- **Async draft + target overlap**: estimated +5-15% on factual K=1, marginal on creative. Engineered, then deferred — real win but doesn't cross the bar (needs to compose with one of the above).
