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
