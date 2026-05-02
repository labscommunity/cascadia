# Autolab loop status

**Branch:** `autolab/intel-gpu-perf` (now also runs distributed campaigns under `d0+`)
**Phase 1 Goal (DONE):** unlock hidden / novel inference-perf gains on a single Intel GPU.
**Phase 2 Goal (ACTIVE):** improve tok/s for **distributed** workloads on alpha + charlie via Thunderbolt 4. The original single-machine work missed tahoma's actual mission (run models that don't fit on one node); this phase corrects that.

## Current state

| | |
|---|---|
| Phase 1 campaigns | 62 (single-machine, completed) |
| Phase 2 campaigns | 0 (distributed, starting) |
| Last commit | docs(loop): update with c57-c59 correction notes (will rebase after first d-bench) |
| Active hypothesis | d0: re-baseline all four distributed engines with proper actual-token counting |
| Pause condition | none active |

## Phase 2 hypothesis tree (distributed)

- **d0** — distributed baselines: re-bench `pytorch`, `pytorch-tp`, `ov-runtime`, `ov-dist-spec` across alpha+charlie/TB4 with proper methodology (actual tokens via tokenizer, ≥3 runs, same prompt as single-node baselines for direct comparison).
- **d1** — network characterization: measure actual TB4 latency + bandwidth (iperf3 + ping). Establishes upper bound on per-token activation transfer.
- **d2** — identify the bottleneck: where does the slowest distributed engine spend time? Network IO vs OV compute vs OV plugin.
- **d3+** — actual perf experiments, planned after d2 finds the dominant cost. Candidate hypotheses:
  - INT8 / FP8 activation compression on the inter-stage wire
  - Micro-batching to overlap network + compute
  - Disaggregated prefill/decode (DistServe-style)
  - Tensor parallelism over TB4 (`pytorch-tp` engine, never benched)
  - Models that don't fit on one node — Mixtral 8×7B INT4 (~24 GB) or Llama 3.3 70B INT4 (~35 GB)
  - Distributed FastDraft + PL (port the Phase 1 wins to multi-stage)

## Phase 2 methodology rules (from Phase 1 corrections)

1. **Always count actual generated tokens via the LLMPipeline tokenizer** — not `max_tokens` cap. The Phase 1 bench inflated headlines 5-14× via this bug.
2. **Warm both sides of any A-vs-B comparison** with multiple full generates before timing.
3. **Use the same chat-template handling** on every engine in a comparison.
4. **Never run multiple LLMPipeline procs against the same physical GPU concurrently** — they serialise on the kernel queue.
5. **≥3 runs** per config; report best + median + variance for sub-10% claims.
6. **Be skeptical of claims that beat memory-bandwidth physics** (~150-200 tok/s for 8B INT4 on Intel iGPU).

## Phase 1 carryover (don't lose these)

- Real per-token rate on 8B INT4 single-node: ~17-30 tok/s.
- 3 verified single-machine wins: FastDraft +55%, PL +40-50% extractive, NPU concurrent +14.6%.
- 1 debunked: LLMPipeline ≈ OVModelForCausalLM at raw rate (the original "10×" was bench artifact).
- See `experiments/SUMMARY.md`, `DISCOVERIES.md`, `DECISION_MATRIX.md`.

## Phase 2 distributed-only addenda

- LLMPipeline does NOT support multi-stage. The single-node FastDraft / PL / CB wins are not directly portable to ov-runtime / ov-dist-spec without engine work.
- The original c4/c5 distributed baseline was **17.59 tok/s** (ov-dist-spec K=4 v5 shards) — likely also inflated by the cap-based bench. Will re-measure in d0.

## Iteration cadence

Continuous: each iteration is `pick → design → run → measure → commit → push`. Long-running experiments (>5 min compile, >30 min wallclock) are time-boxed at 30 min and killed if exceeded.

## Stop conditions

I'll halt the loop on any of:

1. User message containing `stop autolab` (or `pause`, `hold`, `kill autolab`).
2. Hardware unreachable (alpha + charlie both unresponsive for >15 min).
3. 5 consecutive experiment failures with no diagnosed cause.
4. Branch loses sync with main and rebase fails.

When halted I write the cause to `LOOP.md` and a comment on the draft PR.

## What I won't touch

- `tahoma/` outside this branch (existing engines must keep working).
- Production-impact actions on alpha / charlie (no destructive disk ops; no permanent config changes outside the experiment dir).
- Any `pip install` without a `requirements-experiment.txt` snapshot first.

## Status comment

PR description gets a refresh every 10 campaigns. A "campaign" = a related cluster of experiments around one hypothesis area (e.g. `OV plugin properties`, `INT4 group sizes`, `tree spec decode`).
