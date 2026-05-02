# Autolab loop status

**Branch:** `autolab/intel-gpu-perf`
**Goal:** unlock hidden / novel inference-perf gains on Intel GPUs (Arc 140V Lunar Lake, Arc B390 Battlemage). Stop only on user intervention.

## Current state

| | |
|---|---|
| Iteration count | ~370 |
| Campaigns completed | 59 |
| Last commit | exp(c59): CB multi-tenant numbers ARE accurate — no inflation |
| Active hypothesis | none — exhausted obvious search space + corrected bench artifact |
| Pause condition | none active |

## ⚠️ Major correction documented at c57-c59

The bench scripts were reporting `tok_s = max_tokens / total_dt` instead of
actual generated tokens. Discovered at c57 by counting tokens with the
LLMPipeline tokenizer.

- Single-user chat / factual: inflated 5-8×. Real rates ~17-30 tok/s on 8B INT4.
- Single-user PL extractive: inflated 5-14×. Real rates ~20-30 tok/s.
- CB multi-tenant: NOT inflated (concept-explanation prompts fill cap naturally).
- All RELATIVE wins (Discovery #1-#4) remain valid because both A and B in
  any comparison hit the same EOS behavior.

## Top achievements

- **4 Discoveries documented** in DISCOVERIES.md (all cross-platform validated):
  1. LLMPipeline 10× over OVModelForCausalLM
  2. FastDraft 150M +24% short-input chat
  3. Prompt Lookup +50-97% on extractive RAG (peak +94% at 4K input)
  4. NPU concurrent multi-model serving (16 chat + 1 classifier on one laptop)

- **Best per-workload tok/s achieved:**
  - Short factual: 134.9 (alpha + FastDraft)
  - 4K-input extractive RAG: **194.36** (charlie + PL — peak finding)
  - Multi-tenant aggregate: 559 (alpha CB batch=32)
  - 1B Llama: 211.4 (charlie GPU)

- **Quality preservation confirmed**: FastDraft + PL produce byte-identical
  greedy output as plain. Lossless.

- **Decision matrix encoded** in experiments/DECISION_MATRIX.md for engine
  selection by (input_len, output_len, content_pattern).

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
