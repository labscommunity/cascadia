# c36: Adaptive K via assistant_confidence_threshold at long input — REVISED

## Setup
Llama 3.1 8B INT4 + FastDraft 150M on alpha B390 GPU (and charlie 140V).
~1024-token input passage + summary instruction, 32-token output.
Sweep `assistant_confidence_threshold` ∈ {0.3, 0.5, 0.7}.

`assistant_confidence_threshold` is mutually exclusive with
`num_assistant_tokens` (set the threshold in GenerationConfig, do NOT set
num_assistant_tokens).

## Results — ALL with same prompt seed (re-baselined for fair comparison)

### alpha B390

| Engine config                              | tok/s | vs plain |
|--------------------------------------------|------:|---------:|
| LLMPipeline plain                          | 20.24 | (baseline) |
| LLMPipeline + FastDraft K=5 (fixed)         | 18.83 |    -7%   |
| **LLMPipeline + FastDraft adaptive thr=0.3** | **21.80** |  **+8%** |
| LLMPipeline + FastDraft adaptive thr=0.5    | 21.33 |    +5%   |
| LLMPipeline + FastDraft adaptive thr=0.7    | 20.96 |    +4%   |

### charlie 140V

| Engine config                              | tok/s | vs plain |
|--------------------------------------------|------:|---------:|
| LLMPipeline plain                          | 20.41 | (baseline) |
| LLMPipeline + FastDraft adaptive thr=0.3    | 21.04 |    +3%   |
| LLMPipeline + FastDraft adaptive thr=0.5    | 21.12 |    +3.5% |

## Findings

1. **Adaptive K with thr=0.3 is the best spec-decode config at long input**
   on alpha (+8% over plain, +16% over fixed K=5). Fixed K=5 is
   actively hurting (-7% vs plain).
2. **The cross-platform win is much smaller on charlie** (+3% vs alpha's +8%).
   Lunar Lake's tighter per-step compute leaves less headroom for spec
   savings.
3. **Initial c36 first-pass over-stated the win** at +18% because the FastDraft
   K=5 baseline was on a different prompt seed. With same-seed baselines the
   win shrinks to +8% (alpha) / +3% (charlie).

## Recommendation for tahoma

For long inputs (≥1K tokens), `assistant_confidence_threshold=0.3` is a
modest but real improvement over `num_assistant_tokens=5`. Worth exposing
as an `--ov-spec-threshold` flag, but the magnitude is small enough that
the default `--spec-k` (fixed K=5) for short input + opt-in adaptive K for
long input is the cleanest API split.

## Other comparisons in this campaign

We also confirmed FastDraft K=5 is a NET LOSS at long input + short
output (-7% on alpha, c34 corroborated). The decision matrix:

| Input | Output | Best engine                           |
|-------|--------|---------------------------------------|
| <100  | any    | FastDraft K=5 (or K=3 for long out)   |
| ≥1K   | <64    | **Adaptive K thr=0.3** (or plain)     |
| ≥1K   | ≥128   | plain — FastDraft brings nothing      |
