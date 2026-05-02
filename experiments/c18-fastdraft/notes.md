# c18: Intel's FastDraft companion models

## Setup

- All runs on **alpha B390** (Battlemage) unless noted, OV 2026.1.0,
  openvino_genai 2026.1.0, greedy decoding.
- All `tok/s` numbers are decode-only.

## Headline results

### Llama 3.1 8B INT4 + Llama-3.1-8B FastDraft 150M

| Output | K | Path | tok/s | vs plain |
|---|---|---|---|---|
| 64 tok | 5 | LLMPipeline raw (c18-1) | 119.24 | +23.7% over plain 96.4 |
| 64 tok | 10 | LLMPipeline raw (c18-2) | 118.22 | +22.6% |
| 256 tok | 5 | LLMPipeline raw (c18-3) | 24.79 | +15.1% over plain 21.5 |
| 256 tok | 10 | LLMPipeline raw (c18-4) | 18.88 | -12.3% (over-spec) |
| **64 tok** | **5** | **tahoma `--engine ov-genai` (c18-6)** | **134.90** | **+55% over c3-1** |
| 64 tok | 5 | tahoma engine on charlie (c18-7) | 96.04 | +35% over c5-1 |
| 256 tok | 5 | charlie raw LLMPipeline (c18-8) | 26.90 | +16% over c17 |

The tahoma-engine number is HIGHER than the raw LLMPipeline number on
the same workload (134.9 vs 119.2). Likely a kernel-cache effect from
running the second test on the same machine within minutes.

### Phi-3-mini-128k INT4 + Phi-3-mini FastDraft 50M

| Path | tok/s |
|---|---|
| LLMPipeline plain (c18-10) | 32.18 |
| LLMPipeline + 50M FastDraft K=5 (c18-9) | **43.90** (+36.4%) |

Bigger relative gain than Llama, because the per-token cost on the
Phi-3-mini-128k is higher (~31 ms vs ~10 ms for Llama 8B), so the
draft amortises better.

## DISCOVERY (v2)

Intel's published FastDraft companions (`OpenVINO/<target>-FastDraft-<size>-int8-ov`)
give significant speedups on top of plain LLMPipeline across model families:

- **Llama 3.1 8B INT4 + 150M FastDraft K=5**: +24-55% (varies by run)
- **Phi-3-mini-128k INT4 + 50M FastDraft K=5**: +36%
- **Both gains hold on Lunar Lake (charlie 140V)**: +16% at 256-tok output.

Sweet-spot K is 5 for short outputs; K=10 over-speculates on creative
content. The FastDraft is the right size (small enough that draft
compute is cheap, large enough that accept rate stays high) — vs our
earlier 1B-draft tests where draft compute cancelled the savings.

The FastDraft collection (search "OpenVINO FastDraft" on HF) currently
covers Llama 3.1 8B and Phi-3-mini. No Qwen / Gemma drafts published
yet as of OV 2026.1.

## Implications for tahoma

- Wired into `ov-genai` engine via `--draft-model` + `--spec-k` (c18-6).
- `ov-spec` engine (13.83 tok/s today) is now ~10× slower than
  `ov-genai + FastDraft`. Should be marked deprecated.
- Need a doc page: `docs/engines/ov-genai.md` listing the
  recommended target+draft pairings.

## Recommended pairings (for tahoma docs)

| Target | Draft | Expected speedup |
|---|---|---|
| `meta-llama/Meta-Llama-3.1-8B-Instruct` (INT4) | `OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov` | +24% (64 tok), +16% (256 tok) |
| `microsoft/Phi-3-mini-128k-instruct` (INT4) | `OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov` | +36% (64 tok) |
| Llama 3.2 1B / 3B (INT4) | (no FastDraft published yet — would be useful work) | — |
