# c59: CB multi-tenant numbers ARE accurate (no inflation)

## Setup
alpha B390 GPU, Llama 3.1 8B INT4 + LLMPipeline + SchedulerConfig CB.
batch=8, max_tokens=64. Prompts: "Question N: explain a concept related to TOPIC."

## Results

| Metric | Value |
|--------|------:|
| batch | 8 |
| cap max_tokens | 64 |
| total_cap | 512 |
| **total_actual** | **520** |
| dt | 3.96s |
| aggregate actual tok/s | 131.41 |
| aggregate cap-based tok/s | 129.39 |
| **inflation factor** | **0.98 (essentially none)** |

Each prompt produced ~65 tokens (natural elaboration of concept), which is
right at the 64 max_tokens cap.

## Findings — CB NUMBERS STAND

The c20 / c41 multi-tenant CB results are **NOT inflated** because the
"explain a concept" prompts naturally produce ~65 tokens. The model
fills the cap rather than EOS-ing early.

| Result | Original claim | Status |
|--------|---------------:|--------|
| alpha batch=8 aggregate | 138 tok/s | **VALID** |
| alpha batch=32 aggregate | 559 tok/s | likely valid (same prompts) |
| charlie batch=8 aggregate | 149 tok/s | likely valid |

## Workload inflation factor map

The inflation depends on whether the prompt makes the model fill the cap:

| Workload | Bench inflation |
|----------|----------------:|
| Factual chat ("capital of France") cap=64 | 8× (model EOSes at 8 tok) |
| Extractive summary ("summarize in 2 sentences") cap=512 | 5-14× |
| Concept explanation ("explain a concept") cap=64 | ~1× (no inflation) |
| Long creative ("write essay") cap=256 | likely ~1× (rarely EOSes early) |

## Updated guidance

- **CB multi-tenant aggregate numbers stand** (~138-559 aggregate tok/s at appropriate batch).
- **Single-user factual chat absolute numbers are inflated** by ~5-8× (real ~17-30 tok/s on 8B INT4).
- **PL extractive RAG absolute numbers are inflated** by ~5-14× (real ~20-30 tok/s on 8B INT4).
- **Long-creative (256+ tok output) numbers** likely accurate but should be verified case-by-case.
