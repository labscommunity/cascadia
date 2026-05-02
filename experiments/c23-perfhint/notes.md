# c23: PERFORMANCE_HINT plugin property — LATENCY vs THROUGHPUT

## Setup
Llama 3.1 8B INT4 on alpha B390 GPU. LLMPipeline plain (no draft).
64-tok output, factual prompt. 3 runs each, best-of-3.

## Results
| Hint | Best tok/s |
|---|---|
| LATENCY | 134.73 |
| THROUGHPUT | 105.25 (-22%) |

## Finding
LATENCY is the right hint for single-user. THROUGHPUT optimises for batched
workloads at the cost of per-request latency. We were already getting LATENCY
defaults on LLMPipeline; explicit setting is unnecessary but does no harm.

For multi-tenant CB workloads, THROUGHPUT may be worth re-testing — but the
c20 CB sweep already showed the per-request throughput on alpha is the
default of ~138 tok/s aggregate at batch=8, so even there LATENCY-default
seems to be doing the right thing under the LLMPipeline scheduler.
