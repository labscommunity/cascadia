# bench/ — reusable measurement scripts

Bench scripts that talk to the K2.6 pipeline and produce metrics in
autolab campaign format.

## Scripts (filled in as the loop builds them)

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `run_3prompt_eval.sh` | Hit the 2-box matias API with Paris/Pacific/four; capture wall time, tokens, substring match | API URL, max_tokens | stdout: `tok_s=<f>`, `quality=<n>/3`, `prefill_ms=<f>`, `decode_ms=<f>` per prompt |

(Loop adds more as needed: per-stage timing extractor, KV size profiler, expert hit-rate counter, etc.)
