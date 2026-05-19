# bench/ — reusable measurement scripts

Bench scripts that talk to the K2.6 pipeline and produce metrics in
autolab campaign format.

## Scripts (filled in as the loop builds them)

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `run_3prompt_eval.sh` | Hit the 2-box matias API with Paris/Pacific/four; capture wall time, tokens, substring match | API URL, max_tokens | stdout: `tok_s=<f>`, `quality=<n>/3`, `prefill_ms=<f>`, `decode_ms=<f>` per prompt |
| `k26_bench_miner_10.sh` | 10-prompt substring eval on a single API (legacy — see warning below) | API URL, outfile, max_tokens | JSONL: one row per prompt + aggregate, includes `tok_per_sec`, `quality_pass` |
| `k26_quality_eval.sh` | Stronger quality eval: full outputs + first-divergence + length proxy + side-by-side table for baseline-vs-feature | `--baseline-api` + `--feature-api` OR `--api` + `--label`, `--max-tokens`, `--out-dir` | `baseline.jsonl`, `feature.jsonl`, `comparison.jsonl`, `summary.txt`; exit 3 on regression |
| `k26_eyeball_eval.sh` | Pretty-print baseline/feature responses side-by-side from a `k26_quality_eval.sh` run dir | `DIR` (from `--out-dir`), `--idx N`, `--width W` | stdout, one prompt per block, divergence marker |

(Loop adds more as needed: per-stage timing extractor, KV size profiler, expert hit-rate counter, etc.)

## When to use which eval

**Substring eval (`k26_bench_miner_10.sh`)** is fine for a quick
go/no-go on a feature you trust. **It is not safe for evaluating a
feature that might silently corrupt generation.** iter 037 measured
+80% TPS at sliding-window W=32 and the 10-prompt substring eval
called it a clean win — but the outputs were `"Paris ?? ?? & Question?
Question? Question? ..."` because `paris` showed up in the first
sentence. Substring eval is blind to coherence past the first match.
Saved as memory `autolab-substring-eval-too-weak`.

**Quality eval (`k26_quality_eval.sh`)** is the replacement for any
A/B where the feature could shift output distribution (sliding-window
attention, sparse softmax router, INT2 experts, top-K reductions,
spec-decode reconciliation, dispatch routing changes). It generates
the FULL response under both configs and reports:

- **first-divergence position** — token-ish (word-split) + byte index
  of the first position where baseline and feature outputs disagree;
  -1 means identical. Early divergence under a quality-preserving
  feature should be a red flag.
- **output-length proxy** — bytes per response. If the feature
  response is < `eos-garbage-frac` * baseline bytes, it flags
  `SHORT` — often an EOS-on-garbage early-stop (the model hit a
  degenerate state and emitted EOS to escape).
- **full content in JSONL** — both runs dump verbatim so the
  eyeball helper or manual `jq -r '.content' baseline.jsonl` can
  always recover what the model actually said.
- **side-by-side table** — `summary.txt` is the human-readable
  artifact for committing into an experiments/ dir.

**Eyeball helper (`k26_eyeball_eval.sh`)** consumes a quality-eval
run dir and prints both responses next to each other, wrapped to a
configurable column width. Use this when the comparison flags
something interesting and you want to see what the model said
without hand-jq'ing the JSONL.

## Not implemented yet

**Perplexity vs baseline-logits** is the strongest cheap signal we
could add, but it requires the API to expose per-token logprobs.
`POST /v1/chat/completions` in `crates/tahoma-api/src/lib.rs`
currently returns content only. When logprobs land, `k26_quality_eval`
already has stub fields (`logprob_baseline` / `logprob_feature`) in
the per-prompt JSONL — plumb them through and add a `cross_entropy`
column to `comparison.jsonl`.
