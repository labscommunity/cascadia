# Campaign 0 — Re-confirm baselines

**Campaign:** c0-baselines

**Falsification:** N/A — measurement, not hypothesis. Records the as-of-now numbers so subsequent experiments are graded against fresh data, not what's in `baselines.md` from main.

**Why:** the cluster has been touched a lot recently (TP work, SSH process kills, package installs). Numbers in `baselines.md` from main may have drifted. Run the same prompts on the same hardware and pin the new numbers before any other experiment lands.

**Predicted outcome:** within ±10% of the baselines.md numbers. If anything is dramatically off, treat as a HW regression and pause.

**Comparison baseline:** carrying forward from `baselines.md`.

## Configurations to run (8B INT4 unless noted)

| ID | Node | Engine | Notes |
|---|---|---|---|
| c0-1 | alpha | ov-optimum / GPU | greedy 32 tokens, prompt "What is the capital of France?" |
| c0-2 | charlie | ov-optimum / GPU | same |
| c0-3 | alpha | ov-spec K=4 / GPU | draft = Llama-3.2-1B-Instruct INT4 |
| c0-4 | alpha+charlie / TB | ov-runtime v5 shards | 2-stage GPU |
| c0-5 | alpha+charlie / TB | ov-dist-spec v5 K=4 | driver=alpha, worker=charlie |
