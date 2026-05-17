# autolab/k26-perf — K2.6 pipeline performance moonshots

Autonomous research loop targeting tok/s on the Kimi K2.6 sparse-MoE
pipeline-parallel inference path in tahoma.

**This branch never merges to main.** It's a long-lived research archive
modeled on the closed-but-productive `autolab/intel-gpu-perf` (PR #1)
and `autolab/distributed-perf` (PR #4) branches that produced the
+9% / +19% async-overlap win in PR #5. Verified wins on this branch
crystallize into separate small perf PRs off `main`; this branch is
where the raw campaign data, journal, and dead-end documentation lives.

## Mission

Maximize end-to-end tok/s on the 2-box K2.6 pipeline (matias-02 ↔
matias-03 over Tailscale DERP). Current baseline: ~0.05 tok/s
(~17-20 s/tok). No fixed bar; pursue every credible moonshot and
document the ceiling we hit.

## Frame

| Decision | Choice |
|---|---|
| Hardware scope | matias-02 + 03 (primary); miner (single-stage + microbench); other cascadia (stage as needed) |
| Code authority | Full — tahoma crates, wire format, new engines, rainier exporters, model re-quantization |
| Moonshot bar | 100 moonshots, each **end-to-end on real K2.6 pipeline**, with measured tok/s delta + quality eval |
| Quality gate | Substring + coherent (paris / pacific / four for the 3-prompt eval; coherent English) |
| PR shape | Long-lived branch (this) + spinout perf PRs off main as wins crystallize |
| Loop mechanism | `/loop` autonomous, push every ~10 commits, PushNotification on each verified discovery |

## Layout

| File | Purpose |
|---|---|
| `README.md` | This file. Scope + methodology. |
| `MOONSHOTS.md` | Taxonomy of moonshot candidates (50+ at start; grows as loop discovers more). |
| `PRIOR_ART.md` | Synthesized perf learnings from PRs #1, #4, #5, #7, #9, #10 and rainier `DISCOVERIES.md`. |
| `JOURNAL.md` | Append-only iteration log. One entry per moonshot: hypothesis → result → learnings. |
| `DISCOVERIES.md` | Verified novel findings with prior-art search. Promoted from JOURNAL when reproducible. |
| `INDEX.md` | One-line per campaign with status + result. |
| `LEADERBOARD.md` | Best result per topology (1-box / 2-box / 3-box). |
| `research_plan.yaml` | Active research questions decomposed from the directive. |
| `.autolab/state.json` | Machine-readable progress (iteration count, moonshot count, ratio). |
| `campaigns/NNN_*.yaml` | Campaign definitions (parameter grids). |
| `experiments/NNN_*/` | Per-campaign raw data + notes. |
| `bench/` | Reusable bench scripts that talk to the K2.6 pipeline. |

## Methodology rules (carried over from PR #1/#4)

1. **Hypothesis first.** Every campaign starts with a written hypothesis in JOURNAL.md before code or YAML.
2. **Literature search first.** Each campaign is preceded by a `WebSearch` / `WebFetch` pass that surfaces prior art for the candidate technique. Recorded in the campaign dir as `prior_art.md`.
3. **One variable per campaign.** Defaults hold everything else constant.
4. **≥3 runs per config.** Report best + median.
5. **Always run the 3-prompt quality eval** (Paris / Pacific / four). A perf win that breaks quality is documented as a negative.
6. **Apples-to-apples baselines.** Same prompts, same max_tokens, same warmup state, same hardware (no draft-on-different-box comparisons).
7. **Negative findings count.** A rigorously-documented dead-end is worth as much as a win.
8. **Tok/s is end-to-end actual** (counted via tokenizer, not max_tokens cap), measured at the API. Per-stage compute time is a secondary metric.

## What a "moonshot" is

A moonshot is one of the 100 candidate optimizations from `MOONSHOTS.md`
that is investigated rigorously end-to-end on K2.6. Each moonshot
produces one of three outcomes:

- **win** — measured tok/s improvement, quality eval passes, reproducible across 3 runs. Logged in LEADERBOARD; spinout PR opened off main if productionizable.
- **neutral** — no measurable delta within noise. Documented in JOURNAL.
- **negative** — measured regression OR quality break OR didn't run (infra blocker after good-faith attempt). Documented in JOURNAL.

All three count toward the 100. The point is converting open
hypotheses into closed ones, not just finding wins.
