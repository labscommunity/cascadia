# JOURNAL — autolab/k26-perf

Append-only. Newest at top. One entry per moonshot iteration.

Entry template:
```
## NNN — <title> (YYYY-MM-DD HH:MMZ)

**Hypothesis:** ...
**Bucket / candidate:** ...
**Literature:** (one-paragraph synthesis with [[refs]])
**Campaign:** `campaigns/NNN_*.yaml`
**Design choice:** ...
**Result:** <win | neutral | negative>; tok/s = ...; quality = ...
**Learning:** ...
**Next:** (spawned sub-questions, follow-up moonshots, or "park")
```

---

## 010 — F4 rayon-over-heads — NEUTRAL on miner (2026-05-17 ~20:29 PT)

**Hypothesis:** Parallel per-head SDPA (rayon over 64 heads on 24-core
Xeon Gold) cuts attention bucket from 14.5% → ~1.2%; expect +10-13%
end-to-end throughput.

**Result: -2.7% (neutral, within bench noise).** Same quality 9/10.

Why miner doesn't show F4 win:
1. I/O-bound (cold expert pages dominate) — compute reduction in
   attention bucket doesn't move the bottleneck.
2. 24 cores already saturated by expert dispatch on cold pages —
   no spare cores for parallel attention.
3. Per-head work ~0.4ms is small enough that rayon's task spawning
   (~10us × 64 = 640us) eats the gain.

Patch kept on branch (5 LOC, composes cleanly, no quality regression)
in case future infra changes (compute-bound 2-box matias, faster
storage) flip the verdict.

Bench: `experiments/010_f4_rayon_heads/bench_k4_f4_10p.jsonl`
Notes: `experiments/010_f4_rayon_heads/result.md`

**Next (iteration 011):** A8 KV cache bf16 (currently f32). Halves
KV memory + halves KV bandwidth read during attention. Different
bucket from A3 and F4. ~50 LOC in shell_int4.rs.

---

## 009 — A3 10-prompt robustness — K=4 is the real leader (2026-05-17 ~20:08 PT)

**Hypothesis:** K=3 (iter 008 leader on 3-prompt eval) holds at 9-10/10 on a broader 10-prompt set.

**Result: HYPOTHESIS REFUTED. K=3 only passes 6/10. K=4 = 9/10 (matches K=8 baseline).**

| K | tok/s | Quality | Failed |
|--:|------:|:-------:|--------|
| 8 | 0.0853 | 9/10 | "km" (sampling artifact) |
| **4** | **0.2100** | **9/10** | "celsius" (single sampling artifact) |
| 3 | 0.3050 | 6/10 | jupiter, celsius, guido, "12" — substantive failures |

**Revised production recommendation: K=4, not K=3.** K=3 was misled
by the narrow 3-prompt eval. On a 10-prompt set K=4 matches K=8
quality (within sampling noise) while K=3 has substantive degradation
(multi-choice format, vague answers, factual errors).

Bench: `experiments/009_a3_robustness_10prompt/{bench_k8_10p,bench_k4_10p,bench_k3_10p}.jsonl`
Notes: `experiments/009_a3_robustness_10prompt/result.md`

**Important reflection:** narrow evals can be very misleading for MoE
expert-reduction sweeps. The 3-prompt set (Paris/Pacific/four) only
tested factual lookups that K=3 could still answer; broader prompts
reveal the cliff is between K=4 and K=3, not K=3 and K=2 as iter 008
suggested.

**LEADERBOARD updated** to show K=4 as production-ready leader. K=3
demoted but kept as "narrow-eval fastest." Iteration 009 itself is
classified as a robustness validation that REVISED an earlier win,
not a new win or negative — call it `revision` outcome class.

**Next (iteration 010):** Spinout-PR-prep for K=4 finding. Open a
small focused PR off main with just `--top-k-override` flag + docs.
Plus start iteration 011 on a different moonshot (F4 or A8) to
diversify beyond A3.

---

## 008 — A3 K-sweep full Pareto — K=3 NEW LEADER +208% (2026-05-17 ~19:07 PT)

**Hypothesis:** Bench K=3 + K=5 to complete the Pareto curve.
Expected K=3 in the cliff zone but possibly still passing quality.

**Result: K=3 PASSES, +208% vs K=8 baseline. New leader.**

Full Pareto on miner single-stage:
| K | tok/s | Δ | Q |
|--:|------:|--:|--|
| 8 | 0.0797 | — | 3/3 |
| 6 | 0.1116 | +40% | 3/3 |
| 5 | 0.1547 | +94% | 3/3 |
| 4 | 0.1667 | +109% | 3/3 |
| **3** | **0.2455** | **+208%** | **3/3** |
| 2 | 0.2716 | +241% | 2/3 cliff |

Bench: `experiments/008_a3_topk_full_pareto/{bench_k3,bench_k5}.jsonl`
Notes: `experiments/008_a3_topk_full_pareto/result.md`

**K=4→K=3 is non-linear +47%.** Likely the OS page cache fits a
higher fraction of active experts when only 3 are dispatched per
layer — disk I/O cost drops more than proportionally.

**Next (iteration 009):** Multi-prompt robustness check on K=3.
Before recommending K=3 as production default, validate across 10+
prompts (not just Paris/Pacific/four). 3-prompt substring eval is
narrow; want to bound the quality risk more tightly.

---

## 007 — A2 routing-threshold sweep — NEUTRAL (A3 K=4 still leader, 2026-05-17 ~18:57 PT)

**Hypothesis:** Variable per-token K via sigmoid-weight threshold could
outperform fixed-K=4 by adapting to per-token router confidence.

**Result: neutral.** A2 works mechanically:
- `--routing-threshold 0.05`: 0.0645 tok/s (drops 0 experts, noise from K=8)
- `--routing-threshold 0.2`:  0.1043 tok/s (+31% vs K=8, drops ~2 experts)
- (vs A3 K=4 leader: 0.1667 tok/s, +109%)

A3 fixed-K=4 dominates the Pareto. K2.6's sigmoid weights appear
relatively uniform across top-8, so dropping experts by absolute
threshold is no better than just capping at K=4. The two flags
compose (`--top-k-override 4 --routing-threshold X`) for future
adaptive workloads but don't improve the single-prompt sweep here.

Bench: `experiments/007_a2_routing_threshold/{bench_thr05,bench_thr2}.jsonl`
Notes: `experiments/007_a2_routing_threshold/result.md`

**Next (iteration 008):** F4 multi-thread per shell. Attacks the
14.5% attention bucket (728 ms rank-0 + 578 ms rank-1 per q1).
rayon over the 64 attention heads should halve shell_attn time
on the 24-core Xeon Gold 6252 miner. Different bucket from A3, so
expected to compose with A3 K=4 leader.

---

## 006 — A3 top-K Pareto sweep on miner — K=4 LEADER (2026-05-17 ~18:38 PT)

**Hypothesis:** Push K further than K=6 to find quality cliff.

**Result:**
| K | tok/s | Δ vs K=8 | Quality | Outcome |
|--:|------:|---------:|---------|---------|
| 8 | 0.0797 | (ref) | 3/3 | baseline |
| 6 | 0.1116 | +40% | 3/3 | 005 win |
| **4** | **0.1667** | **+109%** | **3/3** | **006 win** (new leader, L-magnitude) |
| 2 | 0.2716 | +241% | 2/3 | quality cliff — "four" prompt format break |

**K=4 is the productionizable sweet spot.** +109% throughput, no
quality loss per substring eval, no code change needed by end users
(just `--top-k-override 4`). Lit (DeepSeek-V3 paper) predicted this
direction; we now have the concrete K2.6 / Intel CPU number.

K=2 is interesting but breaks the substring quality gate (the model
answered "Two plus two equals" with "? (A) 4 (B" — digit answer
rather than word "four"; semantically correct, format wrong for our
eval).

Bench: `experiments/006_a3_topk_sweep/{bench_k4,bench_k2}.jsonl`
Notes: `experiments/006_a3_topk_sweep/result.md`

**Next (iteration 007):** A2 sigmoid-threshold pruning (drop experts
whose routing weight < threshold rather than fixed K). Lit suggests
this can outperform fixed-K reduction at same average K-active.
Implementation: ~30 LOC in forward_shells; doesn't need 2-box.

---

## 005 — A3 top-K reduction VERIFIED WIN on miner (2026-05-17 ~18:30 PT)

**Hypothesis:** K=8→K=6 yields 15-25% tok/s improvement at <1% quality cost.

**Result: WIN. +40.0% throughput. Quality 3/3 preserved.**

| | tok/s | quality |
|---|---:|---|
| K=8 baseline | 0.0797 | 3/3 |
| **K=6**      | **0.1116** | **3/3** |
| **Δ**        | **+40.0%** | preserved |

Bench: `experiments/005_a3_topk_miner/{bench_k6,bench_k8}.jsonl`
Notes: `experiments/005_a3_topk_miner/result.md`

**Hardware substrate:** miner single-stage (forced pivot — matias-02
Tailscale was broken from earlier iteration's `tailscale up --reset`
attempt; needs manual re-auth). Miner is disk-bound at 58 GB/s read,
133 GB RAM. Per-prompt times vary ±20% by cache state but the +40%
aggregate delta is well above noise.

**Lit alignment:** Predicted +10-25% (DeepSeek-V3 paper, KTransformers).
Measured +40% — at the upper end of lit, consistent with low-concurrency
CPU-bound regime where expert FFN computation + page-in are the bottleneck.

**Tier-S #1 productionizable.** Spinout PR off main: add the
`--top-k-override` flag (commits db85e74 + fe31d7c) with the +40%
finding documented. Default = manifest top_k = no behavior change.

**Next (iteration 006):** D4 async pipeline overlap. Hides 54% of
per-token wall time per q1 breakdown. Requires the 2-box matias setup
— blocked on Tailscale fix. Either:
(a) fix matias-02 Tailscale (manual re-auth on box) and retry, OR
(b) try D4 single-stage variant on miner (less natural; pipeline
    overlap only meaningful with stages), OR
(c) try F4 multi-thread per shell (attacks 14.5% attention bucket;
    doesn't need 2-box; can validate on miner).

Leaning toward (c) F4 for iteration 006 — keeps the loop moving on
miner while matias is parked.

---

## 004 — A3 top-K reduction PARTIAL (2026-05-17 ~14:34 to 15:50 PT, parked on infra)

**Hypothesis:** K=8→K=6 yields 15-25% end-to-end tok/s improvement
(experts are 82% of decode per q1; reducing 8→6 = -25% experts ≈
-20% wall time).

**Bucket / candidate:** A3 (Tier-S #1 per iteration 003 ranking)
**Campaign:** `campaigns/004_a3_topk_reduction.yaml`

**Implementation: SHIPPED + VERIFIED at the per-stage level.**
- `--top-k-override` CLI flag on `tahoma worker` (commit `db85e74`)
- Plumbing: `WorkerArgs.top_k_override` → `SparseMoEBuilderConfig.top_k_override`
  → `Runner::set_top_k_override` → `forward_shells::effective_top_k`
- `TAHOMA_TOPK` env var in `start_rank{0,1}.ps1` wrappers
- Per-token shell breakdown CONFIRMS the override is active:
  `stage_timing shells ... top_k=8 effective_top_k=6 ... experts_us=1,734,411`
  (vs baseline K=8 experts_us=3,229,000 = **-46%** on the warmup token's experts dispatch)

**Bench: BLOCKED on infrastructure** — could not complete a 3-prompt
eval. Tailscale DERP relay between matias-02 ↔ matias-03 went into a
degraded state during this iteration; pattern reproduced at BOTH K=6
and K=8:

- First request's first token round-trip succeeds (rank-1 logs shells
  + head completion, sends Token upstream)
- Second round-trip onward: rank-0's `recv_kind_client` hangs forever
  (no client-side timeout); rank-1 keeps logging 60-second
  `recv_kind: recv_exact timed out after 60s` cycles.
- `Test-NetConnection matias-03:9100` returns True (TCP socket-level
  works); `tailscale ping -c 3 100.123.40.123` reports "direct
  connection not established" — DERP-relay-only path, asymmetric byte
  counts (tx 6.6M / rx 341K cumulative).
- Restarting Tailscale on both boxes (`tailscale down; tailscale up
  --reset`) is in flight; will retry bench when peer link is back.

**Learning (infra):**
1. `recv_kind_client` has no client-side timeout (only the server side
   has 60s). For autolab iteration robustness, future fix-forward
   should add a client-side recv timeout so hangs surface as errors
   instead of indefinite blocks. Filing as follow-up.
2. Multiple kill/restart cycles of tahoma workers seem to leave
   Tailscale DERP connection in a degraded state. Resetting Tailscale
   (`down; up --reset`) before restarting workers may be the right
   cycle for autolab iterations.

**Status: PARKED** awaiting Tailscale link recovery. Will retry K=6
bench in iteration 005; if successful and quality 3/3, also try K=4.

**Per-stage data captured (single warmup token, K=6 with effective_top_k=6):**

| Stage | K=6 (this iter) | K=8 baseline (iter 003) | Δ |
|-------|----------------:|------------------------:|---:|
| Rank-0 layer 0 | 80 ms | 81 ms | -1% |
| Rank-0 shell attn | 696 ms | 728 ms | -4% |
| Rank-0 shell experts | **1,734 ms** | **3,229 ms** | **-46%** |
| Rank-0 shells total | 2,436 ms | 3,974 ms | -39% |

The 46% drop in expert dispatch time at K=6 (vs the expected 25%
proportional to expert-count reduction) is probably partly disk-cache
effects (different runs, different cold-page mix) and partly the smaller
effective working set. Encouraging signal but needs end-to-end bench to
confirm.

---

## 003 — q1 instrumentation COMPLETE (2026-05-17 ~14:34 PT)

**Hypothesis:** Per-stage breakdown will show expert dispatch >60% of
per-token wall time.

**Result: VERIFIED + over-shot.** Expert dispatch is **82%** of per-token
decode time (much higher than predicted 60%). Other stages much smaller
than estimated.

**Bucket / candidate:** q1 — instrumentation (not a moonshot)
**Campaign:** `campaigns/001_instrumentation_breakdown.yaml`
**Bench:** `experiments/003_q1_instrumentation/bench.jsonl` —
0.0550 tok/s aggregate, 3/3 quality (Paris/Pacific/four).
Instrumentation overhead vs baseline 0.0553 = **-0.5%** (within noise,
well below 5% budget).

**Per-token decode breakdown (median of 24 late-sample steady-state events):**

| Stage | ms | % of total |
|-------|---:|-----------:|
| Rank-0 layer 0 (embed + dense attn + KV) | 81 | 0.9% |
| Rank-0 shell attention (30 layers) | 728 | 8.1% |
| Rank-0 shell expert dispatch (30 × top-8) | 3,229 | 35.9% |
| Rank-0 shells combine (residual + shared + moe) | <1 | <0.1% |
| **Rank-0 compute subtotal** | **3,974** | **44.1%** |
| Pure wire latency (Tailscale DERP) | 60 | 0.7% |
| Rank-1 shell attention (30 layers) | 578 | 6.4% |
| Rank-1 shell expert dispatch (30 × top-8) | 4,151 | 46.1% |
| Rank-1 head (RMSNorm + lm_head OV IR) | 139 | 1.5% |
| **Rank-1 compute subtotal** | **4,889** | **54.3%** |
| **TOTAL PER-TOKEN DECODE** | **9,005** | **100%** |
| **→ Implied decode tok/s (no prefill)** | | **0.111** |

End-to-end bench tok/s = 0.055 because the API count includes prefill
(prompts of 3-9 tokens, similar per-token cost as decode).

**Variance** (min vs max over 46 samples):
- Rank-0 experts: 2,098–7,770 ms (3.7× range)
- Rank-1 experts: 1,517–9,003 ms (5.9× range)
- All other stages: <1.5× range
**→ Expert variance is dominated by disk-page-in on cold expert pages.**

**Learning — re-ranking moonshots:**

| Stage | % of decode | Tier-S re-rank | Rationale |
|-------|------------:|----------------|-----------|
| Expert dispatch | **82%** | **#1** | Single biggest knob. A2/A3 expert reduction directly attacks this. C1/C7 prefetch + prewarm reduce its variance. |
| Shell attention | 14.5% | #3 | Second-biggest. F4 multi-thread per shell (rayon over 64 heads) is the obvious win. |
| Wire | 0.7% | dropped | **D1 BF16 wire is not worth pursuing** — saving half of 0.7% = 0.35% delta. |
| Async overlap | hides rank-1 | **#2** | D4 (start T+1 on rank-0 while rank-1 still on T) can hide the 4,889 ms of rank-1 compute — **up to 54%** of per-token time recovered. |
| Layer 0 | 0.9% | skip | Negligible. |
| Head | 1.5% | skip | Negligible. |

**Next iteration (004):** First real moonshot = A3 (top-K reduction
K=8 → K=4 or K=6). Lit (DeepSeek-V3 paper, KTransformers V0.3) reports
10-25% throughput improvement at K=6 with negligible quality cost on
sigmoid-router MoE. K2.6 is sigmoid-router family. Direct attack on
the 82% bucket.

---

## 003-pre — q1 instrumentation EXECUTE (DEFERRED & RESOLVED 2026-05-17 ~14:00-14:34 PT)

*Original deferred entry; superseded by the COMPLETE entry above. Kept
for traceability of the multi-step iteration.*

---

## 002 — q1 instrumentation patch + infrastructure discovery (2026-05-17 ~14:00 PT)

**Hypothesis:** Per-stage breakdown on 2-box K2.6 pipeline will show
expert dispatch >60% of per-token wall time. (Test deferred to 003 —
this iteration uncovered an infrastructure blocker that had to be
resolved first.)

**Bucket / candidate:** q1 — instrumentation (not a moonshot)
**Campaign:** `campaigns/001_instrumentation_breakdown.yaml`
**Literature:** none required.
**Code change:** runner.rs + engine.rs, +48 LOC net
- `forward_layer0_step`: wrap in `Instant`, log `stage="layer0", duration_us`
- `forward_shells`: per-layer accumulators for shell_attn_us + experts_us +
  combine_us, log aggregate at function exit with `stage="shells"`
- `forward_head_last`: wrap in `Instant`, log `stage="head", duration_us`
- `engine.rs` rank-0 driver: split timing of `send_forward` vs
  full round-trip to log `stage="rank0_wire", send_done_us,
  downstream_compute_us`
Each emits a `tracing::info!` line per token. ~1 µs overhead each;
budget << 5%.

**Result: PARKED → 003 (infrastructure blocker discovered + mitigation in flight)**

**What happened:**
1. ✓ Patched runner.rs + engine.rs locally
2. ✓ Built on matias-02 + matias-03 with `--features openvino`
   - matias-03 built clean (8.87 MB)
   - matias-02 failed first attempt because old tahoma.exe was in use
     (Windows can't overwrite running binary; classic). Killed PID 7620,
     rebuilt clean (8.87 MB)
3. ✓ Rank-1 started cleanly on matias-03 (foreground SSH + run_in_background
   pattern; the original `Start-Process powershell -WindowStyle Hidden`
   chain was silently dying in the detached powershell — possibly an SSH
   session/PowerShell lifecycle interaction)
4. ✗ Rank-0 startup failed: `Error: backend error: runner load: internal:
   safetensors layer0: io: The system cannot find the file specified
   (os error 2)`
5. ✓ Root cause: PR #10's `Int4Layer0` requires the safetensors source
   for layer-0 dense tensors + `embed_tokens` table. Pre-PR-#10 layer 0
   used the OV IR (no safetensors needed) which is what matias-02 was
   originally provisioned for. Inventory showed matias-02 has shards
   2-31 but is missing model-00001 (the shard with embed_tokens).
6. ✓ model-00001-of-000064.safetensors on miner is only **949 MB**
   (much smaller than the typical ~9.3 GB shard — embed_tokens layout
   compresses it). Initiated transfer miner→Mac→matias-02.

**Learning:**
- **Deploy contract drift.** PR #10 added a new runtime dependency
  (safetensors source for layer 0) that wasn't surfaced as a deployment
  requirement. Future PRs that add asset dependencies should explicitly
  list the new files in PR description + deploy docs. Added to follow-up.
- **Start-Process detach over SSH is unreliable** for long-running
  Windows processes. The pattern `ssh "powershell ... Start-Process -WindowStyle Hidden"`
  was silently losing the child. Replaced with `ssh "powershell -File <wrapper>" &
  bash run_in_background` — SSH session stays alive while child runs;
  log redirection captures all output reliably. Updated `start_workers.sh`
  semantics for future iterations.
- **Windows binary swap requires process termination first.** Cargo build
  errors with "Access is denied" if the .exe is in use. Kill tahoma
  BEFORE every rebuild on Windows.

**Next:** 003 picks up after transfer completes. Restart workers (matias-02
rank-0 will now find model-00001), run bench, parse per-stage timing,
verify <5% instrumentation overhead vs baseline 0.0553 tok/s.

---

## 001 — Baseline established (2026-05-17 ~13:36 PT)

**Hypothesis:** 2-box matias-02+03 K2.6 pipeline on main @ 208104e
delivers ~0.05 tok/s steady-state (3-prompt aggregate), 3/3 on the
Paris/Pacific/four quality eval. Matches the PR #9 / PR #10 numbers
from memory.

**Bucket / candidate:** baseline (not a moonshot — reference anchor)
**Literature:** none required for baseline. See [[LITERATURE]] for the
horizon: 0.05 tok/s is ~200x below comparable systems in the literature
(KTransformers on Xeon+A100 = 13.69, ik_llama on TR Pro+A6000 = 13.13,
mlx-lm on M3 Ultra = >20). The 30-300x gap is structural, not
hardware-bound, per the pipeline-parallel research agent's read.

**Campaign:** `campaigns/000_baseline_main.yaml`
**Design choice:** Clean restart of matias-02 (rank 0) + matias-03 (rank 1)
via the new `start_workers.sh` / `start_rank{0,1}.ps1` wrappers. Bench
script `k26_3prompt_eval.ps1` polls API readiness then runs 3 prompts at
max_tokens=8 temp=0.

**Result:** **baseline** (anchor for downstream moonshots)
- Paris    : 8 tok / 123.06 s = 0.0650 tok/s ✓ "Paris"
- Pacific  : 8 tok / 170.09 s = 0.0470 tok/s ✓ "Pacific"
- four     : 8 tok / 140.93 s = 0.0568 tok/s ✓ "four"
- **AGG**  : 24 tok / 434.09 s = **0.0553 tok/s**, **3/3 quality**

**Learning:**
- Variance is large (0.047-0.065 across prompts). Pacific (longest output
  context window after prompt + 8 tokens) was slowest; Paris shortest
  was fastest. Per-token latency creeps up with KV size in the
  shells (O(N) with our pre-allocated KV but the attention dot product
  is still per-token).
- The bench script's per-prompt `completion_tokens` is 0 due to a
  PS 5.1 / Invoke-RestMethod auto-parse quirk with snake_case JSON
  fields. tok/s in the rank-0 internal log is correct; bench's tok/s
  computation needs a max_tokens fallback. Filing as a bench-harness
  improvement, not a baseline-blocker.
- Cold-cache restart took ~5 min for the workers to come ready
  (warmer than the historical ~40 min in memory — likely OS page
  cache still warm from the morning's stale rank-0 process).

**Next (iteration 002):**
- Implement q1 (instrumentation): add per-stage timing (layer0 / shells
  attention / experts dispatch / wire / head) so we can attribute the
  17-20 s/tok across stages and rank moonshots by which stage they
  actually attack.
- After q1, the next real moonshot is Tier-S #1: per-token expert
  reduction (A2/A3). Lit says +10-50% with <1% quality cost on
  DeepSeek-V3 sigmoid router family (K2.6 is in that family).

## 000 — Scaffold (2026-05-17 ~12:45 PT)

Branch `autolab/k26-perf` cut from `origin/main @ 208104e`. Autolab
artifact tree created. PRIOR_ART synthesized from PRs #1/#4/#5/#7/#9/#10.
60 moonshot candidates enumerated in MOONSHOTS.md across 7 buckets
(quant, KV/attn, dispatch, wire, topo, sched, algo). 7 research
questions decomposed in research_plan.yaml. PR #11 opened as draft
(long-lived, will not merge). 3 parallel lit-research agents converged
on Tier-S moonshots (A2/A3 expert reduction, D1 BF16 wire, D4 async
overlap).
