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

## 003 — q1 instrumentation EXECUTE (LAUNCHED 2026-05-17 ~14:00 PT, fix-forward from 002)

**Picked up from iteration 002's infrastructure blocker.** Instrumentation
patch in `crates/tahoma-engine-sparse-moe/src/{runner,engine}.rs` is
committed and built (8.87 MB binary on both matias boxes, +40 KB vs
baseline). Missing `model-00001-of-000064.safetensors` (949 MB on
miner, contains `language_model.model.embed_tokens.weight` per the
PR-#10 Int4Layer0 contract) is transferring miner→Mac→matias-02 in
background. Once landed: restart both ranks, run 3-prompt bench,
parse per-stage timings.

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
