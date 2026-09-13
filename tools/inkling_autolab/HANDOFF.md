# Restart handoff — Inkling / Panther Lake Autolab

Updated 2026-09-13. Completed campaign execution; no task benchmark/controller
is left running. All results and the full-model deployment blocker are saved. The user is offline and authorized autonomous testing on
**tate-07, 100.82.253.76**, plus commit/push as t8, without coauthor trailers.
Latest target: **25 full-model decode tokens/s for large Inkling on this one PTL
box**. It has NOT been reached. Do not equate layer/component rates with it.

## Current outcome and blocker

The autonomous research loop is operational in this Codex session. Autolab is
the sequential experiment executor, SQLite history and resumption mechanism;
the session supplies research decisions. No Claude hook or separate API key is
needed. Do not claim the research agent runs after the session ends.

Campaigns 001–010 completed; 011 is a read-only Windows residency diagnostic.
Campaign 012 completed: all selected settings confirmed using the same final binary.
Use `JOURNAL.md`, `results/012_final_profile.json` and `.autolab/state.json`
for its completion and exact final numbers. If interrupted, resume 012; SQLite
skips completed experiments and reruns a trial interrupted before it was stored.

Final campaign 012: adaptive rows 1/1 47.106 ms/layer-token, direct rows 1/1
18.230 ms, direct + bf16 rows 2/int4 rows 4 17.456 ms. Six rotating process
groups, all output hashes exact. Combined 2.699x resident-layer gain; tiles alone
1.044x and win all six groups (1.034–1.075x). Earlier frozen-binary comparison
009 measured 2.753x / 1.070x; use the conservative final same-binary numbers.
Smaller pools/affinity subsets and parallel projections lost. Retained two opt-in
AVX2 row kernels; original defaults remain 1/1. Removed parallel-projection
production code; saved the rejected patch/evidence. Kernel commit: 18c3edf9; campaigns/full-decode harness: 83d07a27. Both pushed
as t8 without coauthors.

OpenVINO eight-expert probe: GPU async 3.268 ms vs CPU async 9.541 ms.
Independent f64-dot oracle passed for all eight distinct experts at two inputs.
This is exploratory, allows summation differences, is not a production GPU
backend and omits attention, routing, full expert-population paging and churn.

Full model is blocked on deployment assets/capacity. The unchanged 975B export
is ~512 GB; tate-07 has 64 GB RAM, one fully partitioned 1.024 TB SSD with only
~3.4 GB free, no mapped drives, and a 1 Gb/s physical Ethernet adapter. A bounded
scan of 20,095 directories found no Inkling manifest. The controller Mac also
has no external volume and only ~10 GiB free. Do not delete other projects,
prune the model, offload computation to another box, or invent throughput.
Provisioning storage/checkpoint is a prerequisite, not proof 25 tok/s becomes
attainable. The current batch-one engine reads tens of GB per token; reaching
25 would need a different, validated strategy such as speculation/quantization.

A new `inkling_decode_bench` example is built on PTL as `bin/full-decode.exe`.
It loads every layer/expert/edge table and measures autoregressive decode;
separates prefill, stops at EOS, checks finite/repeated logits and supplied
reference greedy IDs. Non-975B models require `--allow-fixture` and emit a
separate fixture metric. See README full-model instructions and
`full-model-campaign.template.yaml`. The controller's target gate requires a
verified full model, baseline hash, expected greedy IDs, >=32 decode steps,
>=3 repetitions, and the slowest case/repetition >=25 tok/s. No full checkpoint
was available to run that benchmark here.

## Locations and ownership

- Main checkout `/Users/tatef/Workspaces/tahoma`, branch
  `perf/prefill-layer-streaming`; other agents' `glm5_run.rs` and `ngram_sim.rs`
  changes are not ours. Leave alone.
- Original Inkling `/Users/tatef/Workspaces/tahoma-inkling`, `feat/inkling`,
  baseline `9aaebff02a2f68a48913ff67061d90c13121ae35`, PR #154. Docs report
  1.5 full model tok/s on a resident 1.5 TB Mac Pro. Do not rewrite this branch.
- Our isolated worktree **`/private/tmp/tahoma-inkling-panther-autolab`**.
- Our branch **`origin/perf/inkling-panther-autolab`** at
  `https://github.com/labscommunity/cascadia.git`.
- Git identity `Tate Berenbaum <t8@users.noreply.github.com>`; no coauthors.
- Autolab `/Users/tatef/Workspaces/autolab`, `3993e2c4`; its pre-existing dirty
  `src/autolab/runners/ssh.py` is unrelated and untouched.
- Venv `/private/tmp/inkling-autolab-venv`, editable Autolab + pytest. Use its
  Python (system Python lacks yaml).
- Root pointer `tmp/INKLING_AUTOLAB_HANDOFF.md` in the main checkout.

## Host and process rules

SSH: `ssh -o BatchMode=yes -o ConnectTimeout=10 cascadia-tate-07-ts`.
Alias uses `devcloud@100.82.253.76`, `~/.ssh/id_ed25519`.
Actual hostname `pdx88-pa0794`, Windows 11 Pro, Core Ultra X7 358H,
16 cores/threads; AVX2/FMA yes, AVX-512 no. PowerShell default shell, no WSL.

Our root `C:\Users\devcloud\inkling-autolab` contains `repo`, `target`, `bin`,
`synthetic-experts` (8 bins, 255 MB) and launchers. Other services include OVMS
and cascadia-swe-node; do not stop them. Only stop a task process whose executable
path is under our root, or our precisely named Python probe. Do not broad-kill
Python, Cargo, or inference servers. One benchmark/build at a time.

`build.bat` calls vcvars64 at `C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat`,
uses task-only CARGO_TARGET_DIR and explicit stable MSVC toolchain. **Audited
compiler: rustc 1.98.1 (48a229cea), LLVM 22.1.8.** Earlier 1.95 metadata was not
an explicit MSVC query and was corrected; campaign 012 uses one binary for all
arms to remove compiler/binary ambiguity. Build/test scripts should continue to
record `rustc +stable-x86_64-pc-windows-msvc -vV` with new builds.

Frozen binaries: baseline.exe, bf16-rows.exe, int4-rows.exe, projections.exe,
final.exe, full-decode.exe. Never overwrite a frozen comparison baseline.
Hashes are in raw results and `results/final_validation.json`.
`run-bench.ps1` sets child High priority, affinity 0xffff, explicit Rayon pool,
read/schedule/row knobs, fixed output hash, and cleans up its own child on error.
Full real-model runs must retest adaptive vs direct reads under actual paging.

## Validation and resumption

222 MSVC tests passed: full library, DSV4/GLM shared math/expert tests and all
74 Inkling tests. Full decode fixture: 8/8 HF greedy IDs, three repetitions,
hash `1f7cd0eb14a22662`, full_model=0. Clippy completed with existing library
warnings and three benchmark style suggestions. Controller tests: 8 passed.
Autolab campaign/loop/SSH tests: 21 passed during setup. Format/diff checks passed.

Before running, inspect git status and our remote processes. To resume a grid:

```sh
cd /private/tmp/tahoma-inkling-panther-autolab
/private/tmp/inkling-autolab-venv/bin/python -u tools/inkling_autolab/run_campaign.py \
  tools/inkling_autolab/campaigns/012_final_profile.yaml
```

The runner takes flock to prevent simultaneous campaigns, saves raw JSON even
on interruption, and refuses promotion on failure, missing/non-finite metrics,
hash mismatch, numerical oracle failure or incomplete sweep. Campaign regexes
need `(?m)` for multiline stdout; explicit SSH user devcloud; no remote working_dir
(the Autolab version resolves it on the controller). CLI 0 alone is insufficient;
use the wrapper's verified result. New hypotheses get new campaign names.

If /tmp is cleared, recreate an isolated worktree from our pushed branch and
the venv. Rehydrate ResultsDB by `store_result(record)` for each record in the
committed campaign JSON arrays, excluding metadata/summary objects and the older
003 partial snapshot (use complete 003 instead). The remote task files persist.
Do not apply saved candidate patches: the accepted kernels are already in the
branch, and the rejected projection patch is intentionally not applied.

The rg shim hung repeatedly; it was tried first. Use git grep/git ls-files or
bounded Python search if it still hangs. Latest permission profile has unrestricted
filesystem/network and approval policy never; do not pass sandbox_permissions.

If the full checkpoint remains unavailable, report that specific prerequisite;
do not rerun resident sweeps indefinitely or claim the 25 tok/s objective done.
Continue the full-model campaign autonomously when storage/checkpoint becomes
available. Quantization/speculation/GPU integration must be checked against real
weights and correctness before claiming full-model gains.
