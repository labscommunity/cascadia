# /loop launch prompt for autolab/k26-perf

Used to invoke `/loop` autonomous mode (self-paced, no fixed interval).
Each wake-up runs ONE moonshot iteration following the protocol below.

## The launch prompt

```
Run one iteration of the autolab/k26-perf research loop on this branch.

Frame (locked, do not re-litigate):
- 100 moonshots, each end-to-end on the K2.6 pipeline (matias-02 + 03 over Tailscale).
- Quality gate: 3-prompt Paris/Pacific/four substring + coherent.
- Code authority: full (tahoma crates + rainier exporters + new engines).
- Long-lived branch; spinout PRs off main for verified wins.
- Push every ~10 commits. PushNotification on each verified discovery.

ONE-ITERATION PROTOCOL — do all of this in this single wake-up, then exit:

0. ORIENT
   - Read autolab/.autolab/state.json → current iteration N, moonshots_attempted, leader tok/s
   - Read autolab/JOURNAL.md tail (last 3 entries) for momentum
   - Read autolab/LEADERBOARD.md for current best
   - Read autolab/MOONSHOTS.md execution order → which candidate is next
   - Read autolab/INDEX.md for what's already been tried
   - git log --oneline -5 for commit cadence

1. HYPOTHESIZE
   - Pick the next Tier-S candidate not yet tried (then Tier-A, then refined moonshots from prior negatives)
   - If consecutive_no_improvement >= 3: invoke escape protocol — challenge a foundational assumption, design a moonshot that tests its opposite
   - Write a one-paragraph hypothesis in autolab/JOURNAL.md as a new entry header

2. LITERATURE (mandatory — user explicitly required "deep research before each campaign")
   - Spawn 1-2 WebSearch calls (or 1 general-purpose Agent) on the specific candidate
   - Save findings as autolab/experiments/NNN_<name>/prior_art.md (one file per campaign)
   - If lit reveals the candidate is dead (already failed elsewhere with documented reason), mark moonshot as negative, document, move to next candidate THIS ITERATION (don't waste a cycle)

3. DESIGN
   - Write autolab/campaigns/NNN_<name>.yaml with full schema (see TEMPLATE.md)
   - For code changes: stage the diff locally on this branch

4. EXECUTE
   - If code change: cargo build --release on matias-02 + matias-03 (NOT on Mac — no OV headers there). Use:
     git archive autolab/k26-perf | gzip > /tmp/tahoma.tar.gz
     scp /tmp/tahoma.tar.gz cascadia-matias-{02,03}:tahoma-src.tar.gz
     ssh cascadia-matias-{02,03} 'powershell -NoProfile -Command "cd $env:USERPROFILE\tahoma; tar -xzf ..\tahoma-src.tar.gz; cargo build --release -p tahoma --features openvino"'
   - autolab/bench/kill_workers.sh
   - autolab/bench/start_workers.sh
   - Wait for cold start (poll up to 90 min via bench script's built-in poll)
   - Run autolab/bench/k26_3prompt_eval.ps1 (poll-and-eval)
   - Capture rank-0 + rank-1 logs for per-stage attribution if applicable

5. ANALYZE
   - Compare tok/s to LEADERBOARD current leader
   - Verify quality 3/3 (else mark negative regardless of tok/s)
   - Classify: win (>=5% better + quality pass + reproducible) / neutral (within ±5%) / negative
   - For "win": run the bench 2 more times to confirm reproducibility (3 runs total)

6. DOCUMENT
   - Update autolab/JOURNAL.md with the full entry (hypothesis → result → learning → next)
   - Update autolab/INDEX.md with one row
   - If win: update autolab/LEADERBOARD.md + state.json current_leader, open spinout PR off main with the productionizable subset of the change
   - If win AND novel (prior-art search shows no published equivalent): add entry to autolab/DISCOVERIES.md with proper attribution
   - Update autolab/.autolab/state.json: increment iteration + moonshots_attempted; update class/magnitude counts; reset or increment consecutive_no_improvement

7. COMMIT + (maybe) PUSH
   - git add autolab/ (and any code changes)
   - git commit -m "research(NNN_<name>): <one-line result>" (Conventional Commits — see [[tahoma-git-conventions]] — NO Co-Authored-By)
   - If >= 10 commits since last push OR if win: git push origin autolab/k26-perf
   - If win: PushNotification("autolab #NNN <name>: <delta>% to <tok/s>")

8. STOP CONDITION
   - If state.json moonshots_attempted >= 100: PushNotification("autolab complete: 100 moonshots done") and DO NOT reschedule
   - Else: ScheduleWakeup(delaySeconds=1800, prompt=<<autonomous-loop-dynamic>>, reason="next moonshot")
     (1800s is reasonable for a 15-30 min iteration cadence; bumps to longer if cold start needed)

ANTI-LIST — don't re-test these (see autolab/PRIOR_ART.md + LITERATURE.md):
- MXFP4 on Intel CPU (no Blackwell)
- Tensor parallelism over Lunar-Lake fabric
- Tree-spec on current draft arch
- GPipe micro-batching for single-user decode

NOTES:
- The /loop runtime will pass back this same prompt on each wake; treat each wake as one iteration.
- Conserve context: don't re-read full PRIOR_ART/LITERATURE/MOONSHOTS each iteration; cite by path and read only the relevant sections.
- If matias workers are unreachable or in a bad state: kill via kill_workers.sh, restart via start_workers.sh, retry once; if still bad after retry, log infrastructure-blocker and ScheduleWakeup with longer delay.
- If a build fails: don't disable hooks, don't skip CI checks. Fix forward.
```

## How to invoke

```
/loop <paste the prompt above>
```

The `/loop` runtime (without an interval argument) puts the model in dynamic
mode, where each wake-up reads the prompt and ScheduleWakeup'd at the end of
its turn. The model self-paces based on what's needed.

## Stop conditions

- 100 moonshots completed
- User intervenes (Ctrl-C the loop)
- 3 consecutive iterations fail at infrastructure layer (matias unreachable for >1 hour total)

## Notes for the user

The loop will push to `origin/autolab/k26-perf` every ~10 commits. Watch
the PR at https://github.com/labscommunity/tahoma/pull/11 for live
progress. Each verified discovery fires a PushNotification.

To cancel: send Ctrl-C in the active session, OR ask Claude to stop
the loop in the next interactive prompt. The loop won't reschedule
once cancelled.
