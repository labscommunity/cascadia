# e1 — distributed baseline (alpha+charlie/TB4, ov-dist-spec K=3 + FastDraft)

**Hypothesis:** Re-establish the post-Phase-14 distributed baseline on the same creative-writing workload as e0 so we have a single number to beat.

**Setup:**
- alpha (driver, rank 0, B390 dGPU) ↔ charlie (worker, rank 1, LL 140V iGPU) via TB4
- Engine: `ov-dist-spec`, target shards `C:\cascadia\shards_2stage_v5_beam` (v5 16/16 split), draft `fastdraft-150m-int8-ov`
- Spec K = 3 (the prior Phase 14 validation config)
- Same 256-tok creative prompt as e0
- 5 trials. Charlie worker restarted between trials (engine bug — only accepts one connection per process; tracked as a follow-up).
- Engine logs `task active` and `ov-dist-spec done`. Engine doesn't yet print elapsed_s for this engine — measured by parsing the two timestamp lines.

## Result

| Trial | Generated tokens | Steps | Accept | Engine elapsed (s) | tok/s |
|------:|-----------------:|------:|-------:|-------------------:|------:|
| 1     | 256              | 220   | 0.054  | 24.666             | 10.38 |
| 2     | 256              | 220   | 0.054  | 26.114             | 9.80  |
| 3     | 256              | 220   | 0.054  | 25.162             | 10.17 |
| 4     | 256              | 220   | 0.054  | 26.008             | 9.84  |
| 5     | 256              | 220   | 0.054  | 25.910             | 9.88  |

**Median: 9.88 tok/s.** Spread: 9.80 – 10.38 (5.5%).

## Conclusion

- Distributed `ov-dist-spec K=3 + FastDraft` on creative writing = **9.88 tok/s** — that's **43% of single-node monolithic** (e0: 23.01 tok/s). Distributed is **2.3× SLOWER** for single-user creative-writing workload.
- Root cause: **FastDraft 150M acceptance dropped to 5.4%** on creative content. The 220 verify steps for 256 tokens means each step produced ~1 token (the verify; drafts almost always rejected). With K=3, we're paying 3× the worker forward-pass cost per step but getting effectively 1 token per round-trip.
- This contradicts the post-Phase-14 number (29.74 tok/s) which used a *short factual* prompt — there FastDraft hit 0.83 acceptance and spec decode actually amortized cost.
- The bar to beat is 23.01 × 1.20 = **27.6 tok/s** distributed, single-user, on this creative workload. **We are 65% below the bar.**

## What this implies for the campaign roadmap

1. **FastDraft K is workload-sensitive in the extreme.** Need K-sweep (K=1, 2, 4, 6) and possibly adaptive-K (`assistant_confidence_threshold`).
2. **Spec decode at low acceptance is *worse* than no spec decode** for distributed PP — we're paying the worker 3 forward passes for 1 useful token.
3. **A non-spec distributed baseline** (`ov-runtime`, no draft) tells us the floor for sequential PP on this workload — establishes whether spec decode is even net-positive here.
4. **Sequential PP is fundamentally upper-bounded** for single-user: stage_0 → transfer → stage_1 always serializes. The wins must come from spec amortization OR async overlap OR heterogeneous compute (NPU as second device).
5. **The 38.49 tok/s headline from the prior python autolab** was on a different workload. Need to reverify what K + workload + draft combination produces that for the Rust port.

Filing two follow-ups:
- e2: K-sweep on `ov-dist-spec` (creative workload, 5 trials each)
- e3: `ov-runtime` distributed baseline (no spec) on the same creative workload — measures the floor of pure PP

Engine bug tracked: dist-spec worker should accept additional connections after upstream closes, instead of busy-looping with "upstream closed, exiting" warnings. Workaround in bench scripts: kill+restart between trials.
