# 034: what context length can this deployment serve, and at what speed (prepared 2026-09-28; NOT run)

**Owner's question.** Inkling supports a 1M-token context. Is that even possible on the eleven-box fleet, and how does
speed fall with context length? One-shot test, at most 30 minutes, only on the owner's explicit go-ahead
(`bench/run_context_test.sh`; it ends by restoring what the fleet ran before).

**What the fleet serves today: 1,024 tokens of context.** `rank.env` carries `MAX_SEQ=1024` from the install
(`fleet.env`), and the API caps a prompt at 32 KiB. A longer prompt is truncated to its first 1,024 tokens with a
warning; a generation that reaches position 1,024 stops. Every benchmark so far ran inside that.

**What the arithmetic says.**
* Memory: the sliding layers (5 of 6 per rank) keep a ring of 512 positions; only the global layer of each rank
  (8 KV heads x 128 x f32 x K and V = 8 KB per position) grows with context. Per stream: 268 MB per rank at 32k,
  1.07 GB at 128k, 4.3 GB at 512k, **8.6 GB at 1M** (94 GB fleet-wide). Serving boxes have 6-10 GB free (the iGPU owns
  50 GB of the 61). So one stream at 1M does not fit in f32 on this layout; 256-512k might; f16 KV would halve it.
* Prefill: a prompt travels as 8-row windows at ~50 tokens/s through the pipeline (024), plus attention over the
  growing context on the CPU. 32k tokens ~ 12 minutes; 128k ~ 1 hour; **1M ~ 6-8 hours**. Inside a 30-minute test,
  real prompts reach ~30k tokens. The rest of the question is answered by a probe, not a prompt.
* Decode at context N: each rank reads its global layer's N x 8 KB of cache (8.6 GB at 1M: ~80 ms at the bus) and
  the CPU attention over N keys (64 query heads x N x 128 x 4 flops = 33 GFLOP at 1M: seconds per rank). Expect
  decode to slow from ~3.5 tok/s to a few tokens per MINUTE at 1M, if it fits.

**Two measurements in one release** (binary `cascadia-83cefc97-ctx` = the deployed engine 639f0c02 + the probe + API
caps from the environment; overrides = the live configuration + max_seq 1M, 16 slots, the probe list):
1. **The probe, every rank, at load** (`CASCADIA_INKLING_CONTEXT_BENCH`): fill slot 0's caches to N positions with
   synthetic rows (every byte a real sequence writes, so the pages are resident), decode one row at N a few times,
   report memory and time, free the pages. Sizes 4k, 16k, 64k, 100k, 128k, 256k, 512k, 1M. A size that does not fit in
   the box's free memory plus a 3 GiB margin is reported `fits=0` and not tried: the probe cannot take a worker
   down (tests: `tests/inkling_context_bench.rs`, the runner decodes the same tokens afterwards).
   Read with `probe_read.py CX`: per rank and size, `fits`, `need_mb`, `avail_mb`, `decode_ms`, `attn_ms`.
2. **Real prompts, one stream** (`bench/context_scan.py`): natural text (the fleet's own responses, then Dolly) of
   1k, 8k, 32k, 64k and 100k tokens with a needle sentence near the start and a question at the end that asks for
   it; 32 new tokens each. Prompts travel as 64-row windows during the test (8 in production, chosen in 024 for a
   burst of 15 streams): a lone long prompt is limited by the fixed cost per window, so 2-3x the prefill rate. Measured: exact prompt tokens (`usage`), first token = prefill, decode tok/s at that context,
   and whether the answer contains the needle (attention over the whole context works or not). A 13-minute budget:
   a size whose predicted prefill would pass it is skipped and recorded.

**Time.** Publish + settle ~10 min (the probe adds 2-3 min to every rank's load), gates 1 min, scan <= 13 min:
**about 25 minutes**; then the revert release (~8 min) restores `cascadia-639f0c02-streams` + the 040/041 overrides.

**Predictions.** Probe: 4k-128k fit on every rank; 256k on all; 512k on the ranks with >= 7.5 GB free (the box
that plays rank 0 and a few others), not on the rest; **1M on none** (needs 8.6 + 3 GB). decode_ms roughly
35-45 ms up to 16k, +1 ms per 1k positions beyond (CPU attention), so ~300 ms at 256k. Scan (64-row windows): prefill 120-160
tok/s at 1k-8k, falling with the square of the context on the CPU attention (each window's rows attend over
everything before them on the global layer: ~17 TFLOP for a 32k prompt, ~67 at 64k, ~160 at 100k, on CPUs that do
maybe 100 GFLOPS): first token ~10 s / ~1 min / 4-6 min / 12-20 min / (100k: 30-45 min, skipped by the budget);
so 1k, 8k and 32k for real, 64k if the measured rate allows, 100k by the probe only; decode 3.5 -> ~3.0 tok/s at
32k and ~1.5-2 at 100k (probe); needle found at every size (if not, the global layers' attention beyond 1,024 of relative-bias extent is
wrong, which would be a finding on its own).

**Kill / safety.** No traffic before `steady 3/3`; a gate failure on the test release = immediate revert; the scan
stops at the first request error; every request is capped; the script always ends with the revert.

**If a box dies mid-test** (the entry box has lost power four times): every number is on disk the moment it is
measured. The probe lines are saved every 20 s while the fleet settles (they reach the Mac through the entry box, so
they would be lost with it); the scan writes its results file after every size and marks the size in flight; every
exit, planned or not, prints the numbers gathered so far. The revert release is published regardless and the boxes
apply it by themselves once the entry box is back.

**What it does NOT test.** Many streams at long context (memory adds up per stream), prompts beyond ~32k for real,
KV in f16 (a possible follow-up if 512k is wanted), quality beyond the needle.
