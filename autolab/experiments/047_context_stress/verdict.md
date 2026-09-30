# 047 verdict: context on the eleven-box fleet, measured from 1k to 64k tokens (128k did not finish in six hours)

Run 2026-09-29 17:20 to 2026-09-30 05:06 CDT (11 h 46 min) on the owner's go-ahead with a 12-hour budget. Binary
`cascadia-6430d70e-ctx2` = the deployed engine + the fixes below; the fleet came back on its previous release at
05:04, gates pass, no box died (the entry box, installed rank 0, came through both reloads and twelve hours of load).
Figures and data: `context-prefill.*`, `context-decode.*`, `context-boxes.*`, `context-summary.csv`,
`context_stress.csv` (one row per request), `context_stress.json` (every request with its answer, the per-token
timestamps and every box's telemetry sampled every 5 s), `run.log`. Made by `bench/plot_context_stress.py`.

## The numbers (one stream; mean of the repeats, first token ± sd)

| context (tokens) | repeats | first token | prefill tok/s | decode tok/s (min–max) | needle | CPU cores busy per box | iGPU busy | package W per box | min free GB per box |
|---|---|---|---|---|---|---|---|---|---|
| 1,041 | 3 | 22 ± 0 s | 46.4 | 4.73 (4.18–5.22) | 3/3 | 0.97 | 56 % | 15.7 | 6.98 |
| 2,011 | 3 | 38 ± 0 s | 53.7 | 5.82 (5.29–6.62) | 3/3 | 1.02 | 53 % | 16.8 | 6.93 |
| 3,991 | 3 | 76 ± 0 s | 52.9 | 3.91 (3.60–4.42) | 3/3 | 1.00 | 44 % | 16.2 | 6.99 |
| 7,889 | 3 | 2 min 52 s | 45.7 | 3.32 (2.97–3.73) | 3/3 | 1.02 | 35 % | 16.6 | 6.95 |
| 15,897 | 3 | 7 min 51 s | 33.8 | 2.71 (2.29–3.40) | 3/3 | 1.02 | 26 % | 17.1 | 6.90 |
| 31,614 | 2 | 27 min 15 s | 19.3 | 1.52 (1.35–1.68) | 2/2 | 1.00 | 16 % | 16.2 | 6.76 |
| 64,292 | 2 | 1 h 49 min 44 s | 9.8 | 0.82 (0.60–1.04) | 2/2 | 1.00 | 9 % | 15.5 | 6.56 |
| 131,072 | 1 | **not finished in 6 h** (the request's cap) | – | – | – | | | | |

Repeats agree to the second on the first token; decode varies 20–40 % between repeats because a lone stream's
speed depends on how many of the drafter's guesses the model accepts on that text. The needle (a code sentence
after the first document, asked for at the end) was found in the model's output 20 times of 20: the global layers'
attention over the whole context works at 64k, well beyond the 1,024 positions of their relative-position bias.
The 4-stream pass never ran: the 128k request consumed the last six hours, and the harness stops at the first
request error by design (the cap counts as one).

## What sets these numbers

**Prefill time is quadratic in the context**, and the quadratic part belongs to the CPU:

    T_prefill(N) = N / 53 tok/s  +  1.5e-6 s x N^2        (fit over 1k–64k; predicts 7.5 h at 128k, 30 h at 256k)

The linear part is the pipeline's usual prefill rate: 64-row windows through eleven stages at 45–54 tok/s (the same
as the production 8-row window: the larger window did not raise the rate, see below). The quadratic part is each
window's rows attending over everything before them on the global layer of every rank: 16.4k flops x N² per box,
i.e. 67 TFLOP for the 64k prompt, done in ~90 minutes = **11 GFLOPS effective per box**, the throughput of about
one core. The telemetry agrees: during every request each box had **1.0 CPU core busy** and its iGPU 9–56 % busy
(the iGPU share is what the decode phase adds; during prefill it idles). Sixteen cores per box do nothing.

**Decode cost per token is linear in the context, on the CPU, single-threaded.** 034's probe measured it per box:
45 ms at 4k, 221 ms at 64k, 1.5 s at 512k, **2.9 ms per 1,000 positions** of context, all of it the CPU attention
over the cached keys (the iGPU work does not depend on the context). Eleven boxes in series: 0.4 tok/s at 64k
without guesses; measured 0.6–1.0 with the drafter's guesses (a guess row costs a full attention pass too, so the
gain shrinks with the context: from 1.5x at 1k to 1.2x at 64k). The same arithmetic: 2.1 GFLOP per token per box
at 64k in 193 ms = 11 GFLOPS, one core. `attend_state` runs one row's 64 query heads sequentially; the rows of a
frame run in parallel (011), but a lone stream's frame has one row.

**Memory is not the limit at these sizes.** The 64k context cost 0.5 GB per box (8 KB per position, f32, on the one
global layer per rank), from 7.0 to 6.6 GB free. 034: 512k fits on every box, 1M on none (8.2 GB needed).

**Both fixes made in this release held**: prompts of any length go through (a prompt of more than ~22 windows was
being closed by the runner's "no progress" watchdog before; the feed now emits progress chunks the API does not
send), and the 64-row window's device shapes compiled at load (no 34 s first call).

## What the fleet can serve today, by context

| context | first token | decode, one stream | interactive? |
|---|---|---|---|
| up to 4k | 22–76 s | 3.9–5.8 tok/s | yes, after the wait |
| 8k–16k | 3–8 min | 2.7–3.3 tok/s | decode yes, prefill no |
| 32k | 27 min | 1.5 tok/s | no |
| 64k | 1 h 50 min | 0.8 tok/s | no |
| 128k–256k | 7.5–30 h (predicted) | 0.4–0.2 tok/s (probe) | no |
| 1M | does not fit (8.2 GB per box, 6.6–9.9 free); ~470 h of prefill if it did | 0.03 tok/s (probe: 3 s per box per token) | no |

## What would change it (for the paper's discussion, none of it done)

1. **Parallel attention over heads and keys on the CPU.** One core does the work today; sixteen are there. A head-
   or key-blocked kernel with the existing rayon pool is a contained change in `attn.rs`: up to ~12x on both the
   quadratic prefill term (64k: 90 min -> ~8 min) and the decode term (64k: 0.4 -> ~2.5 tok/s per stream).
2. **Attention on the iGPU** (OpenVINO, the projections are already there): another 5–10x on the same terms, and
   the KV cache would move to the device (f16 there).
3. **Larger prefill windows did not help**: 64-row windows prefill at the same 45–54 tok/s as 8-row ones, so the
   fused-experts prefill path is not the fixed-cost-bound regime the decode path is; its per-row cost (~19 ms at
   eleven stages) is the bus reading each window's distinct experts. Only expert reuse across more rows (bigger
   batches of prompts) or fewer bytes per expert would move it.
4. **f16 KV** halves the memory (1M would fit) but changes no time.

## Method notes for the paper

Prompts: natural text (the fleet's own responses, then Dolly instructions), a needle sentence after the first
document, the question at the end, 96 new tokens, temperature 0, exact prompt token counts from the API. One request
at a time on an otherwise idle fleet; the entry box relays the API. Telemetry: every box's CPU, GPU, package power,
free memory and stage profile sampled every 5 s through the fleet's own relay. Time budget 12 h; sizes 1k–256k
doubling with 3/2/1 repeats; the harness predicts each request from the rates measured and stops when the budget
cannot hold the next one. Test binary on the deployed engine commit plus the two fixes; production settings except
the context budget (1M), 16 slots, the API caps, 64-row windows, and rank 0's ready gate.
