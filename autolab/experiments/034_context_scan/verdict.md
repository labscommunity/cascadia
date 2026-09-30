# 034 verdict: 1M does not fit (8.2 GB against 9.4-9.9 GB free), 512k does everywhere; a token at N costs ~45 ms + 2.9 ms per 1k of context per box

Run 2026-09-29 16:25-16:52 CDT on the owner's go-ahead; 26 minutes; the fleet came back on its previous release
(gates pass) and no box died. Binary `cascadia-83cefc97-ctx` (the deployed engine + the probe + API caps).

**The fleet had been serving a 1,024-token context** (`MAX_SEQ=1024` in `rank.env` from the install; the API's 32 KiB
prompt cap): every benchmark before this ran inside that.

**The probe (every rank, at load; `probe_cx.json`).** decode ms per token per box, all eleven within a few percent:

| context | 4k | 16k | 64k | 100k | 128k | 256k | 512k | 1M |
|---|---|---|---|---|---|---|---|---|
| ms per token per box | 45-67 | 70-79 | 211-223 | 312-334 | 386-414 | 751-782 | 1484-1512 | 2990 (the one box that fits it) |
| of which CPU attention | 28-35 | 49-52 | 195-197 | 293-302 | 375-381 | 727-740 | 1464-1472 | - |
| one stream, eleven boxes in series | ~0.6 s | ~0.8 s | ~2.4 s | ~3.5 s | ~4.4 s | ~8.4 s | ~16.5 s | ~33 s |

Linear in the context: **~2.9 ms per 1,000 positions per box**, all of it the CPU attention over the cached keys (the
iGPU work is the same at every length). Memory: a 1M context needs 8.2 GB per box; ten boxes have 9.4-9.9 GB free
(inside the 3 GiB margin: skipped), the box that plays rank 0 has 22 GB and ran it. 512k (4.1 GB) fits on every box.

**Real prompts** (one stream, 64-row prefill windows): 1,040 tokens: first token 34 s (30 tok/s), decode 3.5 tok/s;
the needle was not answered inside 32 tokens (the model reasons first). 8k: closed after 25 s by the runner's
watchdog ("no progress for 3 consecutive steps"): a prompt goes down one window per engine step and steps that send
a window return nothing, so any prompt of more than ~22 windows (176 tokens at the production 8-row window!) was
being closed. The 34 s at 1k was the first 64-row call compiling its kernels on every layer of every rank.
Both fixed for 047: the feed emits progress chunks the API does not send; the window's shapes compile at load.
