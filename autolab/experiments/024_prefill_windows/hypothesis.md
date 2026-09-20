# 024: first token sooner: prompts as 8-row windows pipelined across the stages (config only)

**Why.** "Interactive" is also the wait for the first token: 4.8-6 s for a lone 40-token prompt, 31 s (median) when 15
arrive together. A prompt is ONE frame that crosses eleven stages in series (~450 ms each: nearly every expert of
every layer is read), ten stages idle meanwhile. Windows of W rows sent back to back keep several stages busy on the
same prompt: `first token ~ (10 + L/W) x T(W)`; with T(8) ~ 125 ms and L = 40: ~1.9 s. It costs bytes (windows do
not share expert reads) and it serialises prompts (one feeding stream at a time), so burst arrivals must be measured too.

**Method.** `CASCADIA_STREAMS_PREFILL_WINDOW=8`, binary unchanged (a63debe7). Gates (windowed prefill must give the
same text). Phases: one stream alone; 15 streams arriving 1 s apart (`stag15`); 15 at once (`mix15`).

**Prediction.** Alone: 5 s -> ~2 s. Staggered: median under 4 s. Burst: median 31 s -> ~10 s. Steady decode unchanged.
Also in this release, read-only: rank 0 pages the kernel log of the boot in which it stopped dead at 13:39 CDT.
