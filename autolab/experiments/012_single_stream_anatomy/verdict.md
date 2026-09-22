# 012: where a lone stream's time goes (measurement only, fleet as left by 011)

Three phases of one stream, 96 tokens each, prompts the drafter table had not seen: **2.8-3.05 tok/s**.

| quantity | value |
|---|---|
| work per frame, all ranks summed | **408 ms** (ranks 1-7: 35 ms, 8-9: 32, rank 10: 32.6 + 11.5 head, rank 0: 44.3) |
| of which iGPU attention projections | 9-10.5 ms per rank (int8, at the memory bus's speed) |
| of which fused experts | 19-21 ms per rank for 6 layers x 8 experts = 1.53 GB: 73 GB/s, the bus |
| rank 0's two dense layers | ~18 ms for 2 layers (rank 0's stage is the slowest, 44 ms) |
| round trip seen by rank 0 | 575-607 ms average over all frames |
| guesses sent / right / wrong | 415-423 / 27-31 / 40-41 per 96 tokens; 25-28 tokens had no guess at all |
| acceptance a (right guesses per token) | **0.28-0.32** |
| rank busy time | 47-69 % |

The model `time per token = a*T + (1-a)*L` fits: a = 0.30, T = 44 ms, L = 408 ms of work + ~25 ms of hops +
~45 ms that rank 0 adds by computing up to ten more guess frames (44 ms each, blocking) before it reads the
reply that matters: 0.30 x 44 + 0.70 x 478 = 348 ms, measured 330-350.

What that says:

1. Every stage already runs at the speed of its memory bus. `L` is the 26 GB a token has to read, divided by
   one bus: no kernel work shortens it. Only reading on several buses at once does (expert parallelism), or
   reading less (int4 attention, fewer experts).
2. With L fixed near 440 ms, **10 tok/s needs a >= 0.85; no drafter reaches that on open text**. With a = 0.3
   the fleet is within 10 % of the best this topology can do.
3. So the single-stream target needs BOTH terms to move: a much better drafter (a: 0.3 -> 0.6-0.7) and a lone
   row that is served by many buses (L: 440 -> ~200-250 ms).
4. Small exact wins inside the present design: rank 0 should read replies between guess frames (~45 ms per
   miss), and should not send guesses nobody expects to be right.
