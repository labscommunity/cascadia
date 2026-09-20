# 005 verdict: KEEP. 29.6 tok/s steady at 176 streams; unseen-prompt single stream 1.9 tok/s; admission still blocks.

Gates: single PASS (exact), 8 side by side PASS (exact).

| phase | 004 | 005 |
|---|---|---|
| 48 streams: steady / aggregate / TTFT mean (max) | 20.7 / 12.6 / 80 s (142) | **25.6 / 15.2 / 73 s (93)** |
| 176 streams (14.2 rows/frame) | - | **29.6 / 19.4** / 132 s |
| single stream, UNSEEN prompts | - | **1.92, 1.87 tok/s**, TTFT 6.2 s (guesses 201 sent / 10 right / 20 wrong) |
| single stream, copy task | 2.79 (002b) | **3.15** (52 right / 16 wrong) |

- The frame-time model holds at 14.2 rows/frame: predicted 381 ms on the 25 W boxes, measured
  333-378. The 60 W boxes take 194-225 ms and idle 42 %.
- Memory at 192 slots: 8 GiB available on ranks 1-7, 4.2 GiB on rank 10 (it also holds the head).
- Unseen prompts are the honest single-stream figure: +12-14 % over 1.69. The cross-request table
  guesses often (201 frames for 64 tokens, chains of up to 10 behind each miss) but only 10 of 64
  tokens arrived on a right guess.
- Tokenizer probe (bench/tokenizer_probe.py): 8 of 8 token counts equal o200k_base, including CJK,
  emoji, code and whitespace runs. A model with the o200k vocabulary (Phi-4-mini, gpt-oss) could
  draft for Inkling. What that is worth here: rank 0 would spend ~37 ms of its memory bus per
  draft token, so T0 ~ 93 ms; with a = 0.5 time/token = 0.5 x 93 + 0.5 x 640 = 366 ms = 2.7 tok/s.
- Burst TTFT barely moved (73 s): rank 0 still blocks on the reply of a prefill frame (7-29 s round
  trip) with most of the burst unadmitted. Fixed in binary d5128fcc -> exp 006.
