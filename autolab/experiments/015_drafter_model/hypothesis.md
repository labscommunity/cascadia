# 015: a 0.6B drafter model beside rank 0, a rank 0 that reads replies while it guesses, NIC timers off

**Where we are (012).** One stream, unseen prompts: 2.8-3.05 tok/s. `time/token = a*T + (1-a)*L` with a = 0.30,
T = 44 ms, L = 478 ms (408 ms of bus-bound work + ~25 ms of LAN hops + ~45 ms rank 0 spends computing guess
frames before it reads the reply that matters).

**Changes.**

1. *Drafter model* (`CASCADIA_STREAMS_SPEC_LM`, `lm_draft.rs`). 013: over 192 of this fleet's own responses the
   n-gram drafter names the next token 30-38 % of the time, Qwen3-0.6B 53.6 %, 1.7B 58.2 %, 4B ~61 %, Llama 3.2
   1B/3B 49/53 %. The 0.6B model costs ~0.4 GB of memory traffic per token (rank 0's stage moves ~3 GB per frame),
   so it runs on rank 0's idle CPU cores behind llama.cpp's server; its TEXT is cut into the target's tokens
   (different vocabularies). Exactness is untouched: a guess only decides whether work started early is kept
   (`tests/inkling_streams_spec_lm.rs`: a lying drafter, identical tokens).
2. *Reactive rank 0.* A reply that is already there outranks the next guess frame; and with a drafter model rank 0
   no longer blocks on the reply link while the pipeline has room (the model needs ~60 ms to start over after a
   wrong guess, the old loop would then leave the pipeline empty for the whole trip).
3. *`tx_timer_usecs = 0` on every box* (014: 3.3 ms -> 2.0 ms round trip with one side changed).

**Prediction.** a = 0.45-0.5 live (the drafter is not always ready), L = 440 ms:
`1 / (0.48 x 48 + 0.52 x 440)` = **3.9-4.2 tok/s** on unseen prompts (from 3.0). Gates must stay exact.
**Kill criteria.** Any gate failure; single stream slower than 3.0; aggregate at 176 streams below 60.
