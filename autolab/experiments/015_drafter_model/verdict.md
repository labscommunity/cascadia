# 015 verdict: KEEP (015c). One stream on unseen prompts 3.0 -> 3.1-3.7 (prose), 5.9-10.9 (structured tasks); exact.

Three releases:

- **015 (binary 1d171a37): broken within minutes, rolled back.** Every lone request failed with "engine made no
  progress for 3 consecutive steps": with a drafter model a speculation round can end without a token (no reply yet,
  it tops the pipeline up instead), and the serving loop counts three empty steps as a wedged engine. The pipeline
  test had a drafter that answered instantly and ranks that replied in microseconds, so rank 0 never found the
  pipeline with room and no reply. Fix: rounds repeat inside the step until one yields a chunk; the test now has a
  40 ms drafter and 8 ms ranks and fails the old code with the runner's own rule.
- **015b (a54199fc):** unseen explain-type prompts 3.39 / 3.48 / 3.70 tok/s (012: 2.79 / 2.99 / 3.05), rank 0's round
  trip 575-607 -> 409 ms (reads replies between guess frames; NIC timers off), 176 streams 65.5 tok/s steady (64.2).
  But memorised prompts fell from 8-12 to 4-5 tok/s: the model (right ~4 in 10 live) had replaced the table (9 in 10).
- **015c (b0b7995c):** the tables answer where they are sure (own last four tokens seen before, or a shared 3-token
  context followed by one token >= 90 % of the time), the model elsewhere; the drafter sees the last user turn, not
  the rendered template. Memorised prompts back to 7.0 / 10.7 / 10.3, copy task 12.8 (5.8 in 015b, 3.2 before).

| one stream, unseen prompt, 128 tokens | tok/s | guesses right / wrong (model's share) |
|---|---|---|
| true/false + justification | **10.95** | 47 / 6 (3 / 3) |
| arithmetic word problem | **8.19** | 100 / 19 (54 / 14) |
| translation | 6.60 | 82 / 23 (13 / 17) |
| formal rewrite | 5.85 | 78 / 23 (14 / 19) |
| code | 3.67 | 43 / 48 (36 / 43) |
| explain (the `fresh` family, 96-160 tokens) | 3.09-3.43 | 33-69 / 51-87 (18-52 / 46-79) |
| story | 3.38 | 52 / 71 (39 / 67) |

(The tables had seen other prompts of these families during 013's collection run, which is how a deployed fleet
behaves too: it learns the kinds of requests it serves.)

Cost: the drafter's four CPU threads and ~0.4 GB of memory traffic per token slow rank 0's own frame from 44 to
53 ms (one stage of eleven: +9 ms on a 409 ms trip). Exactness: both gates pass; `inkling_streams_spec_lm.rs`
feeds the pipeline a drafter that lies every fourth word and gets identical tokens.

Deployment, all in `fleet-overrides.env` (no new infra file): rank 0 fetches llama.cpp's official CPU build
(b11056, sha256 7493...601a) and Qwen3-0.6B Q4_K_M (9acf...1b14) through the lab proxy, verifies both, and runs the
server as `nobody` on 127.0.0.1:8099 inside the worker's unit. No server -> the engine uses the tables as before.
