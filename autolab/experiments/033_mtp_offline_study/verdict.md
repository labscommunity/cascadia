# 033 verdict (offline, no fleet time): the model's own MTP head names the next token 0.73 of the time; the delayed-guess idea is closed

The teammate's plan (`spec.md`: E1 dump, E2 logit lens, E4 shipped MTP head), run by a background research agent on
2026-09-20 (1 h 47 min). Scripts, the dump tool and the full tables: `autolab/research/mtp_offline/`,
`crates/cascadia-engine-sparse-moe/examples/inkling_spec_dump.rs`.

**What the head is** (read from the checkpoint and from vLLM's / SGLang's Inkling MTP code; the HF repo ships no
modeling file and transformers ignores `model.mtp.*`): `mtp.safetensors`, 10.52 GB bf16, eight chained modules.
Module k at position t predicts token t+k+2 from
`input_proj(cat[hidden_norm_k(prev), embed_norm_k(rmsnorm(embed[x_{t+k+1}], llm.embed_norm))])` (hidden first) through
one ordinary DENSE Inkling block with its OWN KV cache and conv state over the whole sequence, then the main model's
unembed (`h / 24`, no norm on the block output). `prev` is the trunk's POST-final-norm state for module 0 and the raw
output of module k-1 afterwards. Modules 1 and 3 are global-attention blocks, the other six sliding (window 512).

**Data.** 36 prompts (3 per family of the 013 corpus) rendered exactly as the API does, generated greedily for 160
tokens on the Mac Pro's CPU path (0.45 s/token prefill, 0.69 decode, 77 min); dumped: tokens, final residual at every
position, the ten rank-boundary residuals at generated positions. The CPU path's text departs from the fleet's after
7-70 tokens (numerics), so these are the CPU path's own tokens. Python norm + unembed reproduces the dumped argmax
at 100 % of positions.

| drafter, first draft, teacher-forced | right |
|---|---|
| n-gram tables (013) | 0.30-0.38 |
| Qwen3-0.6B / 4B (013) | 0.54 / 0.61 (0.41 / 0.49 on explain, 0.39 / 0.45 on story) |
| **shipped MTP head, module 0** (n = 5,724) | **0.726**: explain 0.69, story 0.63, code 0.75, rewriting 0.79, translation 0.81, arithmetic 0.87 |

Chained drafts, given the chain right so far: 0.69 / 0.73 / 0.73 / 0.77 / 0.79 / 0.83 / 0.85 for drafts 2-8 **when
each module's context is kept current**; expected accepted drafts per chain of eight 2.48 = 166 ms per token at
today's trip and stage times = **6.0 tok/s** for one stream (5.1 explain, 4.5 story, 8.4 arithmetic). If modules 1-7
run only at the anchor after each miss (no per-token upkeep): 2,165 chains, a1 0.66 at anchors, E = 1.67 =
**4.9 tok/s**. Without their sequence context the deeper modules are worthless (draft 2: 0.11), and module 0 alone
without context drops to 0.37: a wiring must keep the used depths' KV and conv state current.

Teeth: swapped concat order 0.001, pre-final-norm input 0.024, embedding of the wrong position 0.047, raw embedding
0.576, no short convs 0.611, no relative bias 0.641, halves instead of interleaved gate/up 0.447, against 0.726.

Cost per draft from tensor sizes at 110 GB/s: 462 MB (int4 MLP, int8 attention and input_proj) = 4.2 ms, plus the
unembed the drafting rank does not hold today: full int8 vocabulary 1.24 GB = 11.2 ms, or a prefix of 65,536 rows
0.4 GB = 3.7 ms (a1 0.696; 95 % of accepted drafts have an id below 65,536). 15.4 / 7.9 ms per draft: inside the
plan's 25 ms bar. Keeping a module current costs another 4.2 ms per real token on rank 0's bus.

**Logit lens (E2): closed.** The state leaving ranks 0-4 names the next token 0.000 of the time raw (0.49 only at
rank 9); a ridge-fitted affine lens reaches 0.34-0.39 for ranks 0-4, below every rank's break-even bar
(0.42-0.73). Guessing later from a deeper rank does not pay.

**What it means here.** a1 0.73 (0.63-0.69 on prose) is below the plan's 0.75-0.85 prediction and well above every
drafter measured so far. One stream: 3.4-4.1 -> ~5-6 tok/s, not 10. Fifteen streams: a guess row costs a full set of
expert reads, so two rows per stream yield 1.73 tokens for ~1.62x the stage time: about +7 % (PHYSICS.md), and about
2x per stream at 3-8 streams where the pipeline has idle room. Passes the plan's rule (a1 >= 0.7, <= 25 ms per draft):
the next step is the fleet wiring (queue item), a multi-day build: export the MTP blocks, run them on rank 0 with
their own KV/conv state, carry the final state on the reply link, and guess rows for many streams in the scheduler.

Caveats: 36 prompts (3 per family, 477 positions each); CPU int4 path, not the fleet's f16-fused path; the tok/s
figures are a renewal model at today's L = 466 ms and T = 45 ms with the head's own cost on rank 0 not yet charged.
