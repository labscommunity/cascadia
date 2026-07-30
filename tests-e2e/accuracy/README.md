# Packed multi-slot accuracy parity

End-to-end accuracy checks for `--packed-slots` / `--packed-prefix`
(see [docs/perf/NPU_PACKED_SLOTS.md](../../docs/perf/NPU_PACKED_SLOTS.md)).
They drive the real OpenAI-compatible endpoint, compare **complete** generated
text (never a substring or first-sentence match), and generate long enough for
KV drift to surface.

## Why these particular comparisons

Packing changes what a request attends to, so the failure modes are: one
request's output depending on its batch-mates, reused prefix K/V being wrong,
and per-slot KV regions bleeding into each other. Each check targets one:

| check | property | required bar |
|---|---|---|
| solo vs solo-repeat | greedy determinism | exact |
| solo vs batched | **batch-composition invariance** — output must not depend on who else is in flight | exact |
| packed vs baseline | parity with the unpacked path | see note |
| prefix on vs off | KV reuse changes nothing | exact |
| teeth | outputs are distinct, non-empty and long | must hold, else the above are vacuous |

**Note on packed vs baseline.** These are two different compiled graphs, and
greedy `argmax` is discontinuous: an fp16 rounding difference of ~1e-5 on a
near-tied pair of logits flips one token. Expect a high exact-match rate with
occasional single-token substitutions LATE in long generations. Use
`packed_divergence_report.py` to tell that benign case apart from a real bug —
a masking or KV fault diverges EARLY and degrades (repetition, truncation, word
salad), rather than substituting one token and re-converging.

## Running

Start a worker on the config under test, then:

```bash
# main suite (all-different prompts)
python packed_accuracy.py capture http://127.0.0.1:18090 baseline   # --packed-slots 0
python packed_accuracy.py capture http://127.0.0.1:18090 packed4    # --packed-slots 4
python packed_accuracy.py compare baseline packed4
python packed_divergence_report.py baseline packed4                 # characterise any diffs

# prefix cache: needs prompts that SHARE a prefix, or the cache never engages
# and the test passes vacuously
python packed_prefix_accuracy.py capture http://127.0.0.1:18090 pfx_off  # --packed-prefix 0
python packed_prefix_accuracy.py capture http://127.0.0.1:18090 pfx_on   # --packed-prefix 96
python packed_prefix_accuracy.py compare pfx_off pfx_on
```

Both suites also work against a multi-stage pipeline — point them at rank 0's
API port.

## Recorded results (Llama-3.2-1B int4 static, Lunar Lake NPU, OV 2026.2.1)

- determinism 10/10, batch-composition invariance **10/10** (packed and baseline)
- packed vs baseline 8/10 exact; both diffs are a single token
  ("about"/"approximately", "1,143"/"1,145") at 66-73% depth, re-converging
  immediately, identical word counts
- prefix cache on vs off **8/8 exact**, sequential and concurrent

One caveat worth knowing: an early run of this suite failed
batch-composition invariance on **both** configs, which turned out to be a
pre-existing runner chunk-ordering bug (fixed by `abccce5`), not a packing
fault. The harness is sensitive enough to catch that class of defect.
