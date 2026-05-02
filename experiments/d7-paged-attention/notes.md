# d7: paged-attention re-export — not a config option

## Question
Can we re-export Llama 3.1 8B as v5 shards with paged-attention applied,
to engage LLMPipeline-equivalent runtime optimizations in the multi-stage
ov-runtime / ov-dist-spec engines?

## Investigation

`optimum-cli export openvino --help` (relevant flags):

```
--weight-format {fp32,fp16,int8,int4,mxfp4,nf4,cb4}
--quant-mode {int8,f8e4m3,f8e5m2,cb4_f8e4m3,int4_f8e4m3,int4_f8e5m2}
--disable-stateful
```

There is no `--paged-attention` or similar flag. PA is **not baked in at
export time**.

## How LLMPipeline engages PagedAttention

In `openvino_genai.LLMPipeline`, PA is applied via the
`SDPAToPagedAttention` transformation pass at compile time (runtime on
the target hardware). The pass rewrites scaled-dot-product attention
operators in the IR to use OV's paged KV cache + ring-buffer KV reads,
unlocking the U8 KV cache and dynamic-quant XMX optimizations the GPU
plugin applies for PA-shaped models.

The pass operates on the IR after load, not at export. So the v5 shards
we have (`shards_2stage_v5_beam`) actually CAN be PA-ified at compile
time — but only if the runtime applies the pass.

## How ov-runtime / ov-dist-spec compile

Looking at `tahoma/worker/engines/openvino/ov_runtime.py`:

- The shard's IR is loaded via `ov.Core().compile_model(stage_dir/openvino_model.xml, device)`.
- There is **no** `SDPAToPagedAttention` pass invoked.
- KV cache management is the engine's own (stateful via ReadValue/Assign ops baked into the IR at export time).

So the multi-stage engines bypass the LLMPipeline runtime path entirely
and miss the PagedAttention optimization (and U8 KV cache + dynamic
quant that goes with it).

## What it would take to engage PA in multi-stage

Two options:

1. **Apply `SDPAToPagedAttention` pass after compile** in `ov_runtime.py`.
   Inspect openvino-genai source for the exact API:

   ```python
   import openvino as ov
   from openvino_genai import (apply_paged_attention_transform_or_similar)

   core = ov.Core()
   model = core.read_model(stage_dir / "openvino_model.xml")
   # apply transformation pass
   compiled = core.compile_model(model, "GPU")
   ```

   This is engineering work. Need to find the exact public API in
   openvino-genai for the PA transform.

2. **Re-export shards via openvino-genai's pipeline**, which applies PA
   at export. openvino-genai has an internal export path that produces
   PA-shaped IR. Unclear if exposed via optimum-cli.

## Conclusion

The d7 hypothesis ("re-export with paged-attention engages LLMPipeline-class
optimizations in multi-stage") **is not actionable as a config change**.
It requires modifying the ov-runtime engine code or finding an
openvino-genai export pipeline.

Given that the d3 winning config already gets 38.49 tok/s (+36% over
single-node best) WITHOUT this optimization, the marginal value of
adding PA to multi-stage is unclear — could be another +10-30% or
could be no-op if the v5 shards already include some PA-equivalent
graph rewriting.

**Open follow-up**: investigate the openvino-genai source to find the
public API for the PagedAttention transform. If straightforward, modify
`ov_runtime.py` and `dist_spec.py` to apply it after compile. Re-bench.

## Pragmatic answer

Stick with d3 winning config (ov-dist-spec K=4 + FastDraft + 256+ output
= 38.49 tok/s). Distributed beats single-node already; further gains
require engine-level engineering.
