# DeepSeek R1 Distills

DeepSeek-R1 (January 2025) was a 671B-parameter MoE-MLA model. To
share the reasoning behaviour with the open community at sizes that
fit ordinary hardware, DeepSeek distilled it into several smaller
checkpoints, each one fine-tuned over a different base model:

| Distill | Base | `model_type` | INT4 size | Single Lunar Lake? | 2-stage shard? |
|---------|------|--------------|-----------|--------------------|----------------|
| `DeepSeek-R1-Distill-Qwen-1.5B` | Qwen2.5-Math-1.5B | `qwen2` | ~900 MB | yes | overkill |
| `DeepSeek-R1-Distill-Qwen-7B` | Qwen2.5-Math-7B | `qwen2` | ~4 GB | yes | nice for demo |
| `DeepSeek-R1-Distill-Llama-8B` | Llama-3.1-8B | `llama` | ~4.5 GB | yes | nice for demo |
| `DeepSeek-R1-Distill-Qwen-14B` | Qwen2.5-14B | `qwen2` | ~8 GB | tight | comfortable |
| `DeepSeek-R1-Distill-Qwen-32B` | Qwen2.5-32B | `qwen2` | ~17 GB | no | required |
| `DeepSeek-R1-Distill-Llama-70B` | Llama-3.1-70B | `llama` | ~38 GB | no | 3+ box |

Because the distills inherit their base model's `config.json`,
`cascadia shard` routes them through the existing Qwen2 / Llama
paths with **no special handling needed**:

```bash
cascadia shard \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --output-dir ~/cascadia/r1-distill-7b-2stage \
  --num-stages 2 \
  --quantization int4
```

## Reasoning output

The R1 distills emit a `<think>...</think>` block before the final
answer. This is just tokens — Cascadia's OpenAI-compatible API does
not strip it. If you want only the final answer, post-process by
splitting on `</think>` and taking the second half:

```python
import openai
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x")
resp = client.chat.completions.create(
    model="r1-distill-qwen-7b",
    messages=[{"role": "user", "content": "What is 17 * 23?"}],
)
text = resp.choices[0].message.content
answer = text.split("</think>")[-1].strip() if "</think>" in text else text
```

## End-to-end test (miner + beta)

**Known blocker (2026-05-21):** Pipeline-parallel `--engine ov-runtime`
on the **OpenVINO CPU plugin** currently fails to load `cascadia
shard`'s v5_canonical_inputs stateful IR with:

```
Check 'idx < parentEdges.size()' failed at
src/plugins/intel_cpu/src/node.cpp:687:
Node ReadValue_33408 contains less parent edges than 0
```

This reproduces with `openvino_genai 2026.1.0` directly (not just via
cascadia), so it's an upstream OV CPU plugin issue with how
`apply_make_stateful_transformation` + `fuse_cache_reorder`
restructure the IR (the `beam_idx` Gather inserted in front of each
`ReadValue` confuses the CPU plugin's edge accounting). **Workarounds:**

1. **Use `--device GPU` on rank-1 worker** — Intel iGPU plugin
   accepts the IR. Rank-0 worker still needs a host with iGPU.
   Verified to *start* on a Lunar Lake AI PC; full e2e generation
   follows after the upstream fix.
2. **Use `--engine ov-genai` single-stage** — the single-stage path
   bundles the optimum-cli export which doesn't hit the same edge
   accounting issue. Loses pipeline-parallel.
3. **Wait on the upstream OV fix** — track in a separate cascadia
   issue.

The pipeline-parallel recipe below is therefore the *intended* shape
of the test once the CPU plugin is fixed; the test infrastructure
(shards built on miner, copied to beta, workers wired up, API
exercised) all works.

See `tools/scripts/test_r1_distill_pipeline_parallel.sh`. Steps:

1. Export 2-stage shards on the miner:

   ```bash
   ssh miner "source ~/.venv/rainier/bin/activate && \
     cd ~/cascadia && \
     python tools/export_shards.py \
       --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
       --output-dir /tmp/r1_distill_15b_2s \
       --num-stages 2 --quantization int8"
   ```

2. Copy the shards to beta:

   ```bash
   ssh miner "tar -cz -C /tmp r1_distill_15b_2s" | \
     ssh cascadia@beta.local "tar -xz -C /tmp"
   ```

3. Launch the rank-1 worker on beta (downstream):

   ```bash
   ssh cascadia@beta.local "C:\\tahoma\\tahoma.exe worker --rank 1 --total 2 \
       --engine ov-runtime --device CPU \
       --model C:\\tmp\\r1_distill_15b_2s --listen 0.0.0.0:9100"
   ```

4. Launch the rank-0 worker on miner (upstream + API):

   ```bash
   ssh miner "~/cascadia/target/release/cascadia worker --rank 0 --total 2 \
       --engine ov-runtime --device CPU \
       --model /tmp/r1_distill_15b_2s \
       --next 192.168.86.31:9100 --api 0.0.0.0:8000"
   ```

5. From your laptop:

   ```bash
   curl http://miner:8000/v1/chat/completions -d '{
     "model": "r1-distill-qwen-1.5b",
     "messages": [{"role": "user", "content": "Capital of France?"}],
     "max_tokens": 64
   }'
   ```

Expected response: a `<think>` chain followed by "Paris" (or
similar). The shard's KV-cache reset between requests is exercised by
running the curl twice.
