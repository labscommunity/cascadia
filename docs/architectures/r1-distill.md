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

## End-to-end test (iGPU + iGPU)

**End-to-end pipeline-parallel works on Intel iGPU (Lunar Lake).**
Verified 2026-05-21 with `r1-distill-qwen-1.5b` (int4, 917 MB total)
sharded across two Lunar Lake iGPU AI PCs (rank 0 → rank 1), API on
rank 0:8000. (The recipe below is host-agnostic — substitute any two
AI PCs reachable over the network; `node-a.local` / `node-b.local`
and the `192.168.1.10` / `192.168.1.20` addresses below are
placeholders — node A is .10, node B is .20.)

```console
$ curl http://192.168.1.10:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
    "model": "r1-distill-qwen-1.5b",
    "messages": [{"role": "user", "content": "What is 17 * 23? Think step by step."}],
    "max_tokens": 96
  }'

{"choices":[{"message":{"role":"assistant","content":
  "First, I need to multiply 17 by 23.\n\nTo simplify the
  calculation, I'll break down 23 into 20 and 3.\n\nMultiplying 17
  by 20 gives 340.\n\nNext, I'll multiply 17 by 3, which equals 51.
  \n\nFinally, I'll add the two results together: 340 + 51 = 391.\n
  </think>\n\nTo calculate ("}, ...}]}
```

(17 × 23 = 391 ✓; the `</think>` marker is the R1 reasoning-chain
end-of-chain-of-thought.)

**CPU-plugin note (fixed):** an earlier exporter emitted init-less
`ReadValue` nodes (via `apply_make_stateful_transformation`) that the
OpenVINO CPU plugin rejected with:

```text
Check 'idx < parentEdges.size()' failed at
src/plugins/intel_cpu/src/node.cpp:687:
Node ReadValue_33408 contains less parent edges than 0
```

This is fixed (#57/#62): `make_stateful_with_init` now builds the
`ReadValue`/`Assign` nodes directly with explicit zero-length inits, so
the v5_canonical_inputs shards load on the CPU plugin as well as the
iGPU. Pipeline-parallel `--engine ov-runtime` runs on `--device CPU`,
`GPU`, and `NPU`.

Steps to reproduce the pipeline-parallel run:

1. Export 2-stage int4 shards on an export host (only needs CPU +
   plenty of disk; nothing to do with the actual runtime). `int4` is
   important — `int8` triggers a similar OV stateful issue when the
   nncf compressed weights interact with `ReadValue` ops.

   ```bash
   ssh <export-host> "source ~/.venv/export/bin/activate && \
     cascadia shard \
       --model r1-distill-qwen-1.5b \
       --output-dir /tmp/r1_int4 \
       --num-stages 2 --quantization int4"
   ```

   (`r1-distill-qwen-1.5b` resolves via `tools/model_aliases.py` to
   `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.)

2. Tarball + ship shards to both AI PCs:

   ```bash
   ssh <export-host> 'tar -C /tmp -czf /tmp/r1_int4.tgz r1_int4'
   scp <export-host>:/tmp/r1_int4.tgz /tmp/r1_int4.tgz
   scp /tmp/r1_int4.tgz cascadia@node-a.local:C:/Users/cascadia/r1_int4.tgz
   scp /tmp/r1_int4.tgz cascadia@node-b.local:C:/Users/cascadia/r1_int4.tgz
   ssh cascadia@node-a.local 'powershell -Command "mkdir C:\tmp\r1_int4 -Force; tar -xzf C:\Users\cascadia\r1_int4.tgz -C C:\tmp\r1_int4"'
   ssh cascadia@node-b.local 'powershell -Command "mkdir C:\tmp\r1_int4 -Force; tar -xzf C:\Users\cascadia\r1_int4.tgz -C C:\tmp\r1_int4"'
   ```

3. Launch the rank-1 worker on **node-b** (downstream — Lunar Lake
   iGPU). Keep the SSH session OPEN; on Windows, closing the
   parent SSH session kills the worker even with
   `Start-Process -WindowStyle Hidden` (a known OpenSSH-on-Windows
   quirk):

   ```bash
   ssh cascadia@node-b.local 'powershell -NoProfile -ExecutionPolicy Bypass -Command "
     $env:PATH = ''C:\openvino_genai\runtime\bin\intel64\Release;C:\openvino_genai\runtime\3rdparty\tbb\bin;'' + $env:PATH;
     & C:\cascadia\cascadia.exe worker --rank 1 --total 2 --engine ov-runtime --device GPU --model C:\tmp\r1_int4\r1_int4 --listen 0.0.0.0:9100"'
   ```

4. In a second terminal, launch the rank-0 worker on **node-a**
   (upstream — Lunar Lake iGPU — serves the API):

   ```bash
   ssh cascadia@node-a.local 'powershell -NoProfile -ExecutionPolicy Bypass -Command "
     $env:PATH = ''C:\openvino_genai\runtime\bin\intel64\Release;C:\openvino_genai\runtime\3rdparty\tbb\bin;'' + $env:PATH;
     & C:\cascadia\cascadia.exe worker --rank 0 --total 2 --engine ov-runtime --device GPU --model C:\tmp\r1_int4\r1_int4 --next 192.168.1.20:9100 --api 0.0.0.0:8000"'
   ```

5. From a third terminal:

   ```bash
   curl http://node-a.local:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "r1-distill-qwen-1.5b",
     "messages": [{"role": "user", "content": "What is 17 * 23?"}],
     "max_tokens": 64
   }'
   ```

   Expected response: an R1-style reasoning chain ending in `391`.

The KV-cache reset between requests is exercised by running the curl
multiple times in a row.
