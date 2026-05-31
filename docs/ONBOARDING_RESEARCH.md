# Onboarding research — how comparable inference products onboard users

Background research behind the onboarding work tracked in
[issue #52](https://github.com/labscommunity/cascadia/issues/52). Surveyed
the install → first-inference flow of distributed-inference peers and
polished consumer local-inference apps, with each claim adversarially
fact-checked (3-vote, kill on 2/3 refute; 22 of 25 verified claims
survived). Confidence and caveats are noted; this is a snapshot — exo and
the consumer apps move fast.

## The two camps

**Frictionless consumer apps — Ollama is the gold standard** *(high confidence)*
- One-line install (`curl -fsSL https://ollama.com/install.sh | sh`, OS auto-detected), then a single `ollama run <model>` that **auto-downloads the model on first use**.
- Installs as a **background service** (LaunchAgent / systemd / Windows service) exposing an **OpenAI-compatible API at `localhost:11434/v1/`**.
- Bare `ollama` opens an **arrow-key interactive menu** so a first-timer never memorizes a flag.

**Zero-config distributed — exo is the killer pattern** *(high confidence)*
- Run the **identical `exo` command on each device** → automatic peer discovery (UDP/multicast, no master/worker, ~2.5s announce) across heterogeneous hardware → splits the model from a **realtime device-topology view**. README: *"That's it! No configuration required."*
- One command (`uv run exo`) serves a dashboard + OpenAI-compatible API at `localhost:52415`.
- **But exo's friction is the install, not the runtime:** the official path is a multi-step source build (git clone → `npm install && npm run build` → `uv run exo`) needing uv, node, rust nightly. No one-line installer. (The old `pip install -e .` path was *refuted* as current.)

**Middle ground — GPUStack** *(high confidence):* same one-line installer for every role; a worker joins by re-running it with `--server-url` + `--token`.

**Painful end — vLLM** *(high confidence):* every node needs an identical environment (containers recommended), Ray installed separately, head/worker started manually via `run_cluster.sh`, verified with `ray status`. The anti-pattern.

## The finding that matters most for Cascadia: OpenVINO is a first-class onboarding hazard

All **high confidence (3-0 verified):**

1. **OpenVINO GPU inference is never a single install.** Beyond the SDK you need the Intel graphics driver + OpenCL runtime + Compute Runtime + IGC + Level-Zero. On Ubuntu 22.04/24.04: `apt-get install ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero` **plus `usermod -a -G render`**. (Windows bundles OpenCL in the driver — closer to one install.)

2. **The most dangerous failure is silent and hits Cascadia's exact target hardware.** On a Core Ultra 7 155H / 285H with Arc iGPU, a *correct* OpenVINO + driver install can still leave `Core().available_devices == ['CPU']` with **no error**. The GPU can be fully functional at the OpenCL layer (`clinfo` shows "Intel Arc Graphics", 128 CUs) and still be invisible to OpenVINO's GPU plugin. **OS-level diagnostics do not predict OpenVINO detection.** (OV #28892, openvino_notebooks #2702, ollama #12948.)

3. **The oneAPI/IPEX wrapper path compounds it:** separate prereq guide, dedicated conda env, manual env vars (`OLLAMA_NUM_GPU=999`, `ZES_ENABLE_SYSMAN=1`), `source setvars.sh`, and cryptic missing-`.so` errors (`libmkl_core.so.2`, `libsvml.so`).

## Cross-cutting patterns

| Frictionless (copy) | Painful (avoid) |
| --- | --- |
| One artifact, one command to first inference | Multi-step source build as the only path |
| Auto-download models on first reference | Manual download + path wrangling |
| Sensible defaults; bare command does the right thing | Many required flags, useless defaults |
| Auto-discovery; identical command per node | Manual head/worker scripts, hand-coded peer addresses |
| Background service + OpenAI API on a fixed port | Foreground process, no API unless flagged |
| Interactive / no-args fallback | Flag memorization required |
| **Actionable errors + hardware self-check** | **Silent CPU fallback / cryptic link errors** |

## How this maps to Cascadia's changes

- Silent CPU-fallback hazard → **`cascadia doctor`** makes it loud and actionable.
- "Never a single install" → **INSTALL.md** + **`scripts/setup-openvino.{sh,ps1}`** (the apt set + render group) + a **Dockerfile** that bundles the stack.
- Opaque build failure → **actionable `build.rs` error** when `INTEL_OPENVINO_DIR` is wrong/missing.
- Sensible defaults / one command → **`cascadia run <model>`**; auto-download documented.
- Peer-address guesswork → **`cascadia discover`** (read-only LAN listing); full auto-ring formation deferred.
- Silent peer hang → **loud, periodic connect feedback** in the transport.

## Caveats & open questions

- **Verified depth is on Ollama and exo.** "Easy onboarding" claims for **LM Studio / Jan / GPT4All** were *refuted* (uncharacterized here), and **petals / llama.cpp-RPC / Cake** produced no surviving verified claims.
- The OpenVINO silent-fallback evidence is from GitHub issues — representative and on Cascadia's hardware class, but **not a measured frequency**. Whether `doctor` can reliably *self-heal* (render group vs. missing level-zero package vs. ICD config) is unresolved.
- GPUStack's one-line pattern is from 0.3 docs (shape persists in later versions). Windows-vs-Linux OpenVINO GPU setup differs materially — the Linux apt path is where the friction concentrates.

### Primary sources

- exo README — <https://github.com/exo-explore/exo>
- Ollama docs — <https://docs.ollama.com/quickstart>, `/cli`, `/api/openai-compatibility`
- GPUStack multi-node — <https://docs.gpustack.ai/0.3/tutorials/setting-up-a-multi-node-gpustack-cluster/>
- vLLM distributed serving — <https://docs.vllm.ai/en/stable/serving/parallelism_scaling/>
- OpenVINO Intel-GPU config — <https://docs.openvino.ai/2025/get-started/install-openvino/configurations/configurations-intel-gpu.html>
- Silent CPU fallback — OV issue [#28892](https://github.com/openvinotoolkit/openvino/issues/28892), ollama [#12948](https://github.com/ollama/ollama/issues/12948)
- IPEX-LLM Ollama quickstart — <https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/ollama_quickstart.md>
