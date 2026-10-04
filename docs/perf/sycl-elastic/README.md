# sycl-llama `--elastic` — figures

All numbers come from [`data.json`](./data.json) (verbatim copy of
`experiments/2026-10-04-elastic-campaign-v3/pr/pr-summary.json`).

| Figure | Point |
|---|---|
| ![fig0](fig0_hero.png) | `fig0_hero.png` — run big models on whatever VRAM you have (headline stats). |
| ![fig1](fig1_vram.png) | `fig1_vram.png` — peak VRAM stock vs `--elastic`, per model. |
| ![fig2](fig2_packing.png) | `fig2_packing.png` — how many 27B instances fit on one card. |
| ![fig3](fig3_load.png) | `fig3_load.png` — cold-start time, stock vs `--elastic`. |
| ![fig4](fig4_tradeoff.png) | `fig4_tradeoff.png` — the cost of streaming everything (`--elastic-vram 0`). |
| ![fig5](fig5_fleet.png) | `fig5_fleet.png` — mixed fleet (2x 27B + 3x 1.5B) on one card, all generating. |
| ![fig6](fig6_laptop.png) | `fig6_laptop.png` — laptop reference: which device to pick (no elastic on laptop). |
| ![fig7](fig7_reliability.png) | `fig7_reliability.png` — 15 load/unload cycles, leak-free, plus second-card parity. |
| ![fig8](fig8_partial.png) | `fig8_partial.png` — `--elastic-vram` budget sweep: speed follows how much of the model fits. |

## Methodology

- **Hardware (B70 box):** 2x Intel Arc Pro B70 32 GB, kernel 7.0.0-28 (`xe`
  driver), Mesa 26.2.2, Intel compute-runtime 14.37020, oneAPI 2026.0
  (icx/icpx SYCL build of llama.cpp + stream-weights patch,
  `patches/llama.cpp/0001-sycl-stream-weights.patch`).
- **Models:** Qwen2.5-1.5B-Instruct Q4_K_M, Qwen3.8-27B UD Q4_K_S,
  Qwen3.6-35B-A3B UD Q4_K_XL (GGUF).
- **Settings:** `--llama-ctx 4096`, KV cache `q8_0`, `-fa on`, temperature 0,
  thinking off.
- **Two data generations:** `single` / `partial` / `cotenant_auto` come from
  the 2026-10-04 post-reboot sweep (cold model file via
  `posix_fadvise(DONTNEED)` before each arm — `drop_caches` is avoided: on
  this xe host it leaks kernel memory; n=3 for stock and fully-streamed
  arms, n=1-2 for budget arms). Fleets, latency, lifecycle and sweeps come
  from campaign v3. The v3 MoE stock number (34.2 t/s) is superseded —
  post-reboot it measures 78 t/s with both binaries (host state).
- **Runs:** means over repetitions; decode/prefill are server-side
  `timings.predicted_per_second` / `prompt_per_second` (not end-to-end).
- **VRAM:** kernel `vram_mm` under debugfs, sampled ~3 Hz during
  load + decode; peak reported.
- **Hunter laptop (fig6 only):** Core Ultra 9 285H (Arc 140T iGPU, RTX 5060
  Laptop, AI Boost NPU), 32 GB RAM, Windows 11, OpenVINO GenAI 2026.4 via
  `ov-genai`; 48 tokens after an 8-token warmup.

## Reproduce

```bash
# 1. build the patched llama-server and point cascadia at it
scripts/build-llama-stream.sh ~/llama-stream
export CASCADIA_LLAMA_BIN=~/llama-stream/build/bin/llama-server
source /opt/intel/oneapi/setvars.sh   # child needs the oneAPI runtime libs

# 2. per arm (stock arms drop --elastic); the campaign used SYCL1
cascadia run /path/to/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \
  --engine sycl-llama --device SYCL1 --llama-ctx 4096 \
  --llama-args "-ctk q8_0 -fa on" --api 127.0.0.1:19600          # stock
  # + --elastic                                                  # elastic
cascadia run /path/to/Qwen3.8-27B-UD-Q4_K_S.gguf   ... same flags
cascadia run /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ... same flags

# 3. VRAM (needs root): SYCL0 = pci 0000:0b:00.0, SYCL1 = 0000:0f:00.0 here
sudo cat /sys/kernel/debug/dri/<pci>/tile0/vram_mm   # "usage:" line

# 4. --elastic-vram sweep + co-tenancy (harnesses live in
#    experiments/2026-10-04-elastic-campaign-v3/pr/partial/, not committed)
cd experiments/2026-10-04-elastic-campaign-v3/pr/partial
python3 partial.py sweep 27B stock unfused v0 v2 v4 v8 v10 v12 auto v40
python3 partial.py sweep MoE stock unfused v0 v4 v10 v16 auto
python3 partial.py sweep 1.5B stock unfused v0 auto
python3 cotenant.py sweep/cotenant-auto-2.json 2

# 5. figures
experiments/2026-10-04-elastic-campaign-v3/venv/bin/python \
  docs/perf/sycl-elastic/make_figs.py
```

The measurement harnesses live in
`experiments/2026-10-04-elastic-campaign-v3/` (`v3.py` arms,
`v3_life.py` / `pr/life_retry.py` lifecycle, `v3_conc.py` fleets,
`v3_sampler.py` VRAM sampler, `pr/partial/partial.py` + `cotenant.py`
for the budget sweep).
