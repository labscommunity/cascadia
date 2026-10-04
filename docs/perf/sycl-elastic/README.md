# sycl-llama `--elastic` — figures

All numbers come from [`data.json`](./data.json) (verbatim copy of
`experiments/2026-10-04-elastic-campaign-v3/pr/pr-summary.json`).

| Figure | Point |
|---|---|
| ![fig0](fig0_hero.png) | `fig0_hero.png` — big models on a fraction of the VRAM (headline stats). |
| ![fig1](fig1_vram.png) | `fig1_vram.png` — peak VRAM stock vs `--elastic`, per model. |
| ![fig2](fig2_packing.png) | `fig2_packing.png` — how many 27B instances fit on one card. |
| ![fig3](fig3_load.png) | `fig3_load.png` — cold-start time, stock vs `--elastic`. |
| ![fig4](fig4_tradeoff.png) | `fig4_tradeoff.png` — the honest cost: decode slows to ~0.6 t/s on 27B. |
| ![fig5](fig5_fleet.png) | `fig5_fleet.png` — mixed fleet (2x 27B + 3x 1.5B) on one card, all generating. |
| ![fig6](fig6_laptop.png) | `fig6_laptop.png` — laptop reference: which device to pick (no elastic on laptop). |
| ![fig7](fig7_reliability.png) | `fig7_reliability.png` — 15 load/unload cycles, leak-free, plus second-card parity. |

## Methodology

- **Hardware (B70 box):** 2x Intel Arc Pro B70 32 GB, kernel 7.0.0-28 (`xe`
  driver), Mesa 26.2.2, Intel compute-runtime 14.37020, oneAPI 2026.0
  (icx/icpx SYCL build of llama.cpp + stream-weights patch,
  `patches/llama.cpp/0001-sycl-stream-weights.patch`).
- **Models:** Qwen2.5-1.5B-Instruct Q4_K_M, Qwen3.8-27B UD Q4_K_S,
  Qwen3.6-35B-A3B UD Q4_K_XL (GGUF).
- **Settings:** `--llama-ctx 4096`, KV cache `q8_0`, `-fa on`, temperature 0,
  thinking off.
- **Runs:** 3 cold-cache repetitions per arm (`echo 3 > drop_caches` before
  each); reported numbers are means.
- **VRAM:** kernel `vram_mm` under debugfs, sampled ~3 Hz during
  load + decode; peak reported.
- **Speed:** decode/prefill from llama-server's own `timings` block.
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

# 4. figures
experiments/2026-10-04-elastic-campaign-v3/venv/bin/python \
  docs/perf/sycl-elastic/make_figs.py
```

The measurement harnesses live in
`experiments/2026-10-04-elastic-campaign-v3/` (`v3.py` arms,
`v3_life.py` / `pr/life_retry.py` lifecycle, `v3_conc.py` fleets,
`v3_sampler.py` VRAM sampler).
