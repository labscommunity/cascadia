# d6: large model that doesn't fit single-node — DEFERRED

## Goal
Demonstrate that distributed inference is the only option for models too
large to fit on one node. Targets: Mixtral 8x7B INT4 (~24 GB), Llama 3.3
70B INT4 (~35 GB).

## Status: deferred

Multiple attempts to pull `OpenVINO/Mixtral-8x7B-Instruct-v0.1-int4-ov`
from HuggingFace Hub stalled at 4/13 files (small files done; large
shards never started). The HF Hub redirects to `cas-bridge.xethub.hf.co`
URLs which appear to be either rate-limited or slow over our home WiFi.

After ~10 min of waiting per attempt with no progress past 16 MB,
killed downloads and parked this campaign.

## Why this matters for tahoma's mission

Tahoma's distinguishing value vs single-node OV:
1. ~~Faster per-token~~ — d3 showed distributed CAN be faster (+36%) but
   it's not the headline argument.
2. **Capacity** — running models too big for one node. This is the
   killer demo and the actual reason distributed exists.

We have NOT verified the capacity argument empirically. Mixtral 8x7B
INT4 needs ~12-13 GB of weights (only 2 of 8 experts active per token
but all 8 must be loaded). alpha B390 dGPU likely has 12 GB VRAM —
tight; charlie 140V iGPU shares 16 GB system memory — also tight.

Single-node behavior on these tight memory configs is unknown without
benching. Distributed (12 GB / node split) would be guaranteed to fit.

## What's needed to complete d6

1. Re-attempt Mixtral download with a faster network connection (alpha
   on Ethernet not WiFi, or seed via a USB drive).
2. Or pull a smaller "doesn't-fit" model: Llama 3.3 70B INT4 (~35 GB —
   even worse), or Qwen 2.5 32B INT4 (~16 GB — borderline).
3. Re-export as multi-stage v5 shards via rainier's `export_shards_dynamo.py`.
4. Copy stage_1 to charlie via TB4.
5. Run ov-runtime distributed bench. (ov-dist-spec needs a draft model
   too — would need a smaller MoE-compatible draft, not trivial.)

This is multi-hour work. Out of scope for current autolab session.

## Pragmatic answer

The d3 finding (38.49 tok/s ov-dist-spec K=4 + FastDraft on Llama 3.1
8B INT4) is the demonstrated distributed perf win. The capacity argument
remains a future work item.

For tahoma deployment: the d3 config IS the right default for 8B
distributed inference. Capacity testing on bigger models is a separate
engineering project.
