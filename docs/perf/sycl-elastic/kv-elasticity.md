# KV-cache elasticity: design note

Why `--elastic` streams weights but leaves the KV cache fully on the
device, and what a KV answer would look like.

## What is already measured

Streaming does not touch KV. On the 27B Q4_K_S (`-ctk q8_0 -fa on`,
fully streamed weights) the child's peak device footprint above baseline
grows with context, not with the stream:

| `-c` | peak VRAM above idle |
|---|---|
| 4,096 | +3.1 GiB |
| 65,536 | +6.0 GiB |
| 131,072 | +9.2 GiB |

Past roughly 64k context the KV allocation, not the resident-weight
budget, is the footprint driver on this model. `--elastic-vram` is a
weight residency budget; it cannot shrink KV.

## Why per-token KV streaming is not viable

Weight streaming works because decode reads each resident-missing weight
exactly once per token, and the layer order is fixed, so reads pipeline
cleanly behind compute. KV is different: it is appended every token and
re-read every step for the full active context. Streaming KV on demand
would add on the order of GiBs of PCIe reads per token on a 100k+ context —
two orders of magnitude worse than the ~13 GiB/token weight stream that
already runs at ~0.7-0.8 t/s. The arithmetic closes the question.

## The shapes that could work

1. **Idle-sequence suspend/resume.** Move a slot's KV to host memory when
   its session goes idle, page it back on resume. This targets the real
   co-tenant problem (a parked 128k session holding ~9 GiB) without
   touching the per-token path. The mechanisms exist in llama.cpp
   (`llama_state_*` save/load) but they operate on whole-model state, not
   per-slot KV; per-slot offload needs upstream work.
2. **Driver-managed virtual memory.** Let the driver oversubscribe KV —
   the Section-C oversubscription arm already proved the mechanism is
   stable (3x27B, no resets). The missing piece is controlling *what*
   spills; that is driver vmem / vAttention-class territory, not user
   space.
3. **Quantized / compressed KV at longer context.** Already available via
   `-ctk`/`-ctv`; the honest lever that exists today.

## Where it lands in cascadia

None of this belongs in the `sycl-llama` elastic wiring: cascadia spawns
the child and sets budgets; KV layout and eviction live inside the child
and the driver. If upstream llama.cpp gains per-slot KV offload, cascadia
picks it up as a flag the same way `--llama-host-layers` was added for
patch 0003. Until then the documented contract stands: `--elastic` bounds
*weights*, KV is provisioned by `-c` x `-ctk`/`-ctv` sizing, and
`--elastic-share` caps weight residency for co-tenants.
