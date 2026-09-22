# Offline study of Inkling's shipped MTP head and of the logit lens (experiment 033)

Verdict: `../../experiments/033_mtp_offline_study/verdict.md`. This first run used a hidden-state dump from another
machine's CPU path and touched the fleet not at all. **Repeat it on the fleet:** the states the heads will see are the
fleet's (f16 fused experts, int8 attention; its text departs from the CPU path's after 7-70 tokens). Get them with the
fleet state capture (queue item): final residuals from the last rank, rank-boundary residuals from any rank, fetched
through the entry box's relay; then steps 2 and 3 below run unchanged on the build host. The dump tool of step 1 remains
for what only a single-process CPU reference can give: any tensor of any layer, bit-exact.

1. Dump, reference path only (a machine that holds the whole int4 export resident; 033 ran it on a 1.5 TB Mac Pro, 28 threads):
   `cargo run --release -p cascadia-engine-sparse-moe --example inkling_spec_dump -- --export <export> --prompts prompts.jsonl
   --out-dir dump --generate 160 --experts mmap --max-seq 1024` (`ulimit -n 65536`, `CASCADIA_INKLING_PIN_EXPERTS=1`).
   One safetensors per prompt: tokens, `embed_out`, `final_out` (layer 65, before the final norm, every position),
   `argmax`, `layer{5,11,..,59}_out` at generated positions. Resumable (skips existing files). `prompts.jsonl` = one
   `{"i":..,"ids":[..]}` per line, rendered with the export's chat template exactly as the API does.
2. `mtp_score.py` (PyTorch, f32, CPU is enough: ~30 s per sequence): the MTP forward from `mtp.safetensors` +
   the export's embed/head, protocols A/B/C and the teeth variants; `mtp_prodsim.py`: modules 1-7 only at anchors;
   `mtp_tables.py`, `final_tables.py`: the tables in `results/`.
3. `lens_score.py`, `lens_tuned.py`: raw and ridge-fitted logit lens per rank boundary; `corpus_compare.py`: where the
   CPU path's text departs from the fleet's.

`results/`: `tables_mtp.txt` (a1..a8 by protocol and family, teeth, vocabulary prefix), `tables_final.txt` (lens),
`tuned.log`, `corpus_compare.txt`. The per-position JSON and the 1.9 GB dump stayed on the machines that made them.
