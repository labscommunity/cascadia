# Offline study of Inkling's shipped MTP head and of the logit lens (experiment 033)

Verdict: `../../experiments/033_mtp_offline_study/verdict.md`. Nothing here touches the fleet.

1. Dump (any machine that holds the whole int4 export resident; it ran on a 1.5 TB Mac Pro, 28 threads):
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
