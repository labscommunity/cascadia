# e9 — paged-attention re-export (FAILED)

**Hypothesis:** OpenVINO's `SDPAToPagedAttention` transformation, applied at export time via `paged_attention_transformation()` from `openvino._offline_transformations`, would convert v5 stage IRs into PA-shaped models. PA-shaped models engage the OV GPU plugin's U8 KV cache + dynamic-quant + indirect-cache optimizations that drive LLMPipeline's per-stage compute speedup. Expected gain: 30-50% per-stage, projecting to ~22 tok/s distributed.

**Attempt 1**: ran `rainier/scripts/export_cached_shards_v5.py` with `V5_MODE=paged_attention`. Failed at the transformation:

```
paged_attention_transformation FAILED: Check '!model->get_variables().empty()' failed
at src\core\src\pass\sdpa_to_paged_attention.cpp:55:
Model is supposed to be stateful, cannot perform the SDPAToPagedAttention transformation.
```

The OV 2026.1 PA transformation requires the model to be stateful FIRST (have ReadValue/Assign nodes). The v5 export script branched on `apply_paged` and SKIPPED `apply_make_stateful_transformation` for the PA path — bug.

**Attempt 2**: patched the export script (commit upstream) to apply make_stateful unconditionally, then PA on top. Re-exported. Failed at a different check:

```
paged_attention_transformation FAILED: Check 'unregistered_parameters.str().empty()' failed
at src\core\src\model.cpp:266:
Model references undeclared parameters: opset1::Parameter attention_mask () -> (i64[?,?])
```

The PA transformation absorbs `attention_mask` into the PA op's internal masking, but our trace-based stateful model still has an explicit `attention_mask` Parameter. The transform tries to delete it from the parameter list; the validator then complains the graph still references it.

The OV docs hint at the workaround: "For proper conversion run: optimum-cli export openvino --task text-generation-with-past instead of --task text-generation". optimum-cli's exporter produces an IR shape that PA accepts — but optimum-cli is **monolithic-only**; it cannot per-stage. There is no public path to per-stage PA-shaped IRs in OV 2026.1.

## Conclusion

- **PA re-export is dead-end on the current per-stage IR pipeline.** Recovering it would need either:
  - A custom torch.export path that produces an IR shape PA accepts (multi-week effort, OV-specific knowledge)
  - Forking optimum-cli's export to support per-stage (multi-week effort)
  - Applying SDPAToPagedAttention at compile time inside the engine (untested; same model-shape requirements likely apply)
- The big LLMPipeline-vs-our-IR perf gap will NOT close via runtime config or simple re-export. It needs substantial export-pipeline work — out of scope for this autolab session unless the moonshot return is overwhelming.
- **Pivot**: focus on async overlap (a contained engine change that helps high-accept workloads), per-stage detailed timing (e10), and exploration of early-exit / pseudo-head moonshots.

## Negative finding worth saving

The OV 2026.1 `paged_attention_transformation` API has a HARD dependency on `optimum-cli`-shaped IRs and on the stateful transform being applied first. This eliminates the obvious "just rerun the export with a flag" path the prior python autolab d7 hoped for. Filed in DISCOVERIES.
