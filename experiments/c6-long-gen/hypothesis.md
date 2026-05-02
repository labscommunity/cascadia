# c6: spec decode shines more with longer generations

**Hypothesis:** at 64 tokens output, spec decode gave +4% (c2). The per-spec-round overhead (draft compute + verify call) amortises across more decoded tokens for longer outputs. At 256 tokens, expect +10-20% over plain LLMPipeline.

**Falsification:** if spec decode at 256 tokens still gives ≤+5%, conclude that the LLMPipeline + 8B INT4 + Battlemage combo is at the spec-decode "no-op" point — the per-token cost is too low for spec to help.

**Comparison baselines:** rerun c1-1 at 256 tokens (plain LLMPipeline) and c2-4 at 256 tokens (spec K=10).
