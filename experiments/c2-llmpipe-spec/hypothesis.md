# c2: LLMPipeline + speculative decoding

**Campaign:** c2-llmpipe-spec

**Hypothesis:** layering speculative decoding (`draft_model=` on LLMPipeline) on top of c1's 96 tok/s gives a further ≥1.3× speedup. Synthesis #5 says spec decode is 1.5-3× on GPU; on top of an already-fast LLMPipeline base we expect compression of the multiplier.

**Falsification:** if spec decode yields ≤1.05× over c1 on alpha (96.4 tok/s), reject — the per-step cost of running the draft model + the verify step exceeds what we save on accepted draft tokens.

**Predicted outcome:** 130-180 tok/s on alpha, 110-150 tok/s on charlie.

**Comparison baselines:**
- c1-1: alpha LLMPipeline (no spec): 96.41 tok/s
- c1-2: charlie LLMPipeline (no spec): 91.14 tok/s

**Variables to test:**
- `num_assistant_tokens` ∈ {3, 4, 5, 7} — analogous to our K from earlier
- `assistant_confidence_threshold` (the dynamic-K knob; if set, num_assistant_tokens becomes the cap)

**Risk:**
- The 1B draft model dir on alpha was created with our OLD optimum_engine.resolve_or_export path (might not be optimal for LLMPipeline's expectations).
- LLMPipeline draft_model API may have changed shape between 2026.0 and 2026.1.

**Plan:**
1. c2-1: alpha + draft, num_assistant_tokens=5, no confidence threshold. (Synthesis default.)
2. c2-2: charlie same.
3. c2-3..6: alpha sweep K ∈ {3, 4, 5, 7}.
4. c2-7..8: alpha + dynamic K via assistant_confidence_threshold ∈ {0.4, 0.6}.
