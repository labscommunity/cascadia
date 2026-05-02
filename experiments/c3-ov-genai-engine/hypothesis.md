# c3: ov-genai engine wired into tahoma

**Campaign:** c3-ov-genai-engine

**Hypothesis:** running the new `ov-genai` tahoma engine via `tahoma worker --engine ov-genai` should reproduce c1-1's 96 tok/s on alpha — proving the discovery is consumable through tahoma's normal CLI without bypassing the framework.

**Falsification:** if the tahoma path adds >10% overhead vs raw c1-1 (i.e. <87 tok/s on alpha), there's wrapping cost that needs to be eliminated.

**Predicted outcome:** 90-100 tok/s on alpha (within 1-2 tok/s of c1-1).
