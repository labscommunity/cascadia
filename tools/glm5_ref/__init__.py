"""GLM-5.2 (`glm5`) CPU reference — the single bit-exact ground truth for the
Rust shell in `crates/cascadia-engine-sparse-moe/src/glm/`.

Mirrors the `tools/deepseek_v4_ref/` pattern: pure-torch/CPU primitives that the
exporter and the Rust shell are both validated against. GLM-5.2 is DeepSeek-V3
-style MLA + DSA, so this reference is simpler than the V4 one (no
Hyper-Connections, no learned KV compression, sigmoid routing, no YaRN).
"""
