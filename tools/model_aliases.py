"""Short aliases for commonly-tested models.

The exporter (`tools/export_shards.py`) resolves `--model <name>` through
this map first; if a match is found, the canonical HuggingFace repo id is
used in its place. This saves typing for the test recipes and gives a
stable spelling we can quote in docs that survives upstream renames (e.g.
when a model is republished under a different org). (When run via
`cascadia shard`, this module must be bundled alongside the exporter —
wired up with the dedicated-exporter bundling in #48.)

Adding a new alias is a one-line change. Aliases must be lowercase,
hyphen-separated, and unique across the map.
"""

from __future__ import annotations

# Each entry: short alias -> canonical HuggingFace repo id.
ALIASES: dict[str, str] = {
    # DeepSeek R1 Distills (Jan 2025) — fine-tunes of Qwen2.5 / Llama 3.1
    # base models with reasoning chain-of-thought. Cascadia routes these
    # through the existing qwen2 / llama paths; see
    # docs/architectures/r1-distill.md.
    "r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "r1-distill-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "r1-distill-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "r1-distill-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",

    # Llama 3.x family — the reference path. Use unsloth's re-uploads
    # which are not gated.
    "llama-3.1-8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "llama-3.2-1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
    "llama-3.3-70b": "unsloth/Llama-3.3-70B-Instruct",

    # Mistral 7B and 12B (NeMo) — text path supported via the mistral
    # branch in detect_architecture.
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-nemo-12b": "mistralai/Mistral-Nemo-Instruct-2407",

    # Qwen 2.5 / Qwen 3 dense.
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",

    # Qwen 3.5 family — hybrid Gated DeltaNet: 3.5/3.6-35B-A3B are MoE
    # (qwen3_5_moe), 3.8-27B is dense (qwen3_5). Single-stage `ov-genai`
    # (OV >= 2026.2), or staged via `cascadia shard` + `--engine qwen35`
    # (IR surgery on the official int4 IR, #77). The surgery consumes the
    # *OpenVINO IR* repo, not safetensors, so the staged aliases point at
    # Intel's published int4 exports.
    "qwen3.5-35b": "Qwen/Qwen3.5-35B-A3B",
    "qwen3.6-35b": "Qwen/Qwen3.6-35B-A3B",
    "qwen3.6-35b-ov": "OpenVINO/Qwen3.6-35B-A3B-int4-ov",
    "qwen3.8-27b": "OpenVINO/Qwen3.8-27B-int4-ov",

    # Phi-3 / Phi-4 — partial rotary handled by export_shards.py (#69).
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-4": "microsoft/phi-4",
    "phi-4-mini": "microsoft/Phi-4-mini-instruct",

    # Gemma 4 — dispatched to the dedicated tools/export_gemma4.py exporter
    # (#48). google/gemma-4-* is gated; use unsloth's re-uploads.
    "gemma-4-e2b": "unsloth/gemma-4-E2B-it",
    "gemma-4-e4b": "unsloth/gemma-4-E4B-it",
    "gemma-4-31b": "unsloth/gemma-4-31B-it",
}


def resolve(name: str) -> str:
    """Resolve a short alias to the canonical HF repo id.

    If ``name`` contains a ``/`` it is assumed to already be a canonical
    HF id (or a local path) and is returned as-is. Otherwise the alias
    map is consulted; on miss, the original name is returned with no
    error (so users get the natural "model not found on HF" error from
    huggingface_hub rather than a cascadia-specific one).
    """
    if "/" in name or "\\" in name or name.startswith("."):
        return name
    return ALIASES.get(name.lower(), name)
