"""Selective safetensors loader for pipeline-parallel inference.

Loads only the assigned layer range from a HuggingFace model directory.
Memory usage is proportional to the layer count, not the full model.

Supports the common decoder-LM families: Llama, Mistral, Qwen2, Phi3, Gemma,
Gemma2, Phi, Starcoder2. Other architectures fall back to the Llama layer
classes; that works for many but not all models.

Ported from rainier (`cascadia/model/loader.py`); the Kimi-K2.5 INT4 QAT path
is intentionally dropped here — it is not yet known-good. See rainier for the
full multimodal/MoE path.
"""

from __future__ import annotations

import gc
import glob
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tahoma.shared.shard import ShardSpec

logger = logging.getLogger(__name__)


# HF model_type → (layers_attr, embed_attr, norm_attr, head_attr, rotary_attr)
def _torch_device(device: str) -> str:
    """Translate the OpenVINO device hint (CPU/GPU/NPU) to a torch device string."""
    d = device.upper()
    if d == "CPU":
        return "cpu"
    if d == "GPU":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("device=GPU requested but no torch GPU backend; falling back to CPU")
        return "cpu"
    if d == "NPU":
        raise NotImplementedError("NPU is not supported by the PyTorch loader path")
    raise ValueError(f"unknown device hint: {device!r}")


_MODEL_STRUCTURE: dict[str, tuple[str, str, str, str, str | None]] = {
    "llama":      ("model.layers", "model.embed_tokens", "model.norm", "lm_head", "model.rotary_emb"),
    "mistral":    ("model.layers", "model.embed_tokens", "model.norm", "lm_head", "model.rotary_emb"),
    "qwen2":      ("model.layers", "model.embed_tokens", "model.norm", "lm_head", "model.rotary_emb"),
    "phi3":       ("model.layers", "model.embed_tokens", "model.norm", "lm_head", "model.rotary_emb"),
    "gemma":      ("model.layers", "model.embed_tokens", "model.norm", "lm_head", "model.rotary_emb"),
    "gemma2":     ("model.layers", "model.embed_tokens", "model.norm", "lm_head", "model.rotary_emb"),
    "phi":        ("model.layers", "model.embed_tokens", "model.final_layernorm", "lm_head", None),
    "starcoder2": ("model.layers", "model.embed_tokens", "model.norm", "lm_head", None),
}


class ModelShard:
    """A single pipeline stage's slice of a transformer model.

    Loads only the assigned layer range from safetensors, plus the embedding
    (first stage) or final norm + LM head (last stage) as needed.
    """

    def __init__(self, spec: ShardSpec, model_path: str):
        self.spec = spec
        self.model_path = model_path

        self.tokenizer = None
        self._loaded = False
        self._embed: nn.Module | None = None
        self._layers: nn.ModuleList | None = None
        self._norm: nn.Module | None = None
        self._head: nn.Module | None = None
        self._rotary_emb: nn.Module | None = None
        self._config = None
        self._text_config = None
        self._model_structure: tuple[str, str, str, str, str | None] | None = None

    def load(self) -> None:
        """Load only the assigned layers from safetensors."""
        from transformers import AutoConfig, AutoTokenizer
        from safetensors import safe_open

        model_dir = Path(self.model_path)
        spec = self.spec
        logger.info(
            "Loading shard from %s (layers %d-%d, first=%s, last=%s)",
            model_dir, spec.layer_start, spec.layer_end - 1,
            spec.is_first_stage, spec.is_last_stage,
        )

        self._config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

        model_type = getattr(self._config, "model_type", "llama")
        self._model_structure = _MODEL_STRUCTURE.get(model_type)
        if self._model_structure is None:
            logger.warning("unknown model_type '%s'; falling back to llama structure", model_type)
            self._model_structure = _MODEL_STRUCTURE["llama"]

        layers_attr, embed_attr, norm_attr, head_attr, _rotary_attr = self._model_structure

        # Multimodal wrappers nest the LM config under text_config.
        self._text_config = getattr(self._config, "text_config", self._config)

        # Eager attention is required: SDPA/FlashAttention break tracing and
        # cause shape issues in the per-layer forward path.
        self._config._attn_implementation = "eager"
        self._text_config._attn_implementation = "eager"

        weight_prefixes: list[str] = [
            f"{layers_attr}.{i}." for i in range(spec.layer_start, spec.layer_end)
        ]
        if spec.is_first_stage:
            weight_prefixes.append(f"{embed_attr}.")
        if spec.is_last_stage:
            weight_prefixes.append(f"{norm_attr}.")
            weight_prefixes.append(f"{head_attr}.")

        safetensor_files = sorted(glob.glob(str(model_dir / "*.safetensors")))
        if not safetensor_files:
            raise FileNotFoundError(
                f"no .safetensors files in {model_dir}. "
                f"download with: huggingface-cli download <model_id> --local-dir {model_dir}"
            )

        state_dict: dict[str, torch.Tensor] = {}
        for sf_path in safetensor_files:
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if any(key.startswith(p) for p in weight_prefixes):
                        state_dict[key] = f.get_tensor(key)

        weight_mb = sum(t.nbytes for t in state_dict.values()) / 1e6
        logger.info("loaded %d tensors (%.0f MB)", len(state_dict), weight_mb)

        self._build_components(state_dict)

        del state_dict
        gc.collect()

        self._loaded = True
        logger.info(
            "shard ready: embed=%s, layers=%d, norm=%s, head=%s",
            self._embed is not None, len(self._layers),
            self._norm is not None, self._head is not None,
        )

    def _build_components(self, state_dict: dict[str, torch.Tensor]) -> None:
        spec = self.spec
        layer_config = self._text_config
        layers_attr, embed_attr, norm_attr, head_attr, _rotary_attr = self._model_structure
        layer_class, norm_class, rotary_class = self._get_component_classes(self._config.model_type)
        torch_dev = _torch_device(spec.device)

        # Decoder layers
        layers = nn.ModuleList()
        for i in range(spec.layer_start, spec.layer_end):
            prefix = f"{layers_attr}.{i}."
            layer_sd = {
                k.removeprefix(prefix): v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            layer = layer_class(layer_config, layer_idx=i)
            layer.load_state_dict(layer_sd, strict=False)
            layer.eval()
            layer.half()
            layer.to(torch_dev)
            layers.append(layer)
        self._layers = layers

        # Embedding (first stage only)
        if spec.is_first_stage:
            embed = nn.Embedding(layer_config.vocab_size, layer_config.hidden_size)
            embed_sd = {
                k.removeprefix(f"{embed_attr}."): v
                for k, v in state_dict.items()
                if k.startswith(f"{embed_attr}.")
            }
            embed.load_state_dict(embed_sd, strict=False)
            embed.eval()
            embed.half()
            embed.to(torch_dev)
            self._embed = embed

        # Final norm + LM head (last stage only)
        if spec.is_last_stage:
            if norm_class is not None:
                norm = norm_class(
                    layer_config.hidden_size,
                    eps=getattr(
                        layer_config, "rms_norm_eps",
                        getattr(layer_config, "layer_norm_eps", 1e-6),
                    ),
                )
            else:
                norm = nn.LayerNorm(layer_config.hidden_size)

            norm_sd = {
                k.removeprefix(f"{norm_attr}."): v
                for k, v in state_dict.items()
                if k.startswith(f"{norm_attr}.")
            }
            norm.load_state_dict(norm_sd, strict=False)
            norm.eval()
            norm.half()
            norm.to(torch_dev)
            self._norm = norm

            head = nn.Linear(layer_config.hidden_size, layer_config.vocab_size, bias=False)
            head_sd = {
                k.removeprefix(f"{head_attr}."): v
                for k, v in state_dict.items()
                if k.startswith(f"{head_attr}.")
            }
            head.load_state_dict(head_sd, strict=False)
            head.eval()
            head.half()
            head.to(torch_dev)
            self._head = head

        # Rotary embedding (no learned weights, config-derived only)
        if rotary_class is not None:
            self._rotary_emb = rotary_class(config=self._config).to(torch_dev)

    @staticmethod
    def _get_component_classes(model_type: str):
        """Return (DecoderLayer, RMSNorm, RotaryEmbedding) classes for the model_type."""
        try:
            if model_type in ("llama", "mistral"):
                from transformers.models.llama.modeling_llama import (
                    LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding,
                )
                return LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
            if model_type == "qwen2":
                from transformers.models.qwen2.modeling_qwen2 import (
                    Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding,
                )
                return Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding
            if model_type in ("gemma", "gemma2"):
                from transformers.models.gemma.modeling_gemma import (
                    GemmaDecoderLayer, GemmaRMSNorm, GemmaRotaryEmbedding,
                )
                return GemmaDecoderLayer, GemmaRMSNorm, GemmaRotaryEmbedding
            if model_type == "phi3":
                from transformers.models.phi3.modeling_phi3 import (
                    Phi3DecoderLayer, Phi3RMSNorm, Phi3RotaryEmbedding,
                )
                return Phi3DecoderLayer, Phi3RMSNorm, Phi3RotaryEmbedding
        except ImportError:
            pass

        # Fallback: llama (most common)
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding,
        )
        return LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

    def embed(self, input_ids: np.ndarray) -> np.ndarray:
        """Run the embedding layer (first stage only)."""
        if not self.spec.is_first_stage:
            raise RuntimeError("embed() is only valid on the first pipeline stage")
        if self._embed is None:
            raise RuntimeError("embedding not loaded; call load() first")
        torch_dev = _torch_device(self.spec.device)
        with torch.no_grad():
            tensor = torch.tensor(input_ids, dtype=torch.long, device=torch_dev)
            return self._embed(tensor).cpu().numpy()

    def forward_layers(
        self,
        hidden_states: np.ndarray,
        position_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run the assigned decoder layers without KV cache (recompute-only)."""
        if not self._loaded:
            raise RuntimeError("model not loaded; call load() first")

        device = _torch_device(self.spec.device)
        with torch.no_grad():
            hs = torch.tensor(hidden_states, device=device, dtype=torch.float16)
            if position_ids is not None:
                pos = torch.tensor(position_ids, device=device)
            else:
                pos = torch.arange(hs.shape[1], device=device).unsqueeze(0)

            position_embeddings = None
            if self._rotary_emb is not None:
                position_embeddings = self._rotary_emb(hs, pos)

            seq_len = hs.shape[1]
            causal_mask = torch.full(
                (1, 1, seq_len, seq_len), float("-inf"),
                dtype=hs.dtype, device=device,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)

            for layer in self._layers:
                if position_embeddings is not None:
                    out = layer(
                        hs,
                        position_embeddings=position_embeddings,
                        attention_mask=causal_mask,
                    )
                else:
                    out = layer(hs, position_ids=pos, attention_mask=causal_mask)
                hs = out[0]
                if hs.dim() == 2:
                    # Some decoder layers drop the batch dim for batch_size=1.
                    hs = hs.unsqueeze(0)

            return hs.cpu().numpy()

    def forward_layers_cached(
        self,
        hidden_states: np.ndarray,
        position_ids: np.ndarray | None = None,
        past_key_values=None,
    ):
        """Forward pass with KV cache (DynamicCache-backed)."""
        if not self._loaded:
            raise RuntimeError("model not loaded; call load() first")

        from transformers.cache_utils import DynamicCache

        device = _torch_device(self.spec.device)
        with torch.no_grad():
            hs = torch.tensor(hidden_states, device=device, dtype=torch.float16)
            seq_len = hs.shape[1]

            past_len = 0
            if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                # DynamicCache indexes by layer_idx; use the first held layer.
                first_layer_idx = (
                    self._layers[0].self_attn.layer_idx
                    if hasattr(self._layers[0], "self_attn")
                    and hasattr(self._layers[0].self_attn, "layer_idx")
                    else 0
                )
                past_len = past_key_values.get_seq_length(first_layer_idx)

            if position_ids is not None:
                pos = torch.tensor(position_ids, device=device)
            else:
                # Default span [past_len, past_len + seq_len). For prefill that's
                # [0, seq_len); for decode (past_len > 0, seq_len == 1) it's
                # [past_len, past_len + 1).
                pos = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(0)

            # transformers 5.x decoder layers require `position_embeddings`
            # (cos, sin) rather than `position_ids`. Compute via the model's
            # rotary embedding if we have one.
            position_embeddings = None
            if self._rotary_emb is not None:
                position_embeddings = self._rotary_emb(hs, pos)

            total_len = past_len + seq_len
            causal_mask = torch.full(
                (1, 1, seq_len, total_len), float("-inf"),
                dtype=hs.dtype, device=device,
            )
            causal_mask[:, :, :, :total_len] = 0
            for i in range(seq_len):
                causal_mask[:, :, i, past_len + i + 1:] = float("-inf")

            if past_key_values is None:
                past_key_values = DynamicCache()

            for layer in self._layers:
                if position_embeddings is not None:
                    out = layer(
                        hs,
                        position_embeddings=position_embeddings,
                        attention_mask=causal_mask,
                        past_key_value=past_key_values,
                        use_cache=True,
                    )
                else:
                    out = layer(
                        hs,
                        position_ids=pos,
                        attention_mask=causal_mask,
                        past_key_value=past_key_values,
                        use_cache=True,
                    )
                hs = out[0]
                if hs.dim() == 2:
                    hs = hs.unsqueeze(0)

            return hs.cpu().numpy(), past_key_values

    def lm_head(self, hidden_states: np.ndarray) -> np.ndarray:
        """Project hidden states to vocab logits (last stage only)."""
        if not self.spec.is_last_stage:
            raise RuntimeError("lm_head() is only valid on the last pipeline stage")
        if self._norm is None or self._head is None:
            raise RuntimeError("norm/head not loaded; call load() first")
        torch_dev = _torch_device(self.spec.device)
        with torch.no_grad():
            hs = torch.tensor(hidden_states, device=torch_dev, dtype=torch.float16)
            hs = self._norm(hs)
            return self._head(hs).cpu().numpy()

    @property
    def hidden_size(self) -> int | None:
        return self._config.hidden_size if self._config else None

    @property
    def num_layers(self) -> int | None:
        return self._config.num_hidden_layers if self._config else None
