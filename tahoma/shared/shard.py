"""Shard layout types.

`ShardSpec` describes a single node's slice of a model (which layers, which
device). `ShardPlan` is the cluster-wide split: one `ShardSpec` per stage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShardSpec:
    """A single rank's slice of a model — what one Engine instance will run.

    For pure pipeline parallelism (the default) ``tp_size == 1`` and the spec
    describes a contiguous range of layers. When ``tp_size > 1`` the same
    layer range is held by ``tp_size`` peers cooperatively, each owning a
    1/N slice of every weight matrix; engines that opt into TP must perform
    an all-reduce after each attention output and each MLP output. See
    ``tahoma.parallel`` for the collective primitive.
    """

    model_id: str
    layer_start: int     # inclusive
    layer_end: int       # exclusive
    total_layers: int
    device: str          # OpenVINO device hint: "CPU", "GPU", "NPU"
    is_first_stage: bool
    is_last_stage: bool
    # Tensor parallel layout (default = pipeline-only, no TP).
    tp_size: int = 1
    tp_rank: int = 0

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start

    @property
    def is_tp(self) -> bool:
        return self.tp_size > 1


@dataclass(frozen=True)
class ShardPlan:
    """Cluster-wide pipeline layout: one ShardSpec per stage."""

    model_id: str
    total_layers: int
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    stages: tuple[ShardSpec, ...]

    @property
    def total_stages(self) -> int:
        return len(self.stages)

    @classmethod
    def uniform(
        cls,
        *,
        model_id: str,
        total_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        vocab_size: int,
        num_stages: int,
        devices: list[str] | None = None,
    ) -> ShardPlan:
        """Even split of `total_layers` across `num_stages`.

        Trailing layers (when `total_layers % num_stages != 0`) go to the
        last stage. `devices` is a per-stage device hint; defaults to all CPU.
        """
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1 (got {num_stages})")
        if num_stages > total_layers:
            raise ValueError(
                f"num_stages ({num_stages}) exceeds total_layers ({total_layers})"
            )

        if devices is None:
            devices = ["CPU"] * num_stages
        elif len(devices) != num_stages:
            raise ValueError(
                f"devices ({len(devices)}) must match num_stages ({num_stages})"
            )

        per_stage = total_layers // num_stages
        cursor = 0
        stages: list[ShardSpec] = []
        for i in range(num_stages):
            count = per_stage if i < num_stages - 1 else total_layers - cursor
            stages.append(
                ShardSpec(
                    model_id=model_id,
                    layer_start=cursor,
                    layer_end=cursor + count,
                    total_layers=total_layers,
                    device=devices[i],
                    is_first_stage=(i == 0),
                    is_last_stage=(i == num_stages - 1),
                )
            )
            cursor += count

        return cls(
            model_id=model_id,
            total_layers=total_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            stages=tuple(stages),
        )

    @classmethod
    def from_hf_model_id(
        cls,
        model_id: str,
        num_stages: int,
        devices: list[str] | None = None,
    ) -> ShardPlan:
        """Pull HF config and create a uniform plan. Convenience wrapper."""
        from transformers import AutoConfig  # lazy import — only when used

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)
        return cls.uniform(
            model_id=model_id,
            total_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            num_attention_heads=text_config.num_attention_heads,
            vocab_size=text_config.vocab_size,
            num_stages=num_stages,
            devices=devices,
        )
