"""Shared OpenVINO plugin-property builder.

The OV ``core.compile_model(model, device, plugin_config)`` and
``LLMPipeline(model, device, **plugin_config)`` calls accept a small set
of string-keyed properties that influence kernel selection and caching.
This helper centralises construction so every engine in
``tahoma.worker.engines.openvino`` plumbs the same flags consistently.

Three properties are exposed today:

* ``CACHE_DIR`` — directory where compiled-blob artifacts are persisted.
  The big win: on the second-and-later launch of the same model on the
  same device, kernel JIT is skipped and the model loads ~62% faster.
  Applies to every ``compile_model`` call, single-stage or multi-stage.
* ``KV_CACHE_PRECISION`` — ``u8`` or ``f16`` for the GPU plugin's KV
  cache. Defaults are already optimal on Battlemage / Lunar Lake;
  exposed as a debugging knob.
* ``DYNAMIC_QUANTIZATION_GROUP_SIZE`` — group size for the GPU plugin's
  dynamic quantization. Defaults are already optimal; debugging knob.
"""

from __future__ import annotations


def build_plugin_config(
    cache_dir: str | None = None,
    kv_cache_precision: str | None = None,
    dyn_quant_group: str | None = None,
) -> dict[str, str]:
    """Build an OV plugin-config dict from optional CLI knobs.

    Returns an empty dict when no knobs are set, which is the right thing
    to pass to ``compile_model`` (it then uses plugin defaults).
    """
    cfg: dict[str, str] = {}
    if cache_dir:
        cfg["CACHE_DIR"] = cache_dir
    if kv_cache_precision:
        cfg["KV_CACHE_PRECISION"] = kv_cache_precision
    if dyn_quant_group:
        cfg["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = dyn_quant_group
    return cfg


__all__ = ["build_plugin_config"]
