"""Weight compression for the shard exporters.

An export either delivers the quantization that was asked for or stops:
pipeline_config.json declares it and the runtime picks its path from it, so a
tree that says int4 and holds FP16 weights is worse than no tree.
"""

# Requested format -> the constant types only that format's weights can have.
_COMPRESSED_TYPES = {"int4": {"i4"}, "int4_asym": {"u4"}, "int8": {"i8", "u8"}}


def constant_types(ov_model) -> set:
    """The element types of the model's Constant nodes, e.g. {"i4", "f16"}."""
    return {
        n.get_output_element_type(0).get_type_name()
        for n in ov_model.get_ordered_ops()
        if n.get_type_name() == "Constant"
    }


def _refuse(quantization: str, why: str) -> SystemExit:
    return SystemExit(
        f"cannot export with --quantization {quantization}: {why}\n"
        "Re-run with --quantization fp16 to export uncompressed weights."
    )


def require_compressor(quantization: str) -> None:
    """Refuse up front if `quantization` needs NNCF and it can't be imported, so
    the paths that compress find out before the download and the trace."""
    if quantization not in _COMPRESSED_TYPES:
        return
    try:
        import nncf  # noqa: F401
    except ImportError as e:
        raise _refuse(quantization, f"nncf is not installed ({e})") from e


def compress_weights(ov_model, quantization: str, log):
    """Return `ov_model` compressed to `quantization` (returned as it is for an
    uncompressed format), or raise SystemExit if that cannot be done."""
    wanted = _COMPRESSED_TYPES.get(quantization)
    if wanted is None:
        return ov_model
    require_compressor(quantization)
    import nncf

    log(f"  nncf {quantization} compression...")
    try:
        if quantization == "int8":
            out = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM)
        else:
            out = nncf.compress_weights(
                ov_model,
                mode=(
                    nncf.CompressWeightsMode.INT4_SYM
                    if quantization == "int4"
                    else nncf.CompressWeightsMode.INT4_ASYM
                ),
                group_size=128,
                ratio=1.0,
                all_layers=True,
                # 128 needs a channel count it divides; models such as SmolLM2
                # (hidden size 576) get the nearest group size that does.
                advanced_parameters=nncf.AdvancedCompressionParameters(
                    group_size_fallback_mode=nncf.GroupSizeFallbackMode.ADJUST
                ),
            )
    except Exception as e:
        raise _refuse(quantization, f"nncf failed ({e})") from e
    if not constant_types(out) & wanted:
        raise _refuse(quantization, f"nncf wrote no {quantization} weights")
    log("  Quantization OK")
    return out
