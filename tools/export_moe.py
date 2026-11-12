"""Route `cascadia shard` to the sparse-MoE exporters.

These models keep one tree whatever the stage count: the sparse-moe engine picks
the split when it loads the tree, so `--num-stages` doesn't shape the export.
"""
import os
import subprocess
import sys

# config.json model_type -> the exporter that handles it.
EXPORTERS = {
    "glm_moe_dsa": "export_glm5",
    "deepseek_v4": "export_deepseek_v4",
    "minimax_m2": "export_minimax_m2",
    "inkling_mm_model": "export_inkling",
    "inkling_text": "export_inkling",
}


def exporter_for(*model_types):
    """The exporter module for the first model_type that has one, else None."""
    return next((EXPORTERS[t] for t in model_types if t in EXPORTERS), None)


def _refusal(args):
    """Why these flags can't apply to a sparse-MoE export, or None."""
    if getattr(args, "target", "cpu") == "npu":
        return "--target npu"
    if getattr(args, "stage", None) is not None:
        return "--stage"
    if getattr(args, "layer_split", None):
        return "--layer-split"
    return None


def _local_checkpoint(model, revision):
    """The checkpoint dir: `model` if it is one, else the hub repo downloaded."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    local = os.path.join(
        os.path.expanduser("~/.cache/cascadia/models"), model.replace("/", "--")
    )
    print(f"Downloading {model} -> {local}", flush=True)
    snapshot_download(repo_id=model, revision=revision, local_dir=local, max_workers=8)
    return local


def run_export(args, exporter):
    """Run `exporter` beside this file on `args`; exits on failure."""
    flag = _refusal(args)
    if flag:
        raise SystemExit(f"{flag} does not apply to sparse-MoE models ({exporter}).")
    print(
        f"Sparse-MoE model: running {exporter}. It uses its own weight format, so "
        "--quantization is ignored, and the engine splits it when loading, so one "
        "tree serves any --num-stages.",
        flush=True,
    )
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{exporter}.py")
    checkpoint = _local_checkpoint(args.model, getattr(args, "revision", None))
    rc = subprocess.call(
        [sys.executable, "-u", script, "--model", checkpoint, "--out", args.output_dir]
    )
    if rc:
        raise SystemExit(f"{exporter} exited with {rc}; see the output above")
