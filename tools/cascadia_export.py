#!/usr/bin/env python3
"""Whole-model OpenVINO export (`cascadia export`) and the vision-language
routing `cascadia shard` shares with it.

`export` turns an HF repo id or a local checkpoint into one OpenVINO IR
directory with optimum-intel: `--task image-text-to-text` for a
vision-language model (VLM), `text-generation-with-past` for text. The result
is what ov-genai or OVMS serves on one box. A source that already is an
OpenVINO export is left as it is.

`shard` (export_shards.py) calls `shard_vlm` before its generic flow. A VLM
whose language model `vlm_surgery/cut_vlm_lm.py` can cut gets a real
vision tree:
  - an OpenVINO VLM export (openvino_vision_embeddings_model.xml) is cut;
  - a raw checkpoint of a type optimum-intel exports as image-text-to-text
    is exported into a temp dir first, then cut.
Any other VLM is refused with what to do instead; `--text-only` keeps the old
behaviour (the text tower alone, which serves text only).

Both verbs write cascadia-export.json beside their output (`write_sidecar`):
the model id, source repo and revision, stage count, layers, quantization,
size and input modalities, as a publisher's cascadia.json carries them.
Nodes do not read it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

SIDECAR = "cascadia-export.json"

# The vision tower's IR in an optimum-intel VLM export; its presence is what
# makes a directory an OpenVINO VLM export.
VISION_IR = "openvino_vision_embeddings_model.xml"
# Language-model IRs: a VLM export (openvino_language_model.xml) or a text
# export (openvino_model.xml).
OV_MODEL_IRS = ("openvino_language_model.xml", "openvino_model.xml")

VLM_TASK = "image-text-to-text"
TEXT_TASK = "text-generation-with-past"

# Wrapper model_types of vision-language checkpoints that carry no
# vision_config (or name it differently), so is_vlm_config also knows them.
VLM_MODEL_TYPES = (
    "llava", "llava_next", "llava_next_video", "llava_onevision", "mistral3", "pixtral",
    "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe", "gemma3", "gemma3n", "paligemma",
    "idefics", "idefics2", "idefics3", "smolvlm", "mllama", "phi3_v", "phi4mm",
    "internvl_chat", "minicpmv", "minicpmo",
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg: str) -> None:
    print(msg, flush=True)


def is_vlm_config(cfg: dict) -> bool:
    """True when config.json describes a vision-language model."""
    if cfg.get("vision_config") is not None or cfg.get("img_processor") is not None:
        return True
    return (cfg.get("model_type") or "").lower() in VLM_MODEL_TYPES


def cut_family(cfg: dict) -> str | None:
    """The cut_vlm_lm --family for this config.json, or None."""
    sys.path.insert(0, _HERE)
    sys.path.insert(0, os.path.join(_HERE, "vlm_surgery"))
    from cut_vlm_lm import family_for_config

    return family_for_config(cfg)


def cut_families() -> list[str]:
    sys.path.insert(0, _HERE)
    sys.path.insert(0, os.path.join(_HERE, "vlm_surgery"))
    from cut_vlm_lm import FAMILIES

    return sorted(FAMILIES)


def install_line() -> str:
    """`python -m pip install <pins>` from the requirements.txt the CLI writes
    beside this script (or tools/requirements.txt in the repo)."""
    path = os.path.join(_HERE, "requirements.txt")
    pins = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    pins.append(f'"{line}"')
    return f"{sys.executable} -m pip install {' '.join(pins) or 'optimum-intel'}"


def source_identity(model: str, revision: str | None = None) -> dict:
    """model_id, repo and revision for a source. An HF id gives its base name
    and the resolved commit sha; a local directory gives its own name and no
    repo or revision."""
    if os.path.isdir(model):
        return {"model_id": os.path.basename(os.path.abspath(model)),
                "repo": None, "revision": None}
    from huggingface_hub import HfApi

    sha = HfApi().model_info(model, revision=revision).sha
    return {"model_id": model.rstrip("/").rsplit("/", 1)[-1], "repo": model, "revision": sha}


def source_files(model: str, revision: str | None = None) -> list[str]:
    """The source's top-level file names, without downloading it."""
    if os.path.isdir(model):
        return os.listdir(model)
    from huggingface_hub import HfApi

    return HfApi().list_repo_files(model, revision=revision)


def is_openvino_export(files) -> bool:
    return any(f in files for f in OV_MODEL_IRS)


def is_openvino_vlm_export(files) -> bool:
    return VISION_IR in files and "openvino_language_model.xml" in files


def optimum_exports_vlm(model_type: str) -> bool | str | None:
    """Whether optimum-intel exports `model_type` as image-text-to-text with
    the installed transformers: True, False, a reason string when the
    exporter's transformers bounds exclude the installed one, or None when
    optimum-intel is not installed."""
    try:
        from optimum.exporters.tasks import TasksManager
        import optimum.exporters.openvino  # noqa: F401  (registers the openvino exporter)
    except Exception:
        return None
    for mt in (model_type, model_type.replace("_", "-")):
        try:
            tasks = TasksManager.get_supported_tasks_for_model_type(
                mt, exporter="openvino", library_name="transformers")
        except KeyError:
            continue
        if VLM_TASK not in tasks:
            return False
        return transformers_bounds_refusal(tasks[VLM_TASK]) or True
    return False


def transformers_bounds_refusal(constructor) -> str | None:
    """optimum-intel's export configs carry MIN/MAX_TRANSFORMERS_VERSION and
    fail only after loading the model; say so up front."""
    import transformers
    from packaging.version import Version

    cls = getattr(constructor, "func", constructor)
    have = Version(transformers.__version__)
    lo = getattr(cls, "MIN_TRANSFORMERS_VERSION", None)
    hi = getattr(cls, "MAX_TRANSFORMERS_VERSION", None)
    if lo and have < Version(str(lo)):
        return f"its optimum-intel exporter needs transformers >= {lo} (installed: {have})"
    if hi and have > Version(str(hi)):
        return f"its optimum-intel exporter needs transformers <= {hi} (installed: {have})"
    return None


def raw_export_refusal(cfg: dict, supported: bool | str | None) -> str | None:
    """Why optimum-intel cannot export this raw VLM checkpoint, or None.
    `supported` is optimum_exports_vlm(model_type)."""
    mt = (cfg.get("model_type") or "").lower()
    if supported is None:
        return ("exporting a raw checkpoint needs optimum-intel, which this Python lacks. "
                f"Install the exporter pins: {install_line()}")
    if supported is False:
        return f"optimum-intel cannot export model_type {mt!r} as {VLM_TASK}"
    if isinstance(supported, str):
        return f"optimum-intel cannot export model_type {mt!r} here: {supported}"
    if cfg.get("auto_map"):
        # Remote modelling code would run on this machine; the OpenVINO
        # organisation's ready exports avoid it.
        return (f"model_type {mt!r} ships its own modelling code (auto_map), which cascadia "
                "does not run. Use a ready OpenVINO export of it (e.g. from the OpenVINO "
                "organisation on Hugging Face) as --model")
    return None


def download(repo: str, revision: str) -> str:
    """The whole repo at `revision`, in the exporter's model cache."""
    from huggingface_hub import snapshot_download

    local = os.path.join(os.path.expanduser("~/.cache/cascadia/models"), repo.replace("/", "--"))
    log(f"Downloading {repo}@{revision[:12]} -> {local}")
    snapshot_download(repo_id=repo, revision=revision, local_dir=local, max_workers=8)
    return local


def optimum_export(model: str, out_dir: str, task: str, weight_format: str,
                   revision: str | None = None, sym: bool = False) -> None:
    """optimum-cli export openvino, run with this interpreter. A hub id with a known
    revision is downloaded at that revision first: optimum-cli has no --revision."""
    if revision and not os.path.isdir(model):
        model = download(model, revision)
    cmd = [sys.executable, "-m", "optimum.commands.optimum_cli", "export", "openvino",
           "--model", model, "--task", task, "--weight-format", weight_format]
    if sym:
        cmd.append("--sym")
    cmd.append(out_dir)
    log("Running: " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"optimum-cli export exited with {rc}; see the output above")


def tree_facts(out_dir: str) -> dict | None:
    """stages, numLayers and whether the tree carries vision, read from what
    the exporter wrote: pipeline_config.json (ov-runtime, gemma4 trees) or
    manifest.json (qwen35 trees). None when neither is there."""
    for name in ("pipeline_config.json", "manifest.json"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                cfg = json.load(f)
            stages = cfg.get("num_stages")
            if stages is None:
                stages = len(cfg.get("stages") or [])
            return {"stages": stages, "numLayers": cfg.get("num_layers"),
                    "vision": "vision" in cfg, "quant": cfg.get("quantization")}
    return None


def source_quant(model_dir: str | None) -> str | None:
    """The weight format an OpenVINO export was compressed to, from the
    openvino_config.json optimum-intel writes beside it."""
    if not model_dir:
        return None
    path = os.path.join(model_dir, "openvino_config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("dtype"):
        return cfg["dtype"]
    q = cfg.get("quantization_config") or {}
    bits = q.get("bits") or ((q.get("quantization_configs") or {}).get("lm_model") or {}).get("bits")
    return {4: "int4", 8: "int8"}.get(bits)


def dir_size(out_dir: str) -> int:
    total = 0
    for root, _, files in os.walk(out_dir):
        for name in files:
            if root == out_dir and name == SIDECAR:
                continue
            total += os.path.getsize(os.path.join(root, name))
    return total


def modalities(vlm: bool) -> list[str]:
    # Every VLM tree takes video when its node has ffmpeg: natively on Qwen VL
    # and qwen35, as labelled frames elsewhere (docs/VLM.md).
    return ["text", "image", "video"] if vlm else ["text"]


def write_sidecar(out_dir: str, identity: dict, *, vlm: bool, quant: str | None,
                  num_layers: int | None, stages: int | None = None) -> None:
    """cascadia-export.json, with no timestamps, so two machines exporting the
    same source write the same bytes."""
    doc = {
        "model_id": identity["model_id"],
        "source": {"repo": identity["repo"], "revision": identity["revision"]},
    }
    if stages is not None:
        doc["stages"] = stages
    doc.update({
        "numLayers": num_layers,
        "quant": quant,
        "sizeBytes": dir_size(out_dir),
        "inputModalities": modalities(vlm),
    })
    with open(os.path.join(out_dir, SIDECAR), "w", newline="\n") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    log(f"wrote {os.path.join(out_dir, SIDECAR)}")


def text_layers(cfg: dict) -> int | None:
    for key in ("text_config", "llm_config"):
        sub = cfg.get(key) or {}
        if "num_hidden_layers" in sub:
            return sub["num_hidden_layers"]
    return cfg.get("num_hidden_layers")


def write_shard_sidecar(model: str, out_dir: str, identity: dict,
                        source_dir: str | None = None) -> None:
    """The sidecar for a finished `shard` tree, from the tree itself."""
    facts = tree_facts(out_dir)
    if facts is None:
        log(f"  warning: {out_dir} has neither pipeline_config.json nor manifest.json; "
            f"no {SIDECAR} written")
        return
    local = source_dir or (model if os.path.isdir(model) else None)
    write_sidecar(out_dir, identity, vlm=facts["vision"],
                  quant=facts["quant"] or source_quant(local),
                  num_layers=facts["numLayers"], stages=facts["stages"])


# --- shard ------------------------------------------------------------------

def refuse_vlm(mt: str, why: str) -> None:
    raise SystemExit(
        f"{mt!r} is a vision-language model and `cascadia shard` cannot make a vision tree "
        f"for it: {why}.\n"
        f"  - `cascadia export` makes a whole-model OpenVINO export (ov-genai, OVMS) if "
        f"optimum-intel supports it;\n"
        f"  - `cascadia shard --text-only` shards the text tower alone (it then serves text "
        f"only).")


def shard_vlm(args, raw_cfg: dict) -> dict | None:
    """Route a VLM for `cascadia shard`. Returns the source's identity and
    the local export dir when it cut a vision tree, None when the source is
    not a VLM or --text-only asked for the text tower. Refuses (SystemExit) a
    VLM it cannot cut."""
    if args.text_only or not is_vlm_config(raw_cfg):
        return None
    mt = (raw_cfg.get("model_type") or "").lower()
    family = cut_family(raw_cfg)
    if family is None:
        refuse_vlm(mt, f"no stage cutter handles it (cut_vlm_lm families: "
                       f"{', '.join(cut_families())})")
    if args.target == "npu":
        raise SystemExit("a VLM tree is stateful and dynamic-shape; --target npu is not "
                         "supported for it. Use --target cpu-gpu.")
    if args.layer_split:
        raise SystemExit("--layer-split is not supported for VLM trees (cut_vlm_lm splits "
                         "at even decoder-layer ranges); omit it.")

    revision = getattr(args, "revision", None)
    files = source_files(args.model, revision)
    identity = getattr(args, "identity", None) or source_identity(args.model, revision)
    if is_openvino_vlm_export(files):
        log(f"Detected an OpenVINO {family} VLM export - cutting it with "
            f"vlm_surgery/cut_vlm_lm.py (--quantization is ignored: stages keep the "
            f"export's weights)")
        src = args.model if os.path.isdir(args.model) else download(args.model,
                                                                     identity["revision"])
        _cut(src, args, family, identity["model_id"])
        return {"identity": identity, "source_dir": src}

    why = raw_export_refusal(raw_cfg, optimum_exports_vlm(mt))
    if why:
        refuse_vlm(mt, why)
    with tempfile.TemporaryDirectory(prefix="cascadia-vlm-export-") as tmp:
        log(f"Detected a raw {family} checkpoint - exporting it with optimum-intel, "
            f"then cutting it")
        wf, sym = {"fp16": ("fp16", False), "int8": ("int8", False),
                   "int4": ("int4", True), "int4_asym": ("int4", False)}[args.quantization]
        optimum_export(args.model, tmp, VLM_TASK, wf, revision=identity["revision"], sym=sym)
        _cut(tmp, args, family, identity["model_id"])
        quant = source_quant(tmp)
    return {"identity": identity, "source_dir": None, "quant": quant}


def _cut(src: str, args, family: str, model_id: str) -> None:
    sys.path.insert(0, _HERE)
    sys.path.insert(0, os.path.join(_HERE, "vlm_surgery"))
    import cut_vlm_lm

    cut_vlm_lm.run(src, args.output_dir, args.num_stages, family,
                   cut_vlm_lm.g.DEFAULT_LAYERNORM_SUFFIX, args.stage, model_id=model_id)


# --- export -----------------------------------------------------------------

def export(model: str, out_dir: str, weight_format: str = "int4",
           revision: str | None = None) -> None:
    files = source_files(model, revision)
    if is_openvino_export(files):
        log(f"{model} is already an OpenVINO export; serve it as it is (nothing written).")
        return
    sys.path.insert(0, _HERE)
    from export_shards import fetch_config_json

    with open(fetch_config_json(model)) as f:
        cfg = json.load(f)
    mt = (cfg.get("model_type") or "").lower()
    vlm = is_vlm_config(cfg)
    if any(t in mt for t in ("qwen3_5", "qwen3.5")):
        raise SystemExit(f"model_type {mt!r}: no whole-model export is defined for the "
                         "Qwen3.5 family; use `cascadia shard` on its OpenVINO export.")
    supported = optimum_exports_vlm(mt)
    if supported is None:
        raise SystemExit("`cascadia export` needs optimum-intel, which this Python lacks. "
                         f"Install the exporter pins: {install_line()}")
    if vlm:
        why = raw_export_refusal(cfg, supported)
        if why:
            raise SystemExit(f"{mt!r} is a vision-language model: {why}.")
    identity = source_identity(model, revision)
    task = VLM_TASK if vlm else TEXT_TASK
    log(f"Exporting {model} ({mt}, {'vision-language' if vlm else 'text'}) with "
        f"--task {task} --weight-format {weight_format}")
    optimum_export(model, out_dir, task, weight_format, revision=identity["revision"])
    write_sidecar(out_dir, identity, vlm=vlm,
                  quant=source_quant(out_dir) or weight_format,
                  num_layers=text_layers(cfg))
    log(f"EXPORT COMPLETE -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Whole-model OpenVINO export (cascadia export)")
    ap.add_argument("--model", required=True, help="HF repo id or local checkpoint directory")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--weight-format", default="int4",
                    help="optimum-cli --weight-format (default int4)")
    ap.add_argument("--revision", default=None, help="HF revision (default: the hub's current)")
    args = ap.parse_args()
    export(args.model, args.output_dir, args.weight_format, args.revision)


if __name__ == "__main__":
    main()
