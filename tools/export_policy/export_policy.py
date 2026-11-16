"""Driver for `cascadia export-policy`: a LeRobot SmolVLA or pi0.5 checkpoint to a `policy_package`.

The CLI embeds this file, writes it to a temp dir and runs it with the Python of the uv environment it
manages (tools/export_policy/requirements.txt). Two subcommands:

  preflight --model ID --result FILE
      Reads only metadata (the repo listing, config.json, the processor JSON files) and writes a JSON
      verdict: policy type, licence, gated-tokenizer access, normalization stats. The CLI decides
      whether to refuse. Needs only huggingface_hub.
  export --model ID --preflight FILE --out DIR [--rtc] [--stats FILE] [--verify]
      Loads the policy in float32, exports it with physical-ai-studio's OpenVINO exporter into a
      float32 IR, and saves the package's IR from it with fp16 weights.

Heavy modules are imported inside the functions that need them, so the helpers above `main` import
with the standard library alone (tools/tests/test_export_policy.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

SUPPORTED = ("smolvla", "pi05")
# Checkpoints whose tokenizer comes from a gated repository, with the terms that gate it.
GATED_TOKENIZER = {"pi05": "google/paligemma-3b-pt-224"}
STAT_KEYS = ("mean", "std", "q01", "q99", "min", "max")
MODE_KEYS = {"mean_std": ("mean", "std"), "quantiles": ("q01", "q99"), "min_max": ("min", "max")}
# Manifest stats entries fed from the --stats file's "state" or "action" block.
STATS_SOURCE = {"state": "state", "action": "action", "prev_chunk_left_over": "action"}


def log(line: str) -> None:
    print(line, flush=True)


def _step(doc: dict | None, registry_name: str) -> dict | None:
    for step in (doc or {}).get("steps", []):
        if step.get("registry_name") == registry_name:
            return step
    return None


def stats_problem(pre: dict | None, post: dict | None, files: set[str] | None) -> str | None:
    """Why the checkpoint's normalization statistics are missing, or None when they are present.

    `files` is the repository listing, or None when it could not be read (offline).
    """
    if pre is None:
        return (
            "the checkpoint has no policy_preprocessor.json (an older LeRobot format), so the "
            "exporter cannot read its normalization statistics"
        )
    checks = (
        (pre, "policy_preprocessor.json", "normalizer_processor", "STATE"),
        (post, "policy_postprocessor.json", "unnormalizer_processor", "ACTION"),
    )
    for doc, name, registry_name, ftype in checks:
        if doc is None:
            return f"the checkpoint has no {name}"
        step = _step(doc, registry_name)
        if step is None:
            return f"{name} has no {registry_name} step"
        config = step.get("config", {})
        if config.get("norm_map", {}).get(ftype, "IDENTITY") == "IDENTITY":
            continue
        if not any(f.get("type") == ftype for f in config.get("features", {}).values()):
            return f"{name}: {registry_name} lists no {ftype} feature (its features are empty)"
        state_file = step.get("state_file")
        if not state_file:
            return f"{name}: {registry_name} names no state_file"
        if files is not None and state_file not in files:
            return f"{name} names {state_file}, which is not in the repository"
    return None


def license_files(files: set[str] | None) -> list[str]:
    """Licence files at the repository root, copied into the package as they are."""
    prefixes = ("license", "licence", "copying")
    return sorted(f for f in files or () if "/" not in f and f.lower().startswith(prefixes))


def load_stats(path: Path) -> dict:
    """The --stats file: {"state": {...}, "action": {...}}, each with mean/std, q01/q99 or min/max."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a JSON object with \"state\" and \"action\"")
    if "observation.state" in doc and "state" not in doc:
        doc["state"] = doc.pop("observation.state")
    for block in ("state", "action"):
        entry = doc.get(block)
        if not isinstance(entry, dict) or not any(k in entry for k in STAT_KEYS):
            raise ValueError(f"{path}: \"{block}\" needs one of {', '.join(STAT_KEYS)}")
        for key in STAT_KEYS:
            values = entry.get(key)
            if values is None:
                continue
            if not isinstance(values, list) or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in values
            ):
                raise ValueError(f"{path}: {block}.{key} must be a list of finite numbers")
    return doc


def apply_stats(manifest: dict, stats: dict) -> list[str]:
    """Write the --stats values into every normalize/denormalize step. Returns what was replaced."""
    model = manifest.get("model", {})
    done = []
    for step in model.get("preprocessors", []) + model.get("postprocessors", []):
        if step.get("type") not in ("normalize", "denormalize"):
            continue
        mode = step.get("mode")
        for name, entry in step.get("stats", {}).items():
            source = STATS_SOURCE.get(name)
            if source is None:
                raise ValueError(f"{step['type']}: no --stats block feeds the {name!r} entry")
            width = (entry.get("shape") or [0])[-1]
            supplied = stats[source]
            for key in STAT_KEYS:
                if key not in supplied:
                    continue
                if len(supplied[key]) != width:
                    raise ValueError(
                        f"{source}.{key} has {len(supplied[key])} values; the package's {name} "
                        f"has {width}"
                    )
                entry[key] = [float(v) for v in supplied[key]]
            missing = [k for k in MODE_KEYS.get(mode, ()) if k not in supplied]
            if missing:
                raise ValueError(
                    f"{step['type']} uses mode {mode}: the {source} block needs {', '.join(missing)}"
                )
            done.append(f"{step['type']}.{name}")
    if not done:
        raise ValueError("the package has no normalize or denormalize step to take --stats")
    return done


def _read_json(model: str, filename: str) -> dict | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        return json.loads(Path(hf_hub_download(model, filename)).read_text(encoding="utf-8"))
    except (EntryNotFoundError, LocalEntryNotFoundError):
        return None


def _gated_access(repo: str) -> tuple[bool, str]:
    from huggingface_hub import auth_check, try_to_load_from_cache
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    if os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on"):
        cached = try_to_load_from_cache(repo, "tokenizer_config.json")
        if isinstance(cached, str):
            return True, "found in the Hugging Face cache (offline)"
        return False, "offline, and its tokenizer is not in the Hugging Face cache"
    try:
        auth_check(repo)
        return True, "access granted"
    except GatedRepoError:
        return False, "gated: the current Hugging Face login (if any) has no access"
    except RepositoryNotFoundError:
        return False, "not visible without logging in to Hugging Face"
    except HfHubHTTPError as e:
        return False, f"access check failed: {e.__class__.__name__}"


def preflight(model: str) -> dict:
    from huggingface_hub import model_info
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    out: dict = {"model": model, "error": None, "gated": None, "stats": None}
    files = None
    try:
        info = model_info(model)
        files = {s.rfilename for s in info.siblings or []}
        out["revision"] = info.sha
        card = info.card_data
        out["license"] = (card.get("license") if card else None) or None
    except RepositoryNotFoundError:
        out["error"] = f"{model}: no such model on Hugging Face (or it is private and you are not logged in)"
        return out
    except (HfHubHTTPError, OSError) as e:
        if os.environ.get("HF_HUB_OFFLINE", "").lower() not in ("1", "true", "yes", "on"):
            out["error"] = f"{model}: could not read the model listing ({e.__class__.__name__})"
            return out
        out["revision"], out["license"] = None, None
    out["license_files"] = license_files(files)

    config = _read_json(model, "config.json")
    if config is None:
        out["error"] = f"{model} has no config.json; is it a LeRobot policy checkpoint?"
        return out
    kind = config.get("type")
    out["policy_type"] = kind
    out["chunk_size"] = config.get("chunk_size")
    out["n_action_steps"] = config.get("n_action_steps")
    if kind not in SUPPORTED:
        return out
    gated = GATED_TOKENIZER.get(kind)
    if gated:
        ok, detail = _gated_access(gated)
        out["gated"] = {"repo": gated, "ok": ok, "detail": detail}
    pre = _read_json(model, "policy_preprocessor.json")
    post = _read_json(model, "policy_postprocessor.json")
    problem = stats_problem(pre, post, files)
    out["stats"] = {"ok": problem is None, "detail": problem or "present"}
    return out


def _load_policy(kind: str, model: str, rtc: bool):
    if kind == "smolvla":
        from physicalai.policies import SmolVLA as cls

        policy = cls(pretrained_name_or_path=model, dtype="float32")
    else:
        from physicalai.policies import Pi05 as cls

        # n_action_steps=None keeps the checkpoint's chunk length (the default 50 rejects 15-step
        # checkpoints such as DROID's).
        policy = cls(
            pretrained_name_or_path=model,
            dtype="float32",
            use_random_input_noise=True,
            n_action_steps=None,
            gradient_checkpointing=False,
        )
    policy.eval()
    if rtc:
        # RTC serves the whole chunk: the exporter adds a trimmer when n_action_steps differs.
        object.__setattr__(policy.config, "n_action_steps", policy.config.chunk_size)
        if getattr(policy.model, "_n_action_steps", None) is not None:
            policy.model._n_action_steps = None
        policy.rtc_enabled = True
    # The export is float32 and only the saved weights are fp16: the tool's default bf16 export
    # gives all-NaN actions under bf16 inference with a padded prompt.
    orig = cls.extra_export_args.fget

    def no_fp16(self):
        args = orig(self)
        args["openvino"].compress_to_fp16 = False
        return args

    cls.extra_export_args = property(no_fp16)
    return policy


def _verify(policy, sample: dict, ref, xml: Path) -> bool:
    import numpy as np
    import openvino as ov
    import torch

    feed_by_name = {
        k: (v.float() if v.dtype in (torch.bfloat16, torch.float16) else v).numpy()
        for k, v in sample.items()
    }
    core = ov.Core()
    model = core.read_model(xml)
    names = [p.get_any_name() for p in model.inputs]
    if all(n in feed_by_name for n in names):
        feed = [feed_by_name[n] for n in names]
    else:
        feed = list(feed_by_name.values())
    runs = [("CPU", "f32")]
    if any(d.startswith("GPU") for d in core.available_devices):
        runs.append(("GPU", "f16"))
    else:
        log("VERIFY GPU f16 SKIPPED: no GPU device")
    ok = True
    for device, precision in runs:
        compiled = core.compile_model(model, device, {"INFERENCE_PRECISION_HINT": precision})
        got = next(iter(compiled(feed).values()))
        finite = bool(np.isfinite(got).all())
        diff = np.abs(got.astype(np.float64) - ref)
        log(
            f"VERIFY {device} {precision} {'OK' if finite else 'FAILED (non-finite output)'}: "
            f"max_abs={diff.max():.3e} mean_abs={diff.mean():.3e} "
            f"ref_mean_abs={np.abs(ref).mean():.4f} shape={list(got.shape)}"
        )
        ok &= finite
        del compiled
    return ok


def export(args: argparse.Namespace) -> int:
    import openvino as ov
    import torch

    pre = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    kind = pre["policy_type"]
    out = Path(args.out)
    tmp = out.with_name(out.name + ".f32-tmp")
    stats = load_stats(Path(args.stats)) if args.stats else None

    t0 = time.perf_counter()
    policy = _load_policy(kind, args.model, args.rtc)
    log(
        f"LOAD OK {time.perf_counter() - t0:.0f}s kind={kind} chunk={policy.config.chunk_size} "
        f"n_action_steps={policy.config.n_action_steps} rtc={bool(args.rtc)}"
    )
    sample = ref = None
    if args.verify:
        sample = policy._get_default_export_input_sample()
        with torch.no_grad():
            got = policy.model(sample)
        ref = (got[0] if isinstance(got, tuple) else got).double().numpy()

    t0 = time.perf_counter()
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        policy.export(str(tmp), backend="openvino")
        manifest = json.loads((tmp / "manifest.json").read_text(encoding="utf-8"))
        steps = manifest["model"].get("postprocessors", [])
        if args.rtc and any("trimmer" in str(s.get("type") or s.get("class_path")) for s in steps):
            log("EXPORT FAILED: an --rtc package must serve the whole chunk, but it has a trimmer")
            return 1
        ir = manifest["model"]["artifacts"]["openvino"]
        log(f"EXPORT OK {time.perf_counter() - t0:.0f}s ir={ir}")
        out.mkdir(parents=True, exist_ok=True)
        ir_bin = Path(ir).with_suffix(".bin").name
        for p in tmp.iterdir():
            if p.name not in (ir, ir_bin):
                shutil.copy2(p, out / p.name)
        ov.save_model(ov.Core().read_model(tmp / ir), str(out / ir), compress_to_fp16=True)
        log(f"FP16 OK bin_mb={round((out / ir_bin).stat().st_size / 1e6)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if stats is not None:
        for name in ("manifest.json", "config.json"):
            path = out / name
            if not path.is_file():
                continue
            doc = json.loads(path.read_text(encoding="utf-8"))
            if "model" not in doc:
                continue
            replaced = apply_stats(doc, stats)
            path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            log(f"STATS OK {name}: {', '.join(replaced)}")

    if pre.get("license_files"):
        from huggingface_hub import hf_hub_download

        for name in pre["license_files"]:
            shutil.copy2(hf_hub_download(args.model, name), out / name)
        log(f"LICENCE OK copied {', '.join(pre['license_files'])}")

    if args.verify:
        if not _verify(policy, sample, ref, out / ir):
            return 1
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="export_policy.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--model", required=True)
    p.add_argument("--result", required=True)
    e = sub.add_parser("export")
    e.add_argument("--model", required=True)
    e.add_argument("--preflight", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--rtc", action="store_true")
    e.add_argument("--stats")
    e.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "preflight":
        Path(args.result).write_text(json.dumps(preflight(args.model)), encoding="utf-8")
        return 0
    return export(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
