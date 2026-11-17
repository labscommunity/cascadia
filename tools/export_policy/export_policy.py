"""Driver for `cascadia export-policy`: a LeRobot SmolVLA or pi0.5 checkpoint to a `policy_package`.

The CLI embeds this file, writes it to a temp dir and runs it with the Python of the uv environment it
manages (tools/export_policy/requirements.txt). Two subcommands:

  preflight --model ID|DIR --result FILE
      Reads only metadata (the repo listing, config.json, the processor JSON files) and writes a JSON
      verdict: policy type, licence, gated-tokenizer access, normalization stats. The CLI decides
      whether to refuse. Needs only huggingface_hub. A local checkpoint directory is read in place:
      revision "local", the licence from its README.md, and no gated-tokenizer check (a note says so).
  export --model ID|DIR --preflight FILE --out DIR [--rtc] [--stats FILE] [--verify]
      Loads the policy in float32 at the preflight's commit, exports it with physical-ai-studio's
      OpenVINO exporter into a float32 IR, and saves the package's IR from it with fp16 weights.
      DIR is the CLI's staging directory, renamed into place only when everything succeeded.

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
# Values that leave a dimension as it is, for padding stats narrower than the package. LeRobot
# normalizes the state at the robot's width and only then zero-pads it to max_state_dim
# (lerobot v0.6.0: processor/normalize_processor.py `_apply_transform`, then
# policies/pi05/modeling_pi05.py `pad_vector`); identity stats on the padded dimensions map a zero
# pad to zero in every mode, which is the same thing.
IDENTITY_PAD = {"mean": 0.0, "std": 1.0, "q01": -1.0, "q99": 1.0, "min": -1.0, "max": 1.0}
# The manifest's mode when a step names none (cascadia-engine-openvino's manifest default).
DEFAULT_MODE = "mean_std"
# The preflight revision of a local checkpoint directory.
LOCAL_REVISION = "local"


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
        mode = step.get("mode") or DEFAULT_MODE
        for name, entry in step.get("stats", {}).items():
            source = STATS_SOURCE.get(name)
            if source is None:
                raise ValueError(f"{step['type']}: no --stats block feeds the {name!r} entry")
            width = (entry.get("shape") or [0])[-1]
            supplied = stats[source]
            for key in STAT_KEYS:
                if key not in supplied:
                    continue
                values = [float(v) for v in supplied[key]]
                if not values or len(values) > width:
                    raise ValueError(
                        f"{source}.{key} has {len(values)} values; the package's {name} has {width}"
                    )
                entry[key] = values + [IDENTITY_PAD[key]] * (width - len(values))
            missing = [k for k in MODE_KEYS.get(mode, ()) if k not in supplied]
            if missing:
                raise ValueError(
                    f"{step['type']} uses mode {mode}: the {source} block needs {', '.join(missing)}"
                )
            done.append(f"{step['type']}.{name}")
    if not done:
        raise ValueError("the package has no normalize or denormalize step to take --stats")
    return done


def package_mode(kind: str, pre: dict | None, config: dict) -> str:
    """The normalize/denormalize mode physical-ai-studio 8f132416 writes into the manifest.

    SmolVLA always writes mean_std. pi0.5 takes the checkpoint's one non-identity mode when its
    preprocessor uses a single one (`detect_normalization_mode`), else the config's
    `normalization_mode`, else QUANTILES.
    """
    if kind == "smolvla":
        return "mean_std"
    modes = set()
    for step in (pre or {}).get("steps", []):
        norm_map = step.get("norm_map") or step.get("config", {}).get("norm_map") or {}
        modes.update(v for v in norm_map.values() if v != "IDENTITY")
    detected = next(iter(modes)) if len(modes) == 1 and modes <= {"MEAN_STD", "QUANTILES"} else None
    return (detected or config.get("normalization_mode") or "QUANTILES").lower()


def feature_width(config: dict, section: str, ftype: str) -> int | None:
    """The last dimension of the config's first `ftype` feature in `section`."""
    for feature in (config.get(section) or {}).values():
        if feature.get("type") == ftype and feature.get("shape"):
            return int(feature["shape"][-1])
    return None


def copy_package(src: Path, dst: Path, ir: str) -> None:
    """Copy the exporter's output, subdirectories included, except the IR and its weights."""
    rel = Path(ir)
    if not ir or rel.is_absolute() or rel.drive or ".." in rel.parts:
        raise ValueError(f"model.artifacts.openvino {ir!r} is not a path inside the package")
    skip = {rel, rel.with_suffix(".bin")}
    for p in sorted(src.rglob("*")):
        r = p.relative_to(src)
        if p.is_dir() or r in skip:
            continue
        (dst / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / r)


def card_license(readme: str) -> str | None:
    """The `license:` of a model card's YAML front matter, or None."""
    lines = readme.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep and key.strip() == "license" and value.strip():
            return value.strip().strip("'\"")
    return None


def escaping_link(root: Path) -> str | None:
    """The first link under a local checkpoint that resolves outside it, or None.

    The policy loader reads the checkpoint's weights and configs itself, so a link that leaves the
    directory would pull an arbitrary file of this machine into the export.
    """
    top = root.resolve()
    for p in sorted(root.rglob("*")):
        if p.is_symlink() and not p.resolve().is_relative_to(top):
            return p.relative_to(root).as_posix()
    return None


def local_files(root: Path) -> set[str]:
    """Every regular file under a local checkpoint, as repository-relative paths (no links)."""
    return {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink()
    }


def local_file(root: Path, name: str) -> Path | None:
    """`name` under `root` if it is a regular file inside it; None if absent; ValueError for a
    link or a path that leaves the directory."""
    path = root / name
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} in the local checkpoint is a link or not a regular file")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"{name} resolves outside the local checkpoint")
    return path


def _read_local_json(root: Path, filename: str) -> dict | None:
    path = local_file(root, filename)
    return json.loads(path.read_text(encoding="utf-8")) if path else None


def _read_json(model: str, filename: str, revision: str | None) -> dict | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        path = hf_hub_download(model, filename, revision=revision)
        return json.loads(Path(path).read_text(encoding="utf-8"))
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
    out: dict = {
        "model": model, "error": None, "gated": None, "stats": None, "revision": None, "notes": [],
    }
    files = None
    local = Path(model) if Path(model).is_dir() else None
    if local is not None:
        link = escaping_link(local)
        if link is not None:
            out["error"] = (
                f"{model}: {link} is a link that leaves the checkpoint directory; copy the file in "
                f"instead of linking it"
            )
            return out
        files = local_files(local)
        out["revision"] = LOCAL_REVISION
        try:
            readme = local_file(local, "README.md")
            out["license"] = card_license(readme.read_text(encoding="utf-8")) if readme else None
            out["license_files"] = license_files(files)
            return _preflight_checks(out, files, lambda name: _read_local_json(local, name), local=True)
        except ValueError as e:
            out["error"] = f"{model}: {e}"
            return out
    from huggingface_hub import model_info
    from huggingface_hub.errors import HfHubHTTPError, HFValidationError, RepositoryNotFoundError

    try:
        info = model_info(model)
        files = {s.rfilename for s in info.siblings or []}
        out["revision"] = info.sha
        card = info.card_data
        out["license"] = (card.get("license") if card else None) or None
    except HFValidationError as e:
        out["error"] = f"{model!r} is not a valid Hugging Face model id ({e})"
        return out
    except RepositoryNotFoundError:
        out["error"] = f"{model}: no such model on Hugging Face (or it is private and you are not logged in)"
        return out
    except (HfHubHTTPError, OSError) as e:
        if os.environ.get("HF_HUB_OFFLINE", "").lower() not in ("1", "true", "yes", "on"):
            out["error"] = f"{model}: could not read the model listing ({e.__class__.__name__})"
            return out
        out["revision"], out["license"] = None, None
    out["license_files"] = license_files(files)
    revision = out["revision"]
    return _preflight_checks(out, files, lambda name: _read_json(model, name, revision), local=False)


def _preflight_checks(out: dict, files: set[str] | None, read, local: bool) -> dict:
    """The checks shared by a hub and a local checkpoint; `read(name)` loads one JSON file."""
    model = out["model"]
    config = read("config.json")
    if config is None:
        out["error"] = f"{model} has no config.json; is it a LeRobot policy checkpoint?"
        return out
    kind = config.get("type")
    out["policy_type"] = kind
    out["chunk_size"] = config.get("chunk_size")
    out["n_action_steps"] = config.get("n_action_steps")
    out["state_dim"] = feature_width(config, "input_features", "STATE")
    out["action_dim"] = feature_width(config, "output_features", "ACTION")
    if kind not in SUPPORTED:
        return out
    gated = GATED_TOKENIZER.get(kind)
    if gated and local:
        out["notes"].append(
            f"the gated-tokenizer check is skipped for a local checkpoint; the export still loads "
            f"the tokenizer of {gated}, which needs a Hugging Face login that accepted its licence "
            f"(or that tokenizer in the Hugging Face cache)"
        )
    elif gated:
        ok, detail = _gated_access(gated)
        out["gated"] = {"repo": gated, "ok": ok, "detail": detail}
    pre = read("policy_preprocessor.json")
    post = read("policy_postprocessor.json")
    out["stats_mode"] = package_mode(kind, pre, config)
    problem = stats_problem(pre, post, files)
    out["stats"] = {"ok": problem is None, "detail": problem or "present"}
    return out


def _bind_revision(kind: str, model: str, revision: str | None) -> None:
    """Make the policy loader fetch `model` at the commit the preflight resolved, not `main`."""
    if not revision or revision == LOCAL_REVISION:
        return
    import importlib

    module = importlib.import_module(f"physicalai.policies.{kind}.policy")
    orig = module.hf_hub_download

    def bound(repo_id, *args, **kwargs):
        if str(repo_id) == model and kwargs.get("revision") is None:
            kwargs["revision"] = revision
        return orig(repo_id, *args, **kwargs)

    module.hf_hub_download = bound


def _load_policy(kind: str, model: str, rtc: bool):
    # n_action_steps=None keeps the checkpoint's chunk length (the default 50 rejects 15-step
    # checkpoints such as DROID's).
    if kind == "smolvla":
        from physicalai.policies import SmolVLA as cls

        policy = cls(pretrained_name_or_path=model, dtype="float32", n_action_steps=None)
    else:
        from physicalai.policies import Pi05 as cls

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
    kind, revision = pre["policy_type"], pre.get("revision")
    # --out is the CLI's staging directory; the f32 IR goes beside it, under the same unique name.
    out = Path(args.out)
    tmp = out.with_name(out.name + ".f32")
    stats = load_stats(Path(args.stats)) if args.stats else None

    if revision == LOCAL_REVISION:
        link = escaping_link(Path(args.model))
        if link is not None:
            log(f"EXPORT FAILED: {link} is a link that leaves the checkpoint directory")
            return 1
    t0 = time.perf_counter()
    _bind_revision(kind, args.model, revision)
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
        copy_package(tmp, out, ir)
        (out / ir).parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(ov.Core().read_model(tmp / ir), str(out / ir), compress_to_fp16=True)
        log(f"FP16 OK bin_mb={round((out / ir).with_suffix('.bin').stat().st_size / 1e6)}")
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
        for name in pre["license_files"]:
            if revision == LOCAL_REVISION:
                src = local_file(Path(args.model), name)
                if src is None:
                    raise ValueError(f"{name} left the local checkpoint during the export")
                shutil.copy2(src, out / name, follow_symlinks=False)
            else:
                from huggingface_hub import hf_hub_download

                shutil.copy2(hf_hub_download(args.model, name, revision=revision), out / name)
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
