"""Cut the LM head of an existing decision tree's last stage to the last position, in place.

  python tools/decision/cut_lm_head.py TREE            # backs up the files it changes as *.pre-cut, then cuts
  python tools/decision/cut_lm_head.py TREE --restore  # puts the backups back, byte for byte

Same graph as `decision_export.py` with the cut (weight_swap.cut_lm_head_to_last_position): the
LM head (and a grafted lm_head LoRA branch) reads only the last position, `final_hidden` still
carries every row. manifest.json gains "lm_head_last_row": true, placed where a fresh export puts
it. A tree already cut is left alone.

Files changed, each backed up first as <file>.pre-cut: stageN/stage.xml and stageN/stage.bin of
the last stage (renamed, not copied: no second multi-GB copy) and manifest.json (copied).
Peak memory is one stage. Stop the server first: OpenVINO maps stage.bin, and on Windows a mapped
file cannot be replaced.

manifest.json must stay byte-identical across pipeline ranks (HELLO): in a pipeline, run this on
every rank's copy of the tree, with the same commit.
"""
import argparse
import gc
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "qwen36_surgery"))
import weight_swap as ws  # noqa: E402

BACKUP = ".pre-cut"
KEY = "lm_head_last_row"


def last_stage_dir(tree, manifest):
    stages = manifest.get("stages") or []
    if not stages:
        raise SystemExit(f"{tree}: manifest.json lists no stages")
    last = max(stages, key=lambda s: s["stage"])
    if not last.get("has_head"):
        raise SystemExit(f"{tree}: the last stage in manifest.json (stage{last['stage']}) has no head")
    return os.path.join(tree, f"stage{last['stage']}")


def _load(tree):
    mpath = os.path.join(tree, "manifest.json")
    if not os.path.isfile(mpath):
        raise SystemExit(f"{tree}: no manifest.json (not a tree, or an export that did not finish)")
    with open(mpath) as f:
        return mpath, json.load(f)


def with_key(manifest):
    """The manifest with "lm_head_last_row": true just before "stages", where build_manifest puts it."""
    out = {}
    for k, v in manifest.items():
        if k == KEY:
            continue
        if k == "stages":
            out[KEY] = True
        out[k] = v
    if KEY not in out:
        out[KEY] = True
    return out


def _write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(path + ".tmp", path)


def cut(tree):
    import openvino as ov
    mpath, manifest = _load(tree)
    sdir = last_stage_dir(tree, manifest)
    xml, binf = os.path.join(sdir, "stage.xml"), os.path.join(sdir, "stage.bin")
    model = ov.Core().read_model(xml)
    if not ws.cut_lm_head_to_last_position(model):
        del model
        if not manifest.get(KEY):
            raise SystemExit(f"{sdir}: the stage is cut but manifest.json has no {KEY}; "
                             f"run --restore, then cut again")
        print(f"{sdir}: already cut")
        return False
    stale = [p + BACKUP for p in (xml, binf, mpath) if os.path.exists(p + BACKUP)]
    if stale:
        del model
        raise SystemExit(f"{sdir}: the stage is not cut, but backups exist: {stale}. They are not "
                         "this stage's files; move them away, then cut again")
    tmp = os.path.join(sdir, "stage.cut.xml")
    ov.save_model(model, tmp, compress_to_fp16=False)
    del model
    gc.collect()  # release the mapping of stage.bin before renaming it (Windows)
    shutil.copy2(mpath, mpath + BACKUP)
    os.replace(xml, xml + BACKUP)
    os.replace(binf, binf + BACKUP)
    os.replace(tmp, xml)
    os.replace(tmp[:-4] + ".bin", binf)
    _write_json(mpath, with_key(manifest))
    print(f"{sdir}: LM head cut to the last position; backups *{BACKUP}")
    return True


def restore(tree):
    mpath = os.path.join(tree, "manifest.json")
    if not os.path.exists(mpath + BACKUP) and not os.path.isfile(mpath):
        raise SystemExit(f"{tree}: no manifest.json")
    src = mpath + BACKUP if os.path.exists(mpath + BACKUP) else mpath
    with open(src) as f:
        manifest = json.load(f)
    sdir = last_stage_dir(tree, manifest)
    xml, binf = os.path.join(sdir, "stage.xml"), os.path.join(sdir, "stage.bin")
    missing = [p + BACKUP for p in (xml, binf) if not os.path.exists(p + BACKUP)]
    if missing:
        raise SystemExit(f"{sdir}: no backup to restore: {missing}")
    for p in (xml, binf, mpath):
        if os.path.exists(p + BACKUP):
            os.replace(p + BACKUP, p)
    for p in ("stage.cut.xml", "stage.cut.bin"):
        if os.path.exists(os.path.join(sdir, p)):
            os.remove(os.path.join(sdir, p))
    print(f"{sdir}: restored")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree")
    ap.add_argument("--restore", action="store_true", help="put the *.pre-cut backups back")
    a = ap.parse_args(argv)
    (restore if a.restore else cut)(a.tree)


if __name__ == "__main__":
    main()
