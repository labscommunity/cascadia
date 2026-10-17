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
file cannot be renamed (each rename is retried a few times, for handles that linger).

Interruptions are recoverable. Before its first change, a cut or a restore writes
TREE/cut_lm_head.journal. While the journal exists, every *.pre-cut file is an original, so the
next run of either command first moves every backup back over its file (finishing an interrupted
restore, or rolling back an interrupted cut), then deletes the journal. A cut that fails part way
rolls back at once; if the rollback also fails, the journal stays and the next run finishes it.

manifest.json must stay byte-identical across pipeline ranks (HELLO): in a pipeline, run this on
every rank's copy of the tree, with the same commit.
"""
import argparse
import gc
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "qwen36_surgery"))
import weight_swap as ws  # noqa: E402

BACKUP = ".pre-cut"
KEY = "lm_head_last_row"
JOURNAL = "cut_lm_head.journal"
TEMPS = ("stage.cut.xml", "stage.cut.bin")
RETRIES, RETRY_SLEEP_S = 5, 1.0


def _replace(src, dst):
    """os.replace, retried on PermissionError (Windows: a mapping or handle not yet released)."""
    for i in range(RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == RETRIES - 1:
                raise
            time.sleep(RETRY_SLEEP_S)


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
    _replace(path + ".tmp", path)


def _files(tree, sdir):
    return [os.path.join(sdir, "stage.bin"), os.path.join(sdir, "stage.xml"), os.path.join(tree, "manifest.json")]


def _write_journal(tree, op, sdir):
    _write_json(os.path.join(tree, JOURNAL), {"op": op, "stage_dir": os.path.basename(sdir)})


def recover(tree):
    """Finish whatever an interrupted cut or restore left: with a journal present, move every
    backup back over its file and clear temporaries. Returns the journal's op, or None."""
    jpath = os.path.join(tree, JOURNAL)
    if not os.path.exists(jpath):
        return None
    with open(jpath) as f:
        j = json.load(f)
    sdir = os.path.join(tree, j["stage_dir"])
    for p in _files(tree, sdir):
        if os.path.exists(p + BACKUP):
            _replace(p + BACKUP, p)
    for p in [os.path.join(sdir, t) for t in TEMPS] + [os.path.join(tree, "manifest.json" + BACKUP + ".tmp"),
                                                       os.path.join(tree, "manifest.json.tmp"),
                                                       os.path.join(tree, JOURNAL + ".tmp")]:
        if os.path.exists(p):
            os.remove(p)
    os.remove(jpath)
    print(f"{tree}: finished an interrupted {j['op']}: the tree is back to its uncut files")
    return j["op"]


def _rollback_or_explain(tree, err):
    try:
        recover(tree)
    except Exception as e2:  # noqa: BLE001 - the journal stays; the next run finishes it
        raise SystemExit(f"{tree}: the cut failed ({err!r}) and so did the rollback ({e2!r}). The journal "
                         f"{JOURNAL} is kept: stop whatever holds the files, then run --restore") from err
    raise SystemExit(f"{tree}: the cut failed ({err!r}); rolled back, the tree is unchanged") from err


def cut(tree):
    import openvino as ov
    recover(tree)
    mpath, manifest = _load(tree)
    sdir = last_stage_dir(tree, manifest)
    binf, xml, _ = _files(tree, sdir)
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
        raise SystemExit(f"{sdir}: the stage is not cut and no journal says why, but backups exist: {stale}. "
                         "Inspect them by hand: they may be originals from another tool")
    txml, tbin = (os.path.join(sdir, t) for t in TEMPS)
    for p in (txml, tbin, os.path.join(tree, JOURNAL + ".tmp")):
        if os.path.exists(p):
            os.remove(p)
    ov.save_model(model, txml, compress_to_fp16=False)
    del model
    gc.collect()  # release the mapping of stage.bin before renaming it (Windows)
    try:
        _write_journal(tree, "cut", sdir)
    except Exception as e:  # noqa: BLE001 - nothing was changed yet; drop the temporaries
        for p in (txml, tbin, os.path.join(tree, JOURNAL + ".tmp")):
            if os.path.exists(p):
                os.remove(p)
        raise SystemExit(f"{tree}: could not write the journal ({e!r}); the tree is unchanged") from e
    try:
        shutil.copy2(mpath, mpath + BACKUP + ".tmp")
        _replace(mpath + BACKUP + ".tmp", mpath + BACKUP)
        _replace(binf, binf + BACKUP)  # first: the file most likely to be held open
        _replace(xml, xml + BACKUP)
        _replace(tbin, binf)
        _replace(txml, xml)
        _write_json(mpath, with_key(manifest))
    except Exception as e:  # noqa: BLE001
        _rollback_or_explain(tree, e)
    os.remove(os.path.join(tree, JOURNAL))
    print(f"{sdir}: LM head cut to the last position; backups *{BACKUP}")
    return True


def restore(tree):
    if recover(tree) is not None:
        return
    mpath, manifest = _load(tree)
    sdir = last_stage_dir(tree, manifest)
    missing = [p + BACKUP for p in _files(tree, sdir) if not os.path.exists(p + BACKUP)]
    if missing:
        raise SystemExit(f"{sdir}: no backup to restore: {missing}")
    _write_journal(tree, "restore", sdir)
    recover(tree)
    print(f"{sdir}: restored")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree")
    ap.add_argument("--restore", action="store_true", help="put the *.pre-cut backups back")
    a = ap.parse_args(argv)
    (restore if a.restore else cut)(a.tree)


if __name__ == "__main__":
    main()
