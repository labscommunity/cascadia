"""Is a fine-tune's backbone byte-identical to its stock Qwen base?

File hashes can't answer it (the shard layouts differ), so this compares safetensors headers
(names, dtypes, shapes) and then sha256s slices of the raw bytes of EVERY backbone tensor via HTTP
range requests. Nothing large is downloaded.

usage: python backbone_diff.py A_REPO B_REPO [--per-layer] [--slice BYTES]

Names are matched per repo after normalisation, so a text-only `Qwen3_5ForCausalLM` checkpoint
(`model.layers.N.*`, `model.embed_tokens`, `model.norm`) lines up with a VLM checkpoint
(`model.language_model.layers.N.*`): the `model.language_model.` / `model.` prefix is stripped, and
vision (`visual.*`) and MTP (`mtp.*`) tensors are ignored. Compared: every decoder-layer tensor
(attention / linear-attention / MLP projections, norms, conv, A_log, dt_bias), embed_tokens, the final
norm and lm_head. Each tensor's first and last SLICE bytes are hashed (a fine-tune or a merged LoRA
moves essentially every element).

The last line is always one verdict:
  VERDICT: STOCK      every compared tensor is identical            (exit 0)
  VERDICT: FINETUNED  at least one compared tensor differs          (exit 0)
  VERDICT: UNKNOWN    nothing comparable: no common decoder-layer tensors, layer sets differ, or a
                      common tensor has a different dtype/shape     (exit 2)
--per-layer prints one `layer {i}: SAME|DIFF (n tensors[, differing: ...])` line per decoder layer.
"""

import argparse
import hashlib
import json
import re
import struct
import sys
import urllib.request

HF = "https://huggingface.co"
SLICE = 256 << 10


def get(url, rng=None):
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req) as r:
        return r.read()


def headers(repo):
    """tensor name -> (file url, dtype, shape, abs_start, abs_end)"""
    idx = json.loads(get(f"{HF}/{repo}/resolve/main/model.safetensors.index.json"))
    out = {}
    for fname in sorted(set(idx["weight_map"].values())):
        url = f"{HF}/{repo}/resolve/main/{fname}"
        n = struct.unpack("<Q", get(url, (0, 7)))[0]
        hdr = json.loads(get(url, (8, 8 + n - 1)))
        hdr.pop("__metadata__", None)
        for name, t in hdr.items():
            s, e = t["data_offsets"]
            out[name] = (url, t["dtype"], tuple(t["shape"]), 8 + n + s, 8 + n + e)
    return out


def tensor_sha(entry, cap=SLICE):
    """sha256 of the tensor's first and last `cap` bytes (all of it when it is no larger than 2*cap)."""
    url, _, _, s, e = entry
    h = hashlib.sha256()
    if e - s <= 2 * cap:
        h.update(get(url, (s, e - 1)) if e > s else b"")
    else:
        h.update(get(url, (s, s + cap - 1)))
        h.update(get(url, (e - cap, e - 1)))
    return h.hexdigest()


_IGNORED = re.compile(r"^(?:model\.)?(?:visual|vision_tower|mtp)\.|^mtp\.")
_LAYER = re.compile(r"^layers\.(\d+)\.(.+)$")
_TOP = {"embed_tokens.weight", "norm.weight", "lm_head.weight"}


def canonical(name):
    """Backbone name with the repo's prefix stripped (`layers.3.mlp.up_proj.weight`, `norm.weight`,
    `lm_head.weight`), or None for a tensor outside the text backbone (vision tower, MTP head)."""
    if _IGNORED.match(name):
        return None
    for p in ("model.language_model.", "language_model.model.", "language_model.", "model."):
        if name.startswith(p):
            name = name[len(p):]
            break
    if _LAYER.match(name) or name in _TOP:
        return name
    return None


def normalise(hdr):
    """{canonical name: entry}. Two raw names mapping to one canonical name is an error."""
    out = {}
    for raw, entry in hdr.items():
        c = canonical(raw)
        if c is None:
            continue
        if c in out:
            raise SystemExit(f"two tensors normalise to {c!r}; cannot compare this checkpoint")
        out[c] = entry
    return out


def layer_of(name):
    m = _LAYER.match(name)
    return int(m.group(1)) if m else None


def compare(a_hdr, b_hdr, sha=tensor_sha, per_layer=False, log=print):
    """-> (verdict, exit code). Prints the evidence through `log`; the verdict line is printed last."""
    a, b = normalise(a_hdr), normalise(b_hdr)
    la = {layer_of(n) for n in a} - {None}
    lb = {layer_of(n) for n in b} - {None}
    common = sorted(a.keys() & b.keys(), key=lambda n: (layer_of(n) if layer_of(n) is not None else -1, n))
    log(f"A: {len(a)} backbone tensors ({len(la)} layers) | B: {len(b)} ({len(lb)} layers) | common {len(common)}")
    only_a, only_b = sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())
    if only_a:
        log(f"only in A: {len(only_a)}  {only_a[:8]}")
    if only_b:
        log(f"only in B: {len(only_b)}  {only_b[:8]}")

    def verdict(v, why):
        log(f"VERDICT: {v}  ({why})")
        return v, (2 if v == "UNKNOWN" else 0)

    layered = [n for n in common if layer_of(n) is not None]
    if not layered:
        return verdict("UNKNOWN", "no common decoder-layer tensors after normalising names")
    if la != lb:
        return verdict("UNKNOWN", f"decoder layer sets differ: A has {len(la)}, B has {len(lb)}")
    missing = [n for n in a.keys() ^ b.keys() if layer_of(n) is not None]
    if missing:
        return verdict("UNKNOWN", f"{len(missing)} decoder-layer tensor(s) exist on one side only, e.g. {sorted(missing)[:3]}")
    mism = [n for n in common if a[n][1:3] != b[n][1:3]]
    if mism:
        return verdict("UNKNOWN", f"{len(mism)} common tensor(s) differ in dtype/shape, e.g. "
                                  f"{mism[0]}: {a[mism[0]][1:3]} vs {b[mism[0]][1:3]}")
    diff_by_layer, top_diff, n_diff = {}, [], 0
    for n in common:
        same = sha(a[n]) == sha(b[n])
        n_diff += not same
        i = layer_of(n)
        if i is None:
            log(f"  {'SAME' if same else 'DIFF'}  {n}  {a[n][1]} {a[n][2]}")
            if not same:
                top_diff.append(n)
        else:
            diff_by_layer.setdefault(i, [0, []])
            diff_by_layer[i][0] += 1
            if not same:
                diff_by_layer[i][1].append(n.split(".", 2)[2])
    changed = sorted(i for i, (_, d) in diff_by_layer.items() if d)
    if per_layer:
        for i in sorted(diff_by_layer):
            cnt, d = diff_by_layer[i]
            log(f"layer {i}: {'DIFF' if d else 'SAME'} ({cnt} tensors{', differing: ' + ', '.join(d) if d else ''})")
    log(f"compared {len(common)} tensors: {len(common) - n_diff} identical, {n_diff} different; "
        f"layers changed: {changed if changed else 'none'}")
    if n_diff:
        return verdict("FINETUNED", f"{len(changed)} layer(s) and {len(top_diff)} top-level tensor(s) differ")
    return verdict("STOCK", f"all {len(common)} common backbone tensors identical")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a_repo")
    ap.add_argument("b_repo")
    ap.add_argument("--per-layer", action="store_true", help="print one SAME/DIFF line per decoder layer")
    ap.add_argument("--slice", type=int, default=SLICE, help="bytes hashed at each end of every tensor")
    a = ap.parse_args(argv)
    print(f"A = {a.a_repo}\nB = {a.b_repo}")
    _, code = compare(headers(a.a_repo), headers(a.b_repo), sha=lambda e: tensor_sha(e, a.slice), per_layer=a.per_layer)
    sys.exit(code)


if __name__ == "__main__":
    main()
