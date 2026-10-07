"""Spike step 1: is Clef's backbone byte-identical to its stock Qwen base?

File hashes can't answer it (the shard layouts differ), so this compares
safetensors headers (names, dtypes, shapes) and then sha256s the raw bytes of
sampled tensors via HTTP range requests. Nothing large is downloaded.

usage: python backbone_diff.py Cloudflare/clef-flash Qwen/Qwen3.5-9B [samples] [--per-layer]

--per-layer also compares the first 8 MB of every decoder layer's mlp.gate_proj.weight
(`layers.{i}.` under the auto-detected prefix) and prints one `layer {i}: SAME|DIFF` line each.
That probe found Clef-27B's fine-tune touching only layers 40-63.
"""

import hashlib
import json
import re
import struct
import sys
import urllib.request

HF = "https://huggingface.co"


def get(url, rng=None):
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req) as r:
        return r.read()


def headers(repo):
    """tensor name -> (file, dtype, shape, abs_start, abs_end)"""
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


def tensor_sha(entry, cap=8 << 20):
    url, _, _, s, e = entry
    # Hash at most the first `cap` bytes: enough to tell a fine-tune apart.
    return hashlib.sha256(get(url, (s, min(e, s + cap) - 1))).hexdigest()


_LAYER_RE = re.compile(r"^(.*?)layers\.(\d+)\.mlp\.gate_proj\.weight$")


def layer_tensors(names):
    """[(layer, tensor name)] for every decoder layer's mlp.gate_proj.weight, sorted by layer.
    The prefix (`model.layers.`, `model.language_model.layers.`, ...) is auto-detected: the one
    holding the most layers wins, so a vision tower's blocks never mix in."""
    by_prefix = {}
    for n in names:
        m = _LAYER_RE.match(n)
        if m:
            by_prefix.setdefault(m.group(1), []).append((int(m.group(2)), n))
    if not by_prefix:
        return []
    return sorted(max(by_prefix.values(), key=len))


def per_layer(a, b):
    layers = layer_tensors(sorted(a.keys() & b.keys()))
    if not layers:
        print("per-layer: no common layers.{i}.mlp.gate_proj.weight tensors")
        return
    for i, n in layers:
        print(f"layer {i}: {'SAME' if tensor_sha(a[n]) == tensor_sha(b[n]) else 'DIFF'}")


def main():
    argv = [x for x in sys.argv[1:] if x != "--per-layer"]
    a_repo, b_repo = argv[0], argv[1]
    samples = int(argv[2]) if len(argv) > 2 else 24
    a, b = headers(a_repo), headers(b_repo)
    only_a, only_b = sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())
    common = sorted(a.keys() & b.keys())
    print(f"{a_repo}: {len(a)} tensors | {b_repo}: {len(b)} tensors | common {len(common)}")

    def summarize(names, label):
        groups = {}
        for n in names:
            key = ".".join(p for p in n.split(".")[:3] if not p.isdigit())
            groups[key] = groups.get(key, 0) + 1
        print(f"only in {label}: {len(names)}  {dict(sorted(groups.items()))}")

    summarize(only_a, a_repo)
    summarize(only_b, b_repo)

    mism = [n for n in common if a[n][1:3] != b[n][1:3]]
    print(f"dtype/shape mismatches among common: {len(mism)}  {mism[:5]}")

    step = max(1, len(common) // samples)
    picks = common[::step][:samples]
    same = diff = 0
    for n in picks:
        eq = tensor_sha(a[n]) == tensor_sha(b[n])
        same += eq
        diff += not eq
        print(f"  {'SAME' if eq else 'DIFF'}  {n}  {a[n][1]} {a[n][2]}")
    print(f"sampled {len(picks)}: {same} identical, {diff} different")
    if "--per-layer" in sys.argv[1:]:
        per_layer(a, b)


if __name__ == "__main__":
    main()
