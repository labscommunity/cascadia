"""Is a fine-tune's backbone byte-identical to its stock Qwen base?

File hashes can't answer it (the shard layouts differ), so this compares safetensors headers
(names, dtypes, shapes) and then sha256s slices of the raw bytes of EVERY backbone tensor. No
shard is ever held whole in memory, but it is not a quick probe: see "Cost" below.

usage: python backbone_diff.py A B [--per-layer] [--slice BYTES] [--adapter ADAPTER]

A and B are each a Hugging Face repo id (read with HTTP range requests) or a local snapshot
directory (`model.safetensors.index.json` + shards, or a single `model.safetensors`; read from
disk, no network). For a gated or private repo set HF_TOKEN in the environment: it is sent only as
the `Authorization` header of huggingface.co requests (never to the CDN a download redirects to)
and is never printed or logged.

Names are matched per repo after normalisation, so a text-only `Qwen3_5ForCausalLM` checkpoint
(`model.layers.N.*`, `model.embed_tokens`, `model.norm`) lines up with a VLM checkpoint
(`model.language_model.layers.N.*`): the `model.language_model.` / `model.` prefix is stripped, and
vision (`visual.*`) and MTP (`mtp.*`) tensors are ignored. Compared: every decoder-layer tensor
(attention / linear-attention / MLP projections, norms, conv, A_log, dt_bias), embed_tokens, the final
norm and lm_head. Each tensor's first and last SLICE bytes are hashed (a fine-tune or a merged LoRA
moves essentially every element).

--adapter (a PEFT adapter dir or its adapter_model.safetensors; adapter_config.json's
target_modules when there is no adapter file) names the modules the fine-tune's LoRA targets. When
every differing tensor is the `.weight` of one of those modules, the shards most likely carry the
adapter already merged, and exporting them with the same adapter grafted again applies it twice.

The last line is always one verdict:
  VERDICT: STOCK      every compared tensor is identical                              (exit 0)
  VERDICT: FINETUNED  at least one compared tensor differs (with --adapter: at least one of them
                      outside the adapter's targets)                                  (exit 0)
  VERDICT: MERGED     with --adapter: every difference is on an adapter target        (exit 3)
  VERDICT: UNKNOWN    nothing comparable: no common decoder-layer tensors, layer sets differ, a
                      tensor (decoder-layer, embed_tokens, final norm or lm_head) exists on one
                      side only, or a common tensor has a different dtype/shape        (exit 2)
--per-layer prints one `layer {i}: SAME|DIFF (n tensors[, differing: ...])` line per decoder layer.

Cost (remote repos): one header request per shard, then 1-2 range requests per tensor per repo
(two SLICE reads for any tensor larger than 2*SLICE). A 64-layer 27B backbone has ~850 tensors, so
expect ~1,800+ sequential HTTPS requests and ~0.9 GB downloaded at the default 256 KiB slice, i.e.
tens of minutes on a home link. `--slice 65536` cuts the bytes ~4x with both ends of every tensor
still covered; a local snapshot dir costs only disk reads.
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"
SLICE = 256 << 10


class _Redirect(urllib.request.HTTPRedirectHandler):
    """urllib copies `headers` (Range) onto a redirect but never `unredirected_hdrs`, where the
    token lives. Re-add it only when the redirect stays on the same host (a renamed repo on
    huggingface.co); a hop to the CDN's presigned URL never sees it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        auth = req.unredirected_hdrs.get("Authorization")
        if new is not None and auth and \
                urllib.parse.urlsplit(newurl).netloc == urllib.parse.urlsplit(req.full_url).netloc:
            new.add_unredirected_header("Authorization", auth)
        return new


_OPENER = urllib.request.build_opener(_Redirect)


def _open(req):
    return _OPENER.open(req)


def get(url, rng=None):
    """GET `url` (bytes rng[0]..rng[1] inclusive when given). HF_TOKEN, when set, goes only into the
    Authorization header of this request; it is never printed, and errors name the URL only."""
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with _open(req) as r:
            status, body = r.status, r.read()
    except urllib.error.HTTPError as e:
        hint = " (gated or private: set HF_TOKEN, or pass a local snapshot dir)" if e.code in (401, 403) else ""
        raise SystemExit(f"HTTP {e.code} for {url}{hint}") from None
    if rng:
        want = rng[1] - rng[0] + 1
        if status != 206 or len(body) != want:
            raise SystemExit(f"ranged read of {url} bytes {rng[0]}-{rng[1]}: HTTP {status} with {len(body)} "
                             f"bytes, want 206 with {want}; refusing to hash a whole or short body")
    return body


def read_range(src, start, end):
    """Bytes start..end (inclusive) of a local shard (a Path) or a remote one (a URL string)."""
    if isinstance(src, Path):
        with open(src, "rb") as f:
            f.seek(start)
            data = f.read(end - start + 1)
        if len(data) != end - start + 1:
            raise SystemExit(f"{src}: short read at {start}-{end}")
        return data
    return get(src, (start, end))


def _shards(root):
    """Shard file names of a local snapshot dir: the index's weight_map, else model.safetensors."""
    idx = root / "model.safetensors.index.json"
    if idx.is_file():
        return sorted(set(json.loads(idx.read_text())["weight_map"].values()))
    if (root / "model.safetensors").is_file():
        return ["model.safetensors"]
    raise SystemExit(f"{root}: no model.safetensors.index.json or model.safetensors")


def shard_header(src):
    """(header dict without __metadata__, data start offset) of one safetensors file."""
    n = struct.unpack("<Q", read_range(src, 0, 7))[0]
    hdr = json.loads(read_range(src, 8, 8 + n - 1))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def headers(repo):
    """tensor name -> (source, dtype, shape, abs_start, abs_end). `repo` is a local snapshot dir
    (source: the shard's Path) or a Hugging Face repo id (source: the shard's resolve URL)."""
    root = Path(repo).expanduser()
    if root.is_dir():
        srcs = [root / f for f in _shards(root)]
    else:
        idx = json.loads(get(f"{HF}/{repo}/resolve/main/model.safetensors.index.json"))
        srcs = [f"{HF}/{repo}/resolve/main/{f}" for f in sorted(set(idx["weight_map"].values()))]
    out = {}
    for src in srcs:
        hdr, base = shard_header(src)
        for name, t in hdr.items():
            s, e = t["data_offsets"]
            out[name] = (src, t["dtype"], tuple(t["shape"]), base + s, base + e)
    return out


def tensor_sha(entry, cap=SLICE):
    """sha256 of the tensor's first and last `cap` bytes (all of it when it is no larger than 2*cap)."""
    src, _, _, s, e = entry
    h = hashlib.sha256()
    if e - s <= 2 * cap:
        h.update(read_range(src, s, e - 1) if e > s else b"")
    else:
        h.update(read_range(src, s, s + cap - 1))
        h.update(read_range(src, e - cap, e - 1))
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


def adapter_targets(path):
    """Predicate over canonical tensor names: is this the `.weight` of a module the PEFT adapter at
    `path` targets? `path` is adapter_model.safetensors or a dir holding it (exact modules, from the
    `*.lora_A.weight` keys), else a dir with only adapter_config.json (its `target_modules`: a list
    of module-name suffixes, or a regex string matched against the whole module name)."""
    p = Path(path).expanduser()
    f = p / "adapter_model.safetensors" if p.is_dir() else p
    if f.is_file():
        hdr, _ = shard_header(f)
        mods = set()
        for k in hdr:
            if k.endswith(".lora_A.weight"):
                c = canonical(k.removeprefix("base_model.model.").removesuffix(".lora_A.weight") + ".weight")
                if c is not None:
                    mods.add(c)
        if not mods:
            raise SystemExit(f"{f}: no lora_A weights on a backbone module")
        return mods.__contains__
    cfg_file = p / "adapter_config.json" if p.is_dir() else None
    if cfg_file is None or not cfg_file.is_file():
        raise SystemExit(f"--adapter {path}: no adapter_model.safetensors or adapter_config.json")
    tm = json.loads(cfg_file.read_text()).get("target_modules")
    if not tm:
        raise SystemExit(f"{cfg_file}: no target_modules")

    def on(name):
        if not name.endswith(".weight"):
            return False
        mod = name[:-len(".weight")]
        if isinstance(tm, str):
            return re.fullmatch(tm, mod) is not None or re.fullmatch(tm, "model." + mod) is not None
        return any(mod == t or mod.endswith("." + t) for t in tm)
    return on


def compare(a_hdr, b_hdr, sha=tensor_sha, per_layer=False, log=print, adapter=None):
    """-> (verdict, exit code). Prints the evidence through `log`; the verdict line is printed last.
    `adapter`: a predicate from `adapter_targets`, which turns an all-on-target FINETUNED into MERGED."""
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
        return v, {"UNKNOWN": 2, "MERGED": 3}.get(v, 0)

    layered = [n for n in common if layer_of(n) is not None]
    if not layered:
        return verdict("UNKNOWN", "no common decoder-layer tensors after normalising names")
    if la != lb:
        return verdict("UNKNOWN", f"decoder layer sets differ: A has {len(la)}, B has {len(lb)}")
    missing = [n for n in a.keys() ^ b.keys() if layer_of(n) is not None]
    if missing:
        return verdict("UNKNOWN", f"{len(missing)} decoder-layer tensor(s) exist on one side only, e.g. {sorted(missing)[:3]}")
    top_one_side = sorted(n for n in a.keys() ^ b.keys() if n in _TOP)
    if top_one_side:
        sides = ", ".join(f"{n} only in {'A' if n in a else 'B'}" for n in top_one_side)
        return verdict("UNKNOWN", f"{sides}; tied vs untied heads (or a missing tensor) cannot be compared")
    mism = [n for n in common if a[n][1:3] != b[n][1:3]]
    if mism:
        return verdict("UNKNOWN", f"{len(mism)} common tensor(s) differ in dtype/shape, e.g. "
                                  f"{mism[0]}: {a[mism[0]][1:3]} vs {b[mism[0]][1:3]}")
    diff_by_layer, top_diff, n_diff, differing = {}, [], 0, []
    for n in common:
        same = sha(a[n]) == sha(b[n])
        n_diff += not same
        if not same:
            differing.append(n)
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
    if n_diff and adapter is not None:
        off = [n for n in differing if not adapter(n)]
        if not off:
            return verdict("MERGED", f"all {n_diff} differing tensor(s) are the adapter's target modules: the "
                                     f"shards likely hold the LoRA merged; do not graft --lora onto them again")
        log(f"{len(off)} differing tensor(s) outside the adapter's targets, e.g. {off[:3]}")
    if n_diff:
        return verdict("FINETUNED", f"{len(changed)} layer(s) and {len(top_diff)} top-level tensor(s) differ")
    return verdict("STOCK", f"all {len(common)} common backbone tensors identical")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a_repo", help="HF repo id or local snapshot dir")
    ap.add_argument("b_repo", help="HF repo id or local snapshot dir")
    ap.add_argument("--per-layer", action="store_true", help="print one SAME/DIFF line per decoder layer")
    ap.add_argument("--slice", type=int, default=SLICE, help="bytes hashed at each end of every tensor")
    ap.add_argument("--adapter", default=None,
                    help="PEFT adapter dir / adapter_model.safetensors of the fine-tune: an all-on-target "
                         "difference is reported as MERGED (exit 3)")
    a = ap.parse_args(argv)
    if a.slice < 1:
        ap.error("--slice must be at least 1")
    targets = adapter_targets(a.adapter) if a.adapter else None
    print(f"A = {a.a_repo}\nB = {a.b_repo}")
    _, code = compare(headers(a.a_repo), headers(a.b_repo), sha=lambda e: tensor_sha(e, a.slice),
                      per_layer=a.per_layer, adapter=targets)
    sys.exit(code)


if __name__ == "__main__":
    main()
