"""Is a fine-tune's backbone byte-identical to its stock Qwen base?

File hashes can't answer it (the shard layouts differ), so this compares safetensors headers
(names, dtypes, shapes) and then sha256s slices of the raw bytes of EVERY backbone tensor. No
shard is ever held whole in memory, but it is not a quick probe: see "Cost" below.

usage: python backbone_diff.py A B [--per-layer] [--slice BYTES] [--adapter ADAPTER]

A and B are each a Hugging Face repo id (read with HTTP range requests) or a local snapshot
directory (`model.safetensors.index.json` + shards, or a single `model.safetensors`; read from
disk, no network). A is the stock base, B the fine-tune (the merge check below accepts either
order). For a gated or private repo set HF_TOKEN in the environment: surrounding whitespace is
stripped, a value holding any other control, space or non-ASCII character is refused without being
shown, and it is sent only as the `Authorization` header of huggingface.co requests (never to the
CDN a download redirects to). It is never printed or logged: every request failure is reported by
URL and exception type only.

Names are matched per repo after normalisation, so a text-only `Qwen3_5ForCausalLM` checkpoint
(`model.layers.N.*`, `model.embed_tokens`, `model.norm`) lines up with a VLM checkpoint
(`model.language_model.layers.N.*`): the `model.language_model.` / `model.` prefix is stripped, and
vision (`visual.*`) and MTP (`mtp.*`) tensors are ignored. Compared: every decoder-layer tensor
(attention / linear-attention / MLP projections, norms, conv, A_log, dt_bias), embed_tokens, the final
norm and lm_head. Each tensor's first and last SLICE bytes are hashed (a fine-tune or a merged LoRA
moves essentially every element).

--adapter (a PEFT adapter dir or its adapter_model.safetensors; adapter_config.json's
target_modules when there is no adapter file: a list of module-name suffixes, a regex fullmatched
against the PEFT module name with or without its `model.` / `model.language_model.` prefix, or
"all-linear") names the modules the fine-tune's LoRA targets. When every differing tensor is the
`.weight` of one of those modules, the shards may carry the adapter already merged, and exporting
them with the same adapter grafted again would apply it twice. That pattern alone does not prove
a merge (an independent fine-tune of the same modules looks the same), so it is then verified: for
the one or two smallest differing decoder-layer targets (at most --merge-check-bytes each) both
tensors and the adapter's lora_A/lora_B are read whole, and (B - A) must equal +-scale * lora_B @
lora_A within the shard dtype's rounding (residual RMS <= RMS of one ulp, |cosine| >= 0.9). If it
does, MERGED; if the difference is unrelated to the adapter's delta (|cosine| < 0.5) AND clearly
beyond rounding (at least 5% of the elements outside the half-ulp bound a merge obeys) on every
checked tensor, FINETUNED; anything else (a config-only --adapter with no lora tensors, rank/alpha
patterns, DoRA, no small target, a delta below the dtype's resolution, mixed or in-between
evidence) is UNKNOWN. An --adapter whose targets match none of
the compared tensors is UNKNOWN as well.

The last line is always one verdict:
  VERDICT: STOCK      every compared tensor is identical                              (exit 0)
  VERDICT: FINETUNED  at least one compared tensor differs (with --adapter: at least one of them
                      outside the adapter's targets)                                  (exit 0)
  VERDICT: MERGED     with --adapter: every difference is on an adapter target and the
                      merge check proved B - A = scale * lora_B @ lora_A               (exit 3)
  VERDICT: UNKNOWN    nothing comparable: no common decoder-layer tensors, layer sets differ, a
                      tensor (decoder-layer, embed_tokens, final norm or lm_head) exists on one
                      side only, or a common tensor has a different dtype/shape; with
                      --adapter: targets match nothing, or the merge is not decidable  (exit 2)
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


def hf_token():
    """HF_TOKEN from the environment with surrounding whitespace stripped, or None when unset/blank.
    A value that still holds anything but printable ASCII (a stray CR/LF inside, a space, a control
    or non-ASCII character) is refused: http.client would otherwise reject the header with an error
    that quotes it. The message never contains the value."""
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        return None
    if any(not 0x21 <= ord(c) <= 0x7E for c in tok):
        raise SystemExit("HF_TOKEN is malformed: it holds a space, control or non-ASCII character "
                         "(value not shown); fix it in the environment")
    return tok


def get(url, rng=None):
    """GET `url` (bytes rng[0]..rng[1] inclusive when given). HF_TOKEN, when set, goes only into the
    Authorization header of this request; it is never printed, and every failure (HTTP status, URL,
    connection, header or socket error) becomes a SystemExit naming only the URL and the error type:
    an exception's own text (which may quote a header) is never shown."""
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    token = hf_token()
    if token:
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with _open(req) as r:
            status, body = r.status, r.read()
    except urllib.error.HTTPError as e:
        hint = " (gated or private: set HF_TOKEN, or pass a local snapshot dir)" if e.code in (401, 403) else ""
        raise SystemExit(f"HTTP {e.code} for {url}{hint}") from None
    except Exception as e:  # URLError, http.client.HTTPException, ValueError (bad header), OSError, ...
        raise SystemExit(f"request to {url} failed: {type(e).__name__}") from None
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


def _adapter_file(path):
    p = Path(path).expanduser()
    f = p / "adapter_model.safetensors" if p.is_dir() else p
    return f if f.is_file() else None


def _all_linear(name):
    """PEFT's `target_modules: "all-linear"`: every nn.Linear except the output layer. In a Qwen3.5
    decoder layer that is each 2-D projection `.weight` (attention, linear-attention in_proj_* /
    out_proj, MLP); norms, the depthwise conv1d, A_log, dt_bias, embed_tokens and lm_head are not."""
    if layer_of(name) is None or not name.endswith(".weight"):
        return False
    leaf = name[:-len(".weight")].rsplit(".", 1)[-1]
    return "norm" not in leaf and leaf != "conv1d"


def adapter_lora(path):
    """{canonical target `.weight` name: (lora_A entry, lora_B entry)} from the adapter file at `path`
    (adapter_model.safetensors or a dir holding it), or None when there is no adapter file."""
    f = _adapter_file(path)
    if f is None:
        return None
    hdr, base = shard_header(f)
    out = {}
    for k, t in hdr.items():
        if not k.endswith(".lora_A.weight"):
            continue
        stem = k[:-len(".lora_A.weight")]
        c = canonical(stem.removeprefix("base_model.model.") + ".weight")
        tb = hdr.get(stem + ".lora_B.weight")
        if c is None or tb is None:
            continue
        ent = lambda tt: (f, tt["dtype"], tuple(tt["shape"]), base + tt["data_offsets"][0], base + tt["data_offsets"][1])
        out[c] = (ent(t), ent(tb))
    return out


# Full-name prefixes a canonical `layers.N.…` module can carry under PEFT: bare, text-only CausalLM,
# a Qwen3.5 VLM-class model, and the older LLaVA-style wrapper.
_MODULE_PREFIXES = ("", "model.", "model.language_model.", "language_model.model.")


def adapter_targets(path):
    """Predicate over canonical tensor names: is this the `.weight` of a module the PEFT adapter at
    `path` targets? `path` is adapter_model.safetensors or a dir holding it (exact modules, from the
    `*.lora_A.weight` keys), else a dir with only adapter_config.json (its `target_modules`: a list
    of module-name suffixes, a regex string matched against the whole module name, or "all-linear",
    alone or in a list)."""
    p = Path(path).expanduser()
    f = _adapter_file(path)
    if f is not None:
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
    if isinstance(tm, str) and tm.lower() == "all-linear":
        return _all_linear
    if isinstance(tm, str):
        rx = re.compile(tm)
    else:
        tm = [t for t in tm if isinstance(t, str)]
        if any(t.lower() == "all-linear" for t in tm):
            return _all_linear

    def on(name):
        if not name.endswith(".weight"):
            return False
        mod = name[:-len(".weight")]
        if isinstance(tm, str):
            # PEFT fullmatches the regex against the module's full name, which depends on the model
            # class: text-only `model.layers.N.…`, a Qwen3.5 VLM `model.language_model.layers.N.…`
            return any(rx.fullmatch(pre + mod) is not None for pre in _MODULE_PREFIXES)
        return any(mod == t or mod.endswith("." + t) for t in tm)
    return on


# ---- MERGED proof: (B - A) == +-scale * lora_B @ lora_A within the shard's rounding --------------

_MANT_BITS = {"BF16": 7, "F16": 10, "F32": 23}
MERGE_CHECK_BYTES = 32 << 20
# FINETUNED needs this share of elements outside a genuine merge's rounding bound (a merge has 0).
BEYOND_ROUNDING_FRAC = 0.05


def read_tensor(entry):
    """The whole tensor as a float64 numpy array (BF16 / F16 / F32), or None for another dtype."""
    import numpy as np
    src, dtype, shape, s, e = entry
    if dtype not in _MANT_BITS:
        return None
    raw = read_range(src, s, e - 1) if e > s else b""
    if dtype == "BF16":
        arr = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        arr = np.frombuffer(raw, "<f2" if dtype == "F16" else "<f4")
    return arr.astype(np.float64).reshape(shape)


def _ulp(x, dtype):
    """One unit in the last place of each |x| in `dtype` (0 where x is 0)."""
    import numpy as np
    _, e = np.frexp(np.abs(x))
    return np.where(x == 0, 0.0, np.ldexp(1.0, e - 1 - _MANT_BITS[dtype]))


def merge_evidence(stock, tuned, lora_a, lora_b, scale, dtype):
    """-> (verdict, why) for one target tensor. MERGED when tuned - stock equals +-scale*B@A within
    `dtype`'s rounding (residual RMS no larger than the RMS of one ulp of the larger operand) and
    follows it closely (|cosine| >= 0.9: a delta below the dtype's resolution proves nothing).

    FINETUNED only when the difference is unrelated to the adapter delta (|cosine| < 0.5) AND the
    residual clearly exceeds rounding: at least BEYOND_ROUNDING_FRAC of the elements sit outside the
    bound a genuine merge obeys. Merging rounds each element once, so |diff - delta| is at most half
    an ulp of the larger operand, plus half an ulp of the delta when the delta was itself rounded to
    `dtype` first; a merge lands exactly 0 elements outside, an independent change of any size a
    sizeable share. A small cosine with the residual inside rounding is a delta lost below the
    dtype's resolution, not evidence of an independent fine-tune: UNKNOWN. UNKNOWN also for the right
    direction at the wrong scale."""
    import numpy as np
    if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_b.shape[1] != lora_a.shape[0]:
        return "UNKNOWN", f"lora_A {lora_a.shape} / lora_B {lora_b.shape} do not compose"
    delta = scale * (lora_b @ lora_a)
    diff = tuned - stock
    if delta.shape != diff.shape:
        return "UNKNOWN", f"adapter delta {delta.shape} vs tensor {diff.shape}"
    nd, nf = float(np.linalg.norm(delta)), float(np.linalg.norm(diff))
    if nd == 0.0 or nf == 0.0:
        return "UNKNOWN", "zero adapter delta or zero difference"
    cos = float(np.vdot(diff, delta)) / (nd * nf)
    sign = 1.0 if cos >= 0 else -1.0
    r = np.abs(diff - sign * delta)
    ulp = _ulp(np.maximum(np.abs(stock), np.abs(tuned)), dtype)
    resid = float(np.sqrt(np.mean(r ** 2)))
    noise = float(np.sqrt(np.mean(ulp ** 2)))
    bound = (0.5 * ulp + 2.0 ** -(_MANT_BITS[dtype] + 1) * np.abs(delta)) * (1 + 1e-9)
    beyond = float(np.mean(r > bound))
    stats = (f"cos {cos:+.4f}, residual RMS {resid:.3g} vs rounding RMS {noise:.3g}, "
             f"{beyond:.1%} of elements beyond rounding")
    if resid <= noise and abs(cos) >= 0.9:
        return "MERGED", stats
    if abs(cos) < 0.5 and beyond >= BEYOND_ROUNDING_FRAC:
        return "FINETUNED", stats
    if abs(cos) < 0.5:
        return "UNKNOWN", f"{stats}: the difference is within rounding of the adapter delta (a delta below the dtype's resolution)"
    return "UNKNOWN", stats


def lora_scale(adapter_path):
    """(scale, None) from adapter_config.json beside the adapter file (alpha/r, alpha/sqrt(r) with
    rsLoRA), or (None, reason) when one global scale cannot describe the adapter."""
    f = _adapter_file(adapter_path)
    cfg_file = f.parent / "adapter_config.json"
    if not cfg_file.is_file():
        return None, f"no adapter_config.json beside {f}"
    cfg = json.loads(cfg_file.read_text())
    for k in ("rank_pattern", "alpha_pattern"):
        if cfg.get(k):
            return None, f"adapter_config.json has a non-empty {k}"
    if cfg.get("use_dora"):
        return None, "adapter_config.json has use_dora: true"
    if "r" not in cfg or "lora_alpha" not in cfg:
        return None, "adapter_config.json lacks r / lora_alpha"
    r, alpha = cfg["r"], cfg["lora_alpha"]
    return (alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r), None


def merge_checker(adapter_path, max_tensors=2, max_bytes=MERGE_CHECK_BYTES, log=print):
    """A `merge_check(names, a, b) -> (verdict, why)` for `compare`, proving or refuting MERGED on
    the `max_tensors` smallest differing decoder-layer targets of at most `max_bytes` each."""
    lora = adapter_lora(adapter_path)

    def check(names, a, b):
        if lora is None:
            return "UNKNOWN", ("the pattern fits a merged LoRA, but --adapter has no adapter_model.safetensors "
                               "to verify it; pass the adapter file")
        scale, why = lora_scale(adapter_path)
        if scale is None:
            return "UNKNOWN", f"the pattern fits a merged LoRA, but it cannot be verified: {why}"
        cands = sorted((n for n in names if n in lora and layer_of(n) is not None and len(a[n][2]) == 2
                        and a[n][4] - a[n][3] <= max_bytes), key=lambda n: (a[n][4] - a[n][3], n))[:max_tensors]
        if not cands:
            return "UNKNOWN", (f"the pattern fits a merged LoRA, but no differing decoder-layer target of at most "
                               f"{max_bytes} bytes has adapter tensors to verify it")
        results = []
        for n in cands:
            stock, tuned = read_tensor(a[n]), read_tensor(b[n])
            la, lb = read_tensor(lora[n][0]), read_tensor(lora[n][1])
            if stock is None or tuned is None or la is None or lb is None:
                v, w = "UNKNOWN", "unsupported dtype"
            else:
                v, w = merge_evidence(stock, tuned, la, lb, scale, a[n][1])
            log(f"  merge check {n}: {v} ({w})")
            results.append(v)
        if all(v == "MERGED" for v in results):
            return "MERGED", (f"all {len(names)} differing tensor(s) are the adapter's target modules and "
                              f"{len(results)} checked tensor(s) equal stock + scale*B@A: the shards hold the LoRA "
                              f"merged; do not graft --lora onto them again")
        if all(v == "FINETUNED" for v in results):
            return "FINETUNED", (f"every difference is on an adapter target, but {len(results)} checked tensor(s) "
                                 f"differ from stock by something unrelated to scale*B@A: an independent fine-tune")
        return "UNKNOWN", f"merge check inconclusive ({', '.join(results)}); inspect the tensors"
    return check


def compare(a_hdr, b_hdr, sha=tensor_sha, per_layer=False, log=print, adapter=None, merge_check=None):
    """-> (verdict, exit code). Prints the evidence through `log`; the verdict line is printed last.
    `adapter`: a predicate from `adapter_targets`. When every difference is on an adapter target,
    `merge_check(differing names, a, b) -> (verdict, why)` (from `merge_checker`) decides between
    MERGED, FINETUNED and UNKNOWN; without one the merge is unproven and the verdict is UNKNOWN."""
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
    if adapter is not None and not any(adapter(n) for n in common):
        return verdict("UNKNOWN", "--adapter's target modules match none of the compared tensors; "
                                  "check target_modules (or pass the adapter file)")
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
            if merge_check is None:
                return verdict("UNKNOWN", f"all {n_diff} differing tensor(s) are the adapter's target modules, "
                                          f"but no merge check ran to prove MERGED")
            return verdict(*merge_check(differing, a, b))
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
                         "difference is verified and reported as MERGED (exit 3), FINETUNED or UNKNOWN")
    ap.add_argument("--merge-check-bytes", type=int, default=MERGE_CHECK_BYTES,
                    help="largest target tensor the merge check reads whole (bytes, per side)")
    a = ap.parse_args(argv)
    if a.slice < 1:
        ap.error("--slice must be at least 1")
    hf_token()  # a malformed HF_TOKEN fails here, before any work, without being shown
    targets = adapter_targets(a.adapter) if a.adapter else None
    check = merge_checker(a.adapter, max_bytes=a.merge_check_bytes) if a.adapter else None
    print(f"A = {a.a_repo}\nB = {a.b_repo}")
    _, code = compare(headers(a.a_repo), headers(a.b_repo), sha=lambda e: tensor_sha(e, a.slice),
                      per_layer=a.per_layer, adapter=targets, merge_check=check)
    sys.exit(code)


if __name__ == "__main__":
    main()
