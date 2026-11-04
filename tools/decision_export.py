"""Build a Cascadia decision-model tree (docs/DECISIONS.md).

  jev:  python tools/decision_export.py jev  --base-ir STOCK_INT8_IR --adapter-dir JEV/adapter_vllm --out TREE [--stages N]
        (--weights FT_DIR only for a fine-tuned, NOT LoRA-merged backbone, with --weights-not-merged)
  kev:  python tools/decision_export.py kev  --base-ir STOCK_INT8_IR --weights QWEN35_9B_BASE --lora KEV/adapter_model.safetensors \
                                             --head-pt KEV/head.pt --out TREE
        (--weights and --lora are required; the tree ships KEV/tokenizer.json, the one beside head.pt,
        not the Base checkpoint's; --tokenizer FILE overrides that for any family)
  clef: python tools/decision_export.py clef --base-ir STOCK_INT8_IR --weights CLEF_DIR --clef-dir CLEF_DIR --out TREE
  any family: --max-tokens N sets decision.json max_tokens (default: resolve_max_tokens)
  any family: --stages N (default 1, a single box; one per rank for a pipeline; 8 for a low-memory
              single box). A 1-stage export is refused before any work when its estimated peak
              RAM (export_peak_bytes) exceeds the machine's (--skip-memory-check overrides);
              multi-stage exports are not checked.

The stage IRs come from export_qwen36_moe.run_export (one stage in memory at a time).
The decision profile and head assets are written BEFORE the stage export, so the
manifest run_export writes last stays the commit marker for the whole tree, and a
bad head.pt / adapter dir fails before the long IR export starts.

Tree layout (read by crates/cascadia-api/src/decisions/profile.rs and
HeadAssets::load in crates/cascadia-engine-openvino/src/decision.rs):
  decision.json                  {"family", "max_tokens", "jev"|"kev"|"clef": {...}}
  decision/head.json             {"kind": "verbalizer"} | {"kind": "pointer", "dp"} | {"kind": "joint", "vocab"}
  decision/pointer_{q,k}_{w,b}.f32   kev: little-endian f32, w row-major (dp, hidden)
  decision/joint_head.xml/.bin   clef: OpenVINO joint schema head
  decision/lm_head.f16           clef: little-endian f16, row-major (vocab, hidden)
"""
import argparse
import json
import os
import string
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "qwen36_surgery"))
sys.path.insert(0, os.path.dirname(__file__))


# Calibration bounds DecisionProfile::load refuses a profile outside of
# (MIN_TEMPERATURE, MAX_TEMPERATURE, MAX_ABS_BIAS in crates/cascadia-api/src/decisions/profile.rs).
MIN_TEMPERATURE = 1e-3
MAX_TEMPERATURE = 1e3
MAX_ABS_BIAS = 1e4
# Pass-length caps the server clamps max_tokens to (crates/cascadia-types/src/decision.rs):
# every decision pass, and Clef's joint pass, whose `hidden` input must also fit the shim's
# input limit. test_bounds_mirror_the_rust_constants pins all six.
MAX_DECISION_PROMPT = 131_072
MAX_JOINT_PROMPT = 16_384
MAX_HEAD_INPUT_BYTES = 320 * 1024 * 1024
DEFAULT_MAX_TOKENS = {"jev": 32768, "kev": 73728, "clef": 16384}
HARD_MAX_TOKENS = {"jev": ("MAX_DECISION_PROMPT", MAX_DECISION_PROMPT),
                   "kev": ("MAX_DECISION_PROMPT", MAX_DECISION_PROMPT),
                   "clef": ("MAX_JOINT_PROMPT", MAX_JOINT_PROMPT)}


def joint_prompt_cap(hidden):
    """joint_prompt_cap in decision.rs: MAX_JOINT_PROMPT, tightened so a [T, hidden] f32 input fits
    MAX_HEAD_INPUT_BYTES (it binds only above 5120 wide)."""
    return min(MAX_JOINT_PROMPT, MAX_HEAD_INPUT_BYTES // (4 * max(hidden, 1)))


def _family_cap(family, hidden):
    """(name, cap): the longest pass the server keeps for `family` at model width `hidden` (None: unknown)."""
    name, cap = HARD_MAX_TOKENS[family]
    if family == "clef" and hidden and joint_prompt_cap(hidden) < cap:
        return f"joint_prompt_cap({hidden}) for a {hidden}-wide model", joint_prompt_cap(hidden)
    return name, cap


def resolve_max_tokens(family, requested, mpe, src="the checkpoint", hidden=None):
    """decision.json max_tokens: `requested` (--max-tokens) or the family default, the default
    capped by the max_position_embeddings `mpe` that `src`/config.json records (None: not recorded)
    and by the family's server cap at model width `hidden`. A requested value above `mpe` or the
    server cap is refused (ValueError), never silently lowered."""
    name, cap = _family_cap(family, hidden)
    if requested is None:
        return min(DEFAULT_MAX_TOKENS[family], cap, mpe) if mpe else min(DEFAULT_MAX_TOKENS[family], cap)
    if requested > cap:
        raise ValueError(f"--max-tokens {requested} exceeds {family}'s {cap} ({name} in "
                         f"crates/cascadia-types/src/decision.rs; the server would clamp it)")
    if mpe and requested > mpe:
        raise ValueError(f"--max-tokens {requested} exceeds the max_position_embeddings {mpe} of {src}/config.json")
    return requested


def _in_bounds(v, lo, hi):
    # serde reads a JSON number into f64, never a bool or string; NaN fails both comparisons
    return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi


def check_calibration(profile):
    """Refuse a profile DecisionProfile::load would reject at startup: every JEV and Kev temperature
    in [MIN_TEMPERATURE, MAX_TEMPERATURE], every JEV bias in [-MAX_ABS_BIAS, MAX_ABS_BIAS]."""
    temps = [(f"jev.temperature.{k}", t) for k, t in (profile.get("jev") or {}).get("temperature", {}).items()]
    if "kev" in profile:
        temps.append(("kev.temperature", profile["kev"].get("temperature")))
    for field, t in temps:
        if not _in_bounds(t, MIN_TEMPERATURE, MAX_TEMPERATURE):
            raise ValueError(f"{field} must be in [{MIN_TEMPERATURE:g}, {MAX_TEMPERATURE:g}] (the server "
                             f"refuses the profile otherwise), got {t!r}")
    for i, b in enumerate((profile.get("jev") or {}).get("bias", [])):
        if not _in_bounds(b, -MAX_ABS_BIAS, MAX_ABS_BIAS):
            raise ValueError(f"jev.bias[{i}] must be in [-{MAX_ABS_BIAS:g}, {MAX_ABS_BIAS:g}] (the server "
                             f"refuses the profile otherwise), got {b!r}")


def _write(out_dir, profile, head):
    check_calibration(profile)
    out = Path(out_dir)
    (out / "decision").mkdir(parents=True, exist_ok=True)
    (out / "decision.json").write_text(json.dumps(profile, indent=1))
    (out / "decision" / "head.json").write_text(json.dumps(head))


# JEV's choice labels beyond its trained head, ported from serve_decide.py single_token_labels:
# A..Z then AA, AB, ..., ZZ, each kept when it is one token that survives intact in an option line
# ("x\n{label}) y"), at most MAX_OPTIONS (256) of them.
MAX_CHOICE_LABELS = 256
CHOICE_LABEL_CONTEXT = "x\n{}) y"


def single_token_labels(tok, context=CHOICE_LABEL_CONTEXT, cap=MAX_CHOICE_LABELS):
    """[(label, token id)] in label order; `tok.encode(text, add_special_tokens=False)` -> ids."""
    out = []
    for lab in list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]:
        t = tok.encode(lab, add_special_tokens=False)
        if len(t) == 1 and t[0] in tok.encode(context.format(lab), add_special_tokens=False):
            out.append((lab, t[0]))
        if len(out) == cap:
            break
    return out


class _TokenizerJson:
    """A tokenizer.json behind the transformers-style encode() single_token_labels calls."""

    def __init__(self, path):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(path)

    def encode(self, text, add_special_tokens=True):
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids


def load_tokenizer(path):
    return _TokenizerJson(path)


def shipped_tokenizer(base_ir, weights=None, override=None):
    """The tokenizer.json run_export puts in the tree: the override, else --weights', else the IR's."""
    if override is not None:
        return override
    if weights and (Path(weights) / "tokenizer.json").exists():
        return str(Path(weights) / "tokenizer.json")
    return str(Path(base_ir) / "tokenizer.json")


def write_jev_profile(adapter_dir, calibration_json, out_dir, max_tokens=32768, tokenizer=None):
    h = json.loads((Path(adapter_dir) / "decision_head.json").read_text())
    cal = json.loads(Path(calibration_json).read_text())["per_kind"]
    slots = h["slots"]
    n = len(h["verbalizer_ids"])
    if not (len(slots["verbalizers"]) == n == len(h["bias"])):
        raise ValueError(f"{adapter_dir}/decision_head.json: verbalizers/verbalizer_ids/bias lengths differ: "
                         f"{len(slots['verbalizers'])}/{n}/{len(h['bias'])}")
    missing = [k for k in ("noul", "score", "choice") if k not in slots["ranges"]]
    if missing:
        raise ValueError(f"{adapter_dir}/decision_head.json: slots.ranges lacks {', '.join(missing)}")
    for kind, (lo, hi) in slots["ranges"].items():
        if not 0 <= lo <= hi <= n:
            raise ValueError(f"{adapter_dir}/decision_head.json: slot range {kind} [{lo}, {hi}) "
                             f"out of bounds for {n} verbalizers")
    # Mirrors JevCodec::new: the prompt hard-codes false/true and 0..5, so noul is 2 wide, score 6.
    for kind, width in (("noul", 2), ("score", 6)):
        lo, hi = slots["ranges"][kind]
        if hi - lo != width:
            raise ValueError(f"{adapter_dir}/decision_head.json: slot range {kind} [{lo}, {hi}) "
                             f"must be {width} wide")
    missing = [k for k in ("noul", "choice", "score") if k not in cal]
    if missing:
        # the reference indexes TEMPS[kind] (jev.py:58): no silent 1.0
        raise ValueError(f"{calibration_json}: per_kind lacks temperature for {', '.join(missing)}")
    profile = {"family": "jev", "max_tokens": max_tokens, "jev": {
        "verbalizers": slots["verbalizers"], "verbalizer_ids": h["verbalizer_ids"], "bias": h["bias"],
        "slots": slots["ranges"], "temperature": {k: cal[k] for k in ("noul", "choice", "score")}}}
    if tokenizer is not None:
        # The reference asserts the first labels ARE the trained choice head (serve_decide.py:128);
        # a mismatch means the tree's tokenizer.json is not the one JEV was trained with.
        labels = single_token_labels(tokenizer)
        lo, hi = slots["ranges"]["choice"]
        got, want = [t for _, t in labels[: hi - lo]], h["verbalizer_ids"][lo:hi]
        if got != want:
            raise ValueError(f"{adapter_dir}/decision_head.json: the tokenizer's first {hi - lo} single-token "
                             f"choice labels {[lab for lab, _ in labels[: hi - lo]]} have ids {got}, not the "
                             f"trained choice verbalizer_ids[{lo}:{hi}] {want} (verbalizers "
                             f"{slots['verbalizers'][lo:hi]}): the tree's tokenizer.json is not JEV's")
        profile["jev"]["choice_labels"] = [lab for lab, _ in labels]
        profile["jev"]["choice_label_ids"] = [t for _, t in labels]
    _write(out_dir, profile, {"kind": "verbalizer"})


# The pointer head the server runs (decision.rs HeadAssets::Pointer, kev.rs): z = k(h_opt) . q(h_decide)
# / sqrt(dp) / temperature on causal rows of the stock tokenizer's embeddings.
KEV_HEAD_TENSORS = {"q.weight", "q.bias", "k.weight", "k.bias"}


def kev_head_unsupported(meta):
    """head.pt features (kev.checkpoint.Meta fields) that pointer head does not implement, as messages.
    Absent fields take Meta's defaults, which the server matches. weights_dtype is not refused: it only
    names the dtype upstream ran the backbone in; the tree is int8 either way and parity measures it."""
    bad = []
    if meta.get("option_isolation", False):
        bad.append(f"option_isolation={meta['option_isolation']!r} (each option attends only to the state, "
                   "a packed mask the server's causal rows do not build)")
    if meta.get("special_embeddings", False):
        bad.append(f"special_embeddings={meta['special_embeddings']!r} (trained special-token embedding "
                   "rows the export does not carry)")
    if meta.get("weights") not in (None, "lora", "full"):
        bad.append(f"weights {meta['weights']!r} (a kev export grafts a LoRA onto --weights, or swaps in "
                   "a full-weight checkpoint)")
    head = meta.get("head")
    if not isinstance(head, dict):
        # Meta's default head is None: a run that never saved its pointer head
        bad.append(f"head tensors {head!r} (the pointer head has exactly {sorted(KEV_HEAD_TENSORS)})")
        return bad
    extra = sorted(set(head) - KEV_HEAD_TENSORS)
    missing = sorted(KEV_HEAD_TENSORS - set(head))
    if extra or missing:
        bad.append(f"head tensors {sorted(head)} (the pointer head has exactly {sorted(KEV_HEAD_TENSORS)})")
    elif "head_dim" in meta:
        hd = meta["head_dim"]
        if not isinstance(hd, int) or isinstance(hd, bool):
            bad.append(f"head_dim {hd!r} (an integer: the pointer head's dp)")
        elif hd != head["q.weight"].shape[0]:
            bad.append(f"head_dim {hd} but q.weight has {head['q.weight'].shape[0]} rows")
    return bad


def kev_head_meta(head_pt):
    """head.pt's recorded backbone: {"base": HF id or None, "weights": "lora" | ... | None,
    "unsupported": kev_head_unsupported's messages}."""
    import torch
    meta = torch.load(head_pt, map_location="cpu", weights_only=True)
    if not isinstance(meta, dict):
        raise ValueError(f"expected a dict, got {type(meta).__name__}")
    return {"base": meta.get("base"), "weights": meta.get("weights"), "unsupported": kev_head_unsupported(meta)}


def write_kev_profile(head_pt, out_dir, max_tokens=73728, hidden=None):
    """`hidden`: the base IR's hidden size; the q/k weights' input dim must equal it."""
    import torch
    meta = torch.load(head_pt, map_location="cpu", weights_only=True)
    bad = kev_head_unsupported(meta)
    if bad:
        raise ValueError(f"{head_pt}: the server's Kev pointer head cannot serve " + "; ".join(bad))
    w = meta["head"]
    dp = w["q.weight"].shape[0]
    if hidden is not None:
        for key in ("q.weight", "k.weight"):
            if w[key].shape[1] != hidden:
                raise ValueError(f"{head_pt}: {key} has hidden dim {w[key].shape[1]} but the base IR's "
                                 f"hidden_size is {hidden}")
    profile = {"family": "kev", "max_tokens": max_tokens, "kev": {"temperature": float(meta["temperature"])}}
    try:
        check_calibration(profile)  # before the pointer files, as the other head.pt checks
    except ValueError as e:
        raise ValueError(f"{head_pt}: {e}") from None
    d = Path(out_dir) / "decision"
    d.mkdir(parents=True, exist_ok=True)
    for key, fname in (("q.weight", "pointer_q_w"), ("q.bias", "pointer_q_b"), ("k.weight", "pointer_k_w"), ("k.bias", "pointer_k_b")):
        w[key].float().contiguous().numpy().astype("<f4").tofile(d / f"{fname}.f32")
    _write(out_dir, profile, {"kind": "pointer", "dp": int(dp)})


def _lm_head_name(W, clef_dir):
    """lm_head.weight, or the input embeddings when the checkpoint ties them."""
    if "lm_head.weight" in W.idx:
        return "lm_head.weight"
    cfg_path = Path(clef_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    tied = cfg.get("tie_word_embeddings", (cfg.get("text_config") or {}).get("tie_word_embeddings", False))
    emb = W.prefix + "embed_tokens.weight"
    if tied and emb in W.idx:
        return emb
    raise SystemExit(f"{clef_dir}: no lm_head.weight in model.safetensors.index.json "
                     f"(and tie_word_embeddings is not set) - cannot write decision/lm_head.f16")


def _text_config(model_dir):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return cfg.get("text_config") or cfg


def _max_position_embeddings(model_dir):
    """The text config's max_position_embeddings, or None when model_dir has no config.json or it
    records none."""
    if not (Path(model_dir) / "config.json").exists():
        return None
    mpe = _text_config(model_dir).get("max_position_embeddings")
    return int(mpe) if mpe is not None else None


def _hidden_size(model_dir):
    """The text config's hidden_size, or None when model_dir has no config.json or it records none."""
    if not (Path(model_dir) / "config.json").exists():
        return None
    h = _text_config(model_dir).get("hidden_size")
    return int(h) if h is not None else None


def _positive_int(s):
    try:
        n = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r} is not an integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


# Export peak RAM (process tree working set) as a multiple of the base IR's int8 weights
# (openvino_language_model.bin + openvino_text_embeddings_model.bin: 7.40 + 0.95 = 8.35 GiB for
# Qwen3.5-9B), measured on pawan-01 (Task 58): 1 stage peaked at 8.25 GiB for JEV's graft (no
# --weights, 0.99x) and 20.3 / 20.6 GiB for Kev's / Clef's --weights swap (2.43x / 2.47x);
# Only 1-stage exports are checked: multi-stage peaks do not scale from these (8 stages measured
# 3.1 / 5.1 / 5.3 GiB; Clef-27B at 2 stages needs about 16 GB, far under 2.5 x its weights / 2).
PEAK_PER_WEIGHT_BYTE_GRAFT = 1.0
PEAK_PER_WEIGHT_BYTE_SWAP = 2.5
GIB = 1 << 30
IR_WEIGHT_FILES = ("openvino_language_model.bin", "openvino_text_embeddings_model.bin")


def ir_weight_bytes(base_ir):
    """Bytes of the base IR's int8 weight files the export reads (0 for a missing file)."""
    return sum(os.path.getsize(p) for p in (os.path.join(base_ir, f) for f in IR_WEIGHT_FILES)
               if os.path.isfile(p))


def export_peak_bytes(base_ir, swap):
    """Estimated peak RAM of a 1-stage export of `base_ir`; `swap` when --weights replaces the IR's
    weights (Kev, Clef), else the IR keeps its own (JEV's graft)."""
    per_byte = PEAK_PER_WEIGHT_BYTE_SWAP if swap else PEAK_PER_WEIGHT_BYTE_GRAFT
    return int(ir_weight_bytes(base_ir) * per_byte)


def physical_ram_bytes():
    """This machine's physical RAM in bytes, or None when it cannot be read (stdlib only)."""
    if sys.platform == "win32":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MemoryStatusEx()
        st.dwLength = ctypes.sizeof(st)
        return int(st.ullTotalPhys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)) else None
    try:  # Linux and macOS
        n = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        if n > 0:
            return n
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":
        import subprocess
        try:
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                      check=True).stdout.strip())
        except (OSError, ValueError, subprocess.CalledProcessError):
            return None
    return None


def memory_refusal(base_ir, swap, ram):
    """Why a 1-stage export would not fit in `ram` bytes, or None when it fits (or `ram` is
    unknown)."""
    need = export_peak_bytes(base_ir, swap)
    if ram is None or need <= ram:
        return None
    how = f"x{PEAK_PER_WEIGHT_BYTE_SWAP} for the --weights swap" if swap else f"x{PEAK_PER_WEIGHT_BYTE_GRAFT}"
    return (f"the export needs about {need / GIB:.1f} GiB of RAM at --stages 1 (the base IR's "
            f"{ir_weight_bytes(base_ir) / GIB:.1f} GiB of int8 weights, {how}), but this machine has "
            f"{ram / GIB:.1f} GiB: export with --stages 8 instead (a single-box tree for a low-memory "
            f"machine, one stage in memory at a time), or pass --skip-memory-check to try anyway")


def write_clef_profile(clef_dir, out_dir, max_tokens=16384, chunk=8192):
    from safetensors import safe_open
    from weight_swap import Weights
    from decision import clef_head

    W = Weights(clef_dir)
    name = _lm_head_name(W, clef_dir)
    d = Path(out_dir) / "decision"
    d.mkdir(parents=True, exist_ok=True)
    clef_head.export_joint_head(clef_dir, str(d / "joint_head.xml"))
    # lm_head rows fp16 for the lexical prior, streamed in chunks (vocab x hidden)
    f = safe_open(Path(clef_dir) / W.idx[name], "pt")
    sl = f.get_slice(name)
    vocab, _hidden = sl.get_shape()
    with open(d / "lm_head.f16", "wb") as out:
        for a in range(0, vocab, chunk):
            sl[a:min(vocab, a + chunk)].float().numpy().astype("<f2").tofile(out)
    # head.json last: it names the vocab the lm_head.f16 just written holds
    _write(out_dir, {"family": "clef", "max_tokens": max_tokens, "clef": {}}, {"kind": "joint", "vocab": int(vocab)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["jev", "kev", "clef"])
    ap.add_argument("--base-ir", required=True, help="Intel stock int8 IR dir (OpenVINO/Qwen3.5-9B-int8-ov or Qwen3.8-27B-int8-ov)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", type=_positive_int, default=1,
                    help="stage cut count: 1 for a single box (default), one per rank for a pipeline; "
                         "8 for a single box with too little RAM to export one stage (see --skip-memory-check)")
    ap.add_argument("--skip-memory-check", action="store_true",
                    help="export at --stages 1 even when the estimated peak RAM exceeds this machine's "
                         "physical RAM (multi-stage exports are not checked)")
    ap.add_argument("--layer-split", default=None)
    ap.add_argument("--weights", default=None, help="HF checkpoint whose weights replace the IR's (clef: defaults to --clef-dir)")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--weights-not-merged", action="store_true",
                    help="jev with --weights: state that those shards do NOT already hold the LoRA merged "
                         "(backbone_diff.py --adapter said FINETUNED, not MERGED); the adapter is grafted on top")
    ap.add_argument("--lora-scale", type=float, default=None, help="override adapter_config.json")
    ap.add_argument("--adapter-dir", default=None, help="jev: dir with decision_head.json + adapter_model.safetensors")
    ap.add_argument("--calibration", default=None, help="jev: calibration.json (default: ADAPTER_DIR/../calibration.json)")
    ap.add_argument("--head-pt", default=None, help="kev: head.pt")
    ap.add_argument("--clef-dir", default=None, help="clef: dir with joint_head.* and the checkpoint")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json the tree ships instead of the --weights/IR one (kev default: the one beside --head-pt)")
    ap.add_argument("--max-tokens", type=_positive_int, default=None,
                    help="decision.json max_tokens, the longest pass (default: jev 32768, kev 73728, clef 16384, "
                         "each capped by the checkpoint's max_position_embeddings; a larger N is refused)")
    a = ap.parse_args()
    need = {"jev": ("adapter_dir", "--adapter-dir"), "kev": ("head_pt", "--head-pt"), "clef": ("clef_dir", "--clef-dir")}
    attr, flag = need[a.family]
    if not getattr(a, attr):
        ap.error(f"{a.family} needs {flag}")
    if a.family == "jev" and a.weights and not a.weights_not_merged:
        # JEV always grafts its adapter; shards uploaded with that adapter merged in would get it
        # twice, and a reference built on the same shards with the same --lora agrees (P2-BLD-1).
        ap.error("jev --weights grafts the adapter onto those shards: pass --weights-not-merged to state "
                 "they do not already contain it (run tools/decision/backbone_diff.py STOCK_REPO WEIGHTS "
                 "--adapter ADAPTER_DIR first; VERDICT: MERGED means export without --weights instead)")
    if a.family == "clef" and (a.lora or a.lora_scale is not None):
        # Clef's joint head reads the hidden states of the full checkpoint it was trained on;
        # an adapter grafted on top would move them, and nothing here would notice.
        ap.error("clef takes no --lora/--lora-scale: --clef-dir is the full checkpoint its joint head reads")
    tokenizer = a.tokenizer
    if a.family == "kev":
        # Kev's pointer head was trained on its fine-tuned backbone's hidden states (Base + LoRA, or
        # a full-weight fine-tune): without them, the tree is the stock backbone under Kev's head and
        # serves wrong answers that a reference built the same way would not catch.
        try:
            meta = kev_head_meta(a.head_pt)
        except Exception as e:  # missing file, or torch's varied unpickling errors: a bad head.pt
            ap.error(f"--head-pt {a.head_pt} is not a readable head.pt: {type(e).__name__}: {e}")
        if meta["weights"] == "full":
            # A full-weight fine-tune (Kev-27B v2): --weights is the whole checkpoint the head was
            # trained on, swapped in as for clef, with nothing grafted on top.
            if not a.weights:
                ap.error(f"{a.head_pt} records weights 'full': kev needs --weights (the fine-tuned "
                         f"checkpoint beside it), not the stock IR's backbone")
            if a.lora or a.lora_scale is not None:
                ap.error(f"{a.head_pt} records weights 'full': --weights is the whole fine-tune, so kev "
                         f"takes no --lora/--lora-scale")
        else:
            missing = [f for f, v in (("--weights", a.weights), ("--lora", a.lora)) if not v]
            if missing:
                rec = f" ({a.head_pt} records base {meta['base']!r}, weights {meta['weights']!r})" \
                    if meta["base"] or meta["weights"] else ""
                ap.error(f"kev needs {' and '.join(missing)}: the pointer head reads its fine-tuned "
                         f"backbone's hidden states, not the stock IR's{rec}")
        if meta["weights"] not in (None, "lora", "full"):
            ap.error(f"{a.head_pt} records weights {meta['weights']!r}; a kev export grafts a LoRA "
                     f"(weights 'lora') onto --weights, or swaps in a full-weight checkpoint ('full')")
        if meta.get("unsupported"):
            ap.error(f"{a.head_pt}: the server's Kev pointer head cannot serve " + "; ".join(meta["unsupported"]))
        # The spike (and so the golden reference) tokenizes with Kev's own tokenizer.json, which
        # differs from the Base checkpoint's that run_export would otherwise copy from --weights.
        tokenizer = tokenizer or str(Path(a.head_pt).parent / "tokenizer.json")
    if tokenizer is not None and not Path(tokenizer).is_file():
        ap.error(f"{tokenizer} not found: a {a.family} tree must ship the tokenizer its head was "
                 f"trained with (pass --tokenizer FILE)")
    # Only an explicit override reaches run_export; jev/clef keep its default tokenizer choice.
    tok_kw = {"tokenizer": tokenizer} if tokenizer is not None else {}
    jev_tok = None
    if a.family == "jev":
        # JEV's extended choice labels come from the tokenizer.json the tree ships, so the server
        # encodes them exactly as the export checked them against the trained head.
        tok_path = shipped_tokenizer(a.base_ir, a.weights, tokenizer)
        if not Path(tok_path).is_file():
            ap.error(f"{tok_path} not found: a jev tree must ship a tokenizer.json (pass --tokenizer FILE)")
        try:
            jev_tok = load_tokenizer(tok_path)
        except Exception as e:  # the tokenizers library's varied parse errors: a bad tokenizer.json
            ap.error(f"{tok_path} is not a readable tokenizer.json: {type(e).__name__}: {e}")
    # The tree's window is the config.json run_export ships: the fine-tune's when --weights has one
    # (clef's --weights defaults to --clef-dir), else the base IR's.
    weights = a.weights or (a.clef_dir if a.family == "clef" else None)
    src = weights if weights and (Path(weights) / "config.json").exists() else a.base_ir
    # Clef's joint pass is also bounded by the model width (the base IR's: run_export refuses
    # --weights of another architecture).
    hidden = _hidden_size(a.base_ir) if a.family == "clef" else None
    try:
        max_tokens = resolve_max_tokens(a.family, a.max_tokens, _max_position_embeddings(src), src, hidden)
    except ValueError as e:
        ap.error(str(e))
    # Before any heavy work (asset writers, the stage export): a 1-stage export holds the whole
    # model at once, which a low-memory machine cannot. Multi-stage exports (pipelines, the
    # low-memory single box) are not checked: only the 1-stage peaks were measured.
    if a.stages == 1 and not a.skip_memory_check:
        why = memory_refusal(a.base_ir, bool(weights), physical_ram_bytes())
        if why:
            ap.error(why)
    from export_qwen36_moe import run_export

    # The decision assets below are written before run_export; drop any manifest from an
    # earlier export first, so a failure anywhere (asset writers, or run_export's pre-flight
    # checks that run before its own stale-manifest removal) never leaves the old manifest
    # and stages loadable next to new decision assets. manifest.json is the commit marker.
    stale = Path(a.out) / "manifest.json"
    if stale.exists():
        stale.unlink()

    # validate is always False: run_export refuses --validate with --weights/--lora
    # (decision trees are checked by the golden/parity gates instead).
    if a.family == "jev":
        lora = a.lora or os.path.join(a.adapter_dir, "adapter_model.safetensors")
        cal = a.calibration or os.path.join(os.path.dirname(os.path.abspath(a.adapter_dir)), "calibration.json")
        write_jev_profile(a.adapter_dir, cal, a.out, max_tokens=max_tokens, tokenizer=jev_tok)
        run_export(a.base_ir, a.out, a.stages, False, a.layer_split, weights=a.weights, lora=lora,
                   lora_scale=a.lora_scale, final_hidden=False, **tok_kw)
    elif a.family == "kev":
        write_kev_profile(a.head_pt, a.out, hidden=int(_text_config(a.base_ir)["hidden_size"]), max_tokens=max_tokens)
        run_export(a.base_ir, a.out, a.stages, False, a.layer_split, weights=a.weights, lora=a.lora,
                   lora_scale=a.lora_scale, final_hidden=True, **tok_kw)
    else:
        write_clef_profile(a.clef_dir, a.out, max_tokens=max_tokens)
        run_export(a.base_ir, a.out, a.stages, False, a.layer_split, weights=a.weights or a.clef_dir,
                   lora=None, lora_scale=None, final_hidden=True, **tok_kw)
    print(f"decision tree written: {a.out}")


if __name__ == "__main__":
    main()
