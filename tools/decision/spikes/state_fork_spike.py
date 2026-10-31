"""SPIKE (Task 51, 2026-10-19): GPU-side shared-state fork for decision passes. NOT production code.

A throwaway prototype that answers one question: can a case's shared prefix be prefilled once and each
question's suffix run from a device-side copy of that state, faster than today and with parity? It
drives the tree's embeddings IR and stage IRs with the OpenVINO Python API on GPU, mirroring
qwen36.rs `chain_pass_inner` (inputs per stage, 256-token spans, no fold), and runs the decision
heads in numpy. Nothing in crates/ uses it. Results fed a local design note that is not in the repo.

Verdict: the fork modes are NO-GO on OpenVINO 2026.4 GPU (RemoteTensor states refused, NaN rows once
the prefix exceeds 256 tokens with beam_idx expansion, a fault on shrink, unfaithful copies). H-lite
(mode H) is GO: shipped as Task 55, Kev p50 3.56 s to 2.41 s (docs/DECISIONS.md).

  python state_fork_spike.py analyze --family kev --dump fork-kev-9b.jsonl --cases td40.jsonl --tree DIR
  python state_fork_spike.py run     --family kev --dump ... --cases ... --tree DIR --out res-kev.jsonl \
      [--modes S,H,R,B,BR,P (or S+H+... from cmd)] [--limit N]
  python state_fork_spike.py report  --family kev --dump ... --cases ... --tree DIR \
      --results res-a.jsonl [res-b.jsonl ...] [--server t51-kev-9b.jsonl] [--baseline-p50 3.56]

Run one GPU process per mode set (each with its own S), e.g. S+SP+H+R, S+B, S+BK, S+P: a GPU fault
(CL_OUT_OF_RESOURCES, seen with BR and P) poisons the process, and per-process memory peaks stay
comparable. The OV model cache defaults to the server's (%LOCALAPPDATA%/cascadia/ov-cache).

Modes (each a timed phase over all cases after one untimed warm-up case, S first):
  S   sequential cold: every pass reset + full prefill (the parity reference; HEAD with reuse off)
  SP  as S, but the span is cut at the fork prefix (the boundary every fork mode has)
  H   host round trip: prefix once, VariableState get to host numpy, set back over the live
      requests per question (HEAD Kev's mechanism without its recreate + priming fold)
  HE  as H, but each restore first recreates the requests and runs a priming fold (HEAD's order)
  R   remote tensors: as H, but the saved states are GPU RemoteTensors (default context)
  B   batch fork, right padded: prefix at batch 1, suffixes as one batch with beam_idx=[0]*q
  BR  batch fork, ragged: prefix at batch 1, suffixes in batch steps that end at each distinct suffix
      length; beam_idx drops finished rows between steps, so no padding is ever fed
  BK  as BR, but the batch keeps every row (beam_idx = arange) and finished rows get filler
  BF  as B, but the pad is filler tokens with attention_mask 1 (one batch step, no masking)
  BDF as BF, but the prefix is prefilled as q identical rows (no beam_idx expansion, no sharing)
  BDK as BK, with BDF's duplicated prefix
  P   per-question requests: q extra InferRequests per stage, state copied vs_j.state = vs_0.state

The dump, cases and results hold request content (token ids, head outputs): keep them private.
"""
import argparse
import ctypes
import json
import math
import os
import time
from collections import Counter

import numpy as np

SPAN = 256  # qwen36.rs PREFILL_CHUNK
MROPE_ROWS = 4
MIN_SHARED = 32  # JEV: same-kind siblings share at least this many tokens


# ------------------------------------------------------------------------------------------------
# Inputs: dump lines aligned to cases, and fork groups


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def align(lines, cases):
    """[(case, [line, ...])]: consume dump lines until every question of the case is covered, then
    any round-2 lines (JEV tournament finals come after all of the request's round-1 passes)."""
    out, i = [], 0
    for c in cases:
        need, got = set(c["body"]["questions"]), []
        while need:
            line = lines[i]
            qids = {p[0] for p in line["parts"]}
            if not qids <= set(c["body"]["questions"]):
                raise SystemExit(f"dump line {i + 1} names {qids} outside case {c['id']}")
            need -= qids
            got.append(line)
            i += 1
        while i < len(lines) and lines[i].get("round") == 2:
            got.append(lines[i])
            i += 1
        out.append((c, got))
    if i != len(lines):
        raise SystemExit(f"{len(lines) - i} dump lines beyond the {len(cases)} cases")
    return out


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def read_limit(ro):
    """First position the head reads (a fork prefix must stop at or before it)."""
    h = ro["head"]
    if h["kind"] == "pointer":
        return min([h["decide_pos"], *h["option_pos"]])
    return len(ro["prompt_ids"]) - 1  # verbalizer: the last position's logits


def fork_groups(family, passes):
    """[(prefix_len, [pass index, ...])]. Kev: every pass of the request shares `cache_prefix`.
    JEV: passes of one round clustered by a common prefix >= MIN_SHARED (same-kind questions; not
    only consecutive ones). A group of one runs cold in every mode."""
    groups = []
    if family == "kev":
        cp = {p["readout"].get("cache_prefix") for p in passes}
        ids0 = passes[0]["readout"]["prompt_ids"]
        assert len(cp) == 1 and None not in cp, cp
        P = cp.pop()
        for p in passes:
            assert p["readout"]["prompt_ids"][:P] == ids0[:P]
            assert read_limit(p["readout"]) >= P
        return [(P, list(range(len(passes))))]
    left = list(range(len(passes)))
    while left:
        i = left.pop(0)
        ids_i = passes[i]["readout"]["prompt_ids"]
        rnd = passes[i].get("round", 1)
        members, P = [i], read_limit(passes[i]["readout"])
        for j in list(left):
            if passes[j].get("round", 1) != rnd:
                continue
            ids_j = passes[j]["readout"]["prompt_ids"]
            k = min(lcp(ids_i[:P], ids_j), read_limit(passes[j]["readout"]))
            if k >= MIN_SHARED:
                members.append(j)
                P = k
                left.remove(j)
        groups.append((P if len(members) > 1 else 0, members))
    return groups


def consecutive_sibling_share(passes):
    """Share of passes whose neighbour in dump order shares >= MIN_SHARED tokens (the brief's
    consecutive-only grouping)."""
    ids = [p["readout"]["prompt_ids"] for p in passes]
    hit = 0
    for i in range(len(ids)):
        ok = (i > 0 and lcp(ids[i - 1], ids[i]) >= MIN_SHARED) or \
             (i + 1 < len(ids) and lcp(ids[i], ids[i + 1]) >= MIN_SHARED)
        hit += ok
    return hit, len(ids)


# ------------------------------------------------------------------------------------------------
# Codec: head logits -> probabilities, one shared function for every mode and the server check


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def rescale_score(p6, levels):
    out = np.zeros(levels)
    top = len(p6) - 1
    for j, p in enumerate(p6):
        x = j * (levels - 1) / top
        lo = math.floor(x)
        w = x - lo
        out[lo] += p * (1 - w)
        if w > 0:
            out[lo + 1] += p * w
    return out


class Codec:
    def __init__(self, family, tree):
        self.family = family
        d = json.load(open(os.path.join(tree, "decision.json"), encoding="utf-8"))
        self.p = d[family]

    def probs(self, q, logits):
        """crates/cascadia-api/src/decisions/{kev,jev}.rs `probabilities` for one part."""
        l = np.asarray(logits, dtype=np.float64)
        if self.family == "kev":
            return softmax(l / self.p["temperature"])
        kind = q["type"]
        lo, hi = self.p["slots"][kind]
        if kind == "choice" and len(q["criteria"]) > hi - lo:
            b = np.array([self.p["bias"][lo + i] if lo + i < hi else 0.0 for i in range(len(l))])
            return softmax((l + b) / self.p["temperature"]["choice"])
        b = np.array(self.p["bias"][lo:lo + len(l)])
        p = softmax((l + b) / self.p["temperature"][kind])
        if kind == "score":
            return rescale_score(p, len(q["criteria"]))
        return p


def margin(p):
    s = sorted(p, reverse=True)
    return s[0] - (s[1] if len(s) > 1 else 0.0)


# ------------------------------------------------------------------------------------------------
# Memory probes


class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def proc_private():
    if os.name != "nt":
        return 0
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return -1
    return c.PrivateUsage


class Mem:
    """Running max of GPU plugin allocations (GPU_MEMORY_STATISTICS, all kinds) and process private
    bytes, sampled at the points a mode holds the most state."""

    def __init__(self, core):
        self.core = core
        self.reset()

    def gpu(self):
        try:
            return sum(self.core.get_property("GPU", "GPU_MEMORY_STATISTICS").values())
        except Exception:
            return -1

    def reset(self):
        self.max_gpu, self.max_priv = 0, 0

    def sample(self):
        self.max_gpu = max(self.max_gpu, self.gpu())
        self.max_priv = max(self.max_priv, proc_private())


# ------------------------------------------------------------------------------------------------
# The chain, as qwen36.rs drives it


class Chain:
    def __init__(self, tree, device="GPU", cache_dir=None):
        import openvino as ov
        self.ov = ov
        self.core = ov.Core()
        cfg = {"CACHE_DIR": cache_dir} if cache_dir else {}
        man = json.load(open(os.path.join(tree, "manifest.json"), encoding="utf-8"))
        self.width = man["hidden_size"]
        self.last_only = bool(man.get("last_logits_only"))
        self.has_fh = bool(man.get("final_hidden_output"))
        t = time.perf_counter()
        self.emb = self.core.compile_model(os.path.join(tree, "openvino_text_embeddings_model.xml"),
                                           device, cfg)
        self.stages = [self.core.compile_model(os.path.join(tree, f"stage{i}", "stage.xml"), device, cfg)
                       for i in range(len(man["stages"]))]
        self.compile_s = time.perf_counter() - t
        self.emb_req = self.emb.create_infer_request()
        self.emb_in = self.emb.inputs[0].get_any_name()
        self.names = [[i.get_any_name() for i in s.inputs] for s in self.stages]
        self.reqs = self.new_requests()
        self.restored = False
        self.recreate_after_restore = True
        self.ctx = None
        self.mem = Mem(self.core)

    def new_requests(self):
        return [s.create_infer_request() for s in self.stages]

    def states(self, reqs):
        return [vs for r in reqs for vs in r.query_state()]

    def reset(self, reqs):
        for r in reqs:
            r.reset_state()

    def fresh(self):
        """reset_all: after a set_state on these requests, reset_state leaves residue on this
        model (qwen36.rs `reset_all`), so the engine recreates the requests; otherwise it resets."""
        if self.restored and self.recreate_after_restore:
            self.reqs = self.new_requests()
            self.restored = False
        else:
            self.reset(self.reqs)

    def embed(self, ids2d):
        a = np.ascontiguousarray(np.asarray(ids2d, dtype=np.int64))
        self.emb_req.set_tensor(self.emb_in, self.ov.Tensor(a))
        self.emb_req.infer()
        return self.emb_req.get_output_tensor(0).data.copy()

    def chunk(self, reqs, ids2d, t0, mask=None, beam=None, want_fh=False, emb=None):
        """One chain pass over absolute positions [t0, t0+n) for B rows. Returns (out0, out1)."""
        ov = self.ov
        if emb is None:
            emb = self.embed(np.asarray(ids2d))
        B, n = emb.shape[:2]
        t1 = t0 + n
        if mask is None:
            mask = np.ones((B, t1), dtype=np.int64)
        if beam is None:
            beam = np.arange(B, dtype=np.int32)
        pos = np.ascontiguousarray(np.broadcast_to(np.arange(t0, t1, dtype=np.int64), (MROPE_ROWS, B, n)))
        zeros = None
        hidden, fh = emb, None
        for j, r in enumerate(reqs):
            for name in self.names[j]:
                if name == "stage_hidden":
                    arr = hidden
                elif "embed" in name:
                    if j == 0:
                        arr = hidden
                    else:
                        if zeros is None:
                            zeros = np.zeros((B, n, self.width), dtype=np.float32)
                        arr = zeros
                elif "attention_mask" in name:
                    arr = mask
                elif "position" in name:
                    arr = pos
                elif "beam" in name:
                    arr = np.asarray(beam, dtype=np.int32)
                else:
                    raise SystemExit(f"stage{j}: unexpected input {name}")
                r.set_tensor(name, ov.Tensor(np.ascontiguousarray(arr)))
            r.infer()
            out0 = r.get_output_tensor(0).data.copy()
            if want_fh and j + 1 == len(reqs):
                fh = r.get_output_tensor(1).data.copy()
            hidden = out0
        return hidden, fh

    def prefill(self, reqs, ids, t0, end, need=(), stop_at=None):
        """Prefill ids[t0:end] in SPAN spans (cut at `stop_at` too, as next_prefill_end does for a
        snapshot boundary). Returns (last logits row, {pos: final_hidden row})."""
        rows, last = {}, None
        p = t0
        while p < end:
            e = min(p + SPAN, end)
            if stop_at is not None and p < stop_at < e:
                e = stop_at
            out0, fh = self.chunk(reqs, [ids[p:e]], p, want_fh=bool(need))
            for q in need:
                if p <= q < e:
                    rows[q] = fh[0, q - p].copy()
            last = out0[0, -1].copy()
            p = e
        return last, rows


# ------------------------------------------------------------------------------------------------
# Heads


class Heads:
    def __init__(self, family, tree, width):
        self.family = family
        if family == "kev":
            d = os.path.join(tree, "decision")
            dp = json.load(open(os.path.join(d, "head.json")))["dp"]
            rd = lambda n, s: np.fromfile(os.path.join(d, n), dtype="<f4").reshape(s)
            self.dp = dp
            self.qw, self.qb = rd("pointer_q_w.f32", (dp, width)), rd("pointer_q_b.f32", (dp,))
            self.kw, self.kb = rd("pointer_k_w.f32", (dp, width)), rd("pointer_k_b.f32", (dp,))

    def need(self, ro):
        h = ro["head"]
        return [h["decide_pos"], *h["option_pos"]] if h["kind"] == "pointer" else []

    def run(self, ro, last, rows):
        h = ro["head"]
        if h["kind"] == "pointer":
            q = self.qw @ rows[h["decide_pos"]] + self.qb
            k = np.stack([self.kw @ rows[p] + self.kb for p in h["option_pos"]])
            return (k @ q / math.sqrt(self.dp)).tolist()
        return [float(last[t]) for t in h["token_ids"]]

    def raw(self, ro, last, rows):
        """What the brief compares raw: final_hidden rows (Kev), the logits row (JEV)."""
        if ro["head"]["kind"] == "pointer":
            return np.stack([rows[p] for p in self.need(ro)])
        return last


# ------------------------------------------------------------------------------------------------
# Modes. Each takes (chain, heads, passes, prefix, members) and returns {member: (logits, raw)}
# plus per-mode extras (copy ms per question).


def mode_S(ch, hd, passes, P, members, ctx):
    out = {}
    for m in members:
        ro = passes[m]["readout"]
        ids = ro["prompt_ids"]
        ch.fresh()
        last, rows = ch.prefill(ch.reqs, ids, 0, len(ids), need=hd.need(ro))
        ch.mem.sample()
        out[m] = (hd.run(ro, last, rows), hd.raw(ro, last, rows))
    return out, {}


def mode_SP(ch, hd, passes, P, members, ctx):
    """S with the span cut at the fork prefix: cold, but the same span boundaries every fork mode
    uses, so SP vs S isolates the chunking effect and a fork mode vs SP isolates the fork itself."""
    out = {}
    for m in members:
        ro = passes[m]["readout"]
        ids = ro["prompt_ids"]
        ch.fresh()
        last, rows = ch.prefill(ch.reqs, ids, 0, len(ids), need=hd.need(ro),
                                stop_at=P if len(members) > 1 else None)
        ch.mem.sample()
        out[m] = (hd.run(ro, last, rows), hd.raw(ro, last, rows))
    return out, {}


def _save_host(ch):
    return [np.array(vs.state.data, copy=True) for vs in ch.states(ch.reqs)]


def _restore_host(ch, saved):
    ch.restored = True
    for vs, s in zip(ch.states(ch.reqs), saved):
        vs.state = ch.ov.Tensor(s)


def _save_remote(ch, api):
    out = []
    for vs in ch.states(ch.reqs):
        st = vs.state
        api.setdefault("vs.state type", type(st).__name__)
        rt = ch.ctx.create_tensor(st.get_element_type(), st.get_shape(), {})
        api.setdefault("remote tensor type", type(rt).__name__)
        rt.copy_from(st)
        out.append(rt)
    return out


def _restore_remote(ch, saved):
    ch.restored = True
    for vs, rt in zip(ch.states(ch.reqs), saved):
        vs.state = rt


def _engine_restore(ch, saved):
    """HEAD's warm admission (qwen36.rs finish_decision -> reset_all, then restore_local_parts):
    the previous pass restored, so reset_all recreates the requests; an un-run request drops a
    set_state, so one priming fold (zeros at position 0) runs first; then the state is set."""
    ch.reqs = ch.new_requests()
    ch.chunk(ch.reqs, None, 0, emb=np.zeros((1, 1, ch.width), dtype=np.float32))
    _restore_host(ch, saved)


def _fork_host_like(ch, hd, passes, P, members, save, restore):
    """H and R: today's engine order. The first pass prefills to P, snapshots, carries on; every
    later pass restores the snapshot and prefills its suffix."""
    out, copy_ms = {}, []
    first = passes[members[0]]["readout"]
    ch.fresh()
    ch.prefill(ch.reqs, first["prompt_ids"], 0, P)
    t = time.perf_counter()
    saved = save()
    snap_ms = (time.perf_counter() - t) * 1e3
    for k, m in enumerate(members):
        ro = passes[m]["readout"]
        ids = ro["prompt_ids"]
        if k > 0:
            t = time.perf_counter()
            restore(saved)
            copy_ms.append((time.perf_counter() - t) * 1e3)
        last, rows = ch.prefill(ch.reqs, ids, P, len(ids), need=hd.need(ro))
        ch.mem.sample()
        out[m] = (hd.run(ro, last, rows), hd.raw(ro, last, rows))
    del saved
    return out, {"copy_ms": copy_ms, "snapshot_ms": snap_ms}


def mode_H(ch, hd, passes, P, members, ctx):
    if len(members) < 2:
        return mode_S(ch, hd, passes, P, members, ctx)
    return _fork_host_like(ch, hd, passes, P, members, lambda: _save_host(ch),
                           lambda s: _restore_host(ch, s))


def mode_HE(ch, hd, passes, P, members, ctx):
    """H as HEAD runs it: recreate + priming fold before every restore."""
    if len(members) < 2:
        return mode_S(ch, hd, passes, P, members, ctx)
    return _fork_host_like(ch, hd, passes, P, members, lambda: _save_host(ch),
                           lambda s: _engine_restore(ch, s))


def mode_R(ch, hd, passes, P, members, ctx):
    if len(members) < 2:
        return mode_S(ch, hd, passes, P, members, ctx)
    return _fork_host_like(ch, hd, passes, P, members, lambda: _save_remote(ch, ctx["api"]),
                           lambda s: _restore_remote(ch, s))


def mode_P(ch, hd, passes, P, members, ctx):
    if len(members) < 2:
        return mode_S(ch, hd, passes, P, members, ctx)
    pool = ctx["pool"]
    while len(pool) < len(members):
        pool.append(ch.new_requests())
    first = passes[members[0]]["readout"]
    ch.fresh()
    ch.prefill(ch.reqs, first["prompt_ids"], 0, P)
    out, copy_ms = {}, []
    src = ch.states(ch.reqs)
    for k, m in enumerate(members):
        reqs = pool[k]
        t = time.perf_counter()
        for vs_j, vs_0 in zip(ch.states(reqs), src):
            vs_j.state = vs_0.state
        copy_ms.append((time.perf_counter() - t) * 1e3)
    for k, m in enumerate(members):
        ro = passes[m]["readout"]
        ids = ro["prompt_ids"]
        last, rows = ch.prefill(pool[k], ids, P, len(ids), need=hd.need(ro))
        ch.mem.sample()
        out[m] = (hd.run(ro, last, rows), hd.raw(ro, last, rows))
    return out, {"copy_ms": copy_ms}


def _batch_prefix(ch, ids, P, q, dup):
    """The shared prefix for a batch fork. dup=False: prefill it once at batch 1; the first
    suffix step then expands the state with beam_idx=[0]*q (the fork). dup=True: prefill q
    identical rows at batch q, so no beam_idx expansion ever runs (no state sharing: a probe of
    whether batching alone pays). Returns the first suffix step's beam_idx."""
    if not dup:
        ch.prefill(ch.reqs, ids, 0, P)
        return np.zeros(q, dtype=np.int32)
    p = 0
    while p < P:
        e = min(p + SPAN, P)
        ch.chunk(ch.reqs, np.array([ids[p:e]] * q, dtype=np.int64), p)
        p = e
    return np.arange(q, dtype=np.int32)


def mode_B(ch, hd, passes, P, members, ctx, mask_pad=True, dup=False):
    """Right-padded batch. Needs the full final_hidden rows: on a last_logits_only tree with a
    verbalizer head the logits row of a padded row is its padding's, so the mode does not apply."""
    if len(members) < 2:
        return mode_S(ch, hd, passes, P, members, ctx)
    ros = [passes[m]["readout"] for m in members]
    if any(r["head"]["kind"] != "pointer" for r in ros):
        raise NotApplicable("verbalizer head on a last_logits_only tree: a padded row's last "
                            "position is padding")
    fan = ctx.get("fanout") or 0
    if fan > len(ros):  # memory/time probe at a larger q: repeat the rows, score the originals
        ros = (ros * (fan // len(ros) + 1))[:fan]
    q = len(ros)
    lens = [len(r["prompt_ids"]) - P for r in ros]
    L = max(lens)
    ch.fresh()
    beam0 = _batch_prefix(ch, ros[0]["prompt_ids"], P, q, dup)
    ids = np.zeros((q, L), dtype=np.int64)
    for i, r in enumerate(ros):
        suf = r["prompt_ids"][P:]
        ids[i, :len(suf)] = suf
        ids[i, len(suf):] = suf[-1]
    rows = [dict() for _ in range(q)]
    p = 0
    while p < L:
        e = min(p + SPAN, L)
        mask = np.ones((q, P + e), dtype=np.int64)
        if mask_pad:
            for i, n in enumerate(lens):
                mask[i, P + n:] = 0
        beam = beam0 if p == 0 else np.arange(q, dtype=np.int32)
        _, fh = ch.chunk(ch.reqs, ids[:, p:e], P + p, mask=mask, beam=beam, want_fh=True)
        for i, r in enumerate(ros):
            for pos in hd.need(r):
                if P + p <= pos < P + e:
                    rows[i][pos] = fh[i, pos - P - p].copy()
        p = e
    ch.mem.sample()
    out = {m: (hd.run(r, None, rows[i]), hd.raw(r, None, rows[i]))
           for i, (m, r) in enumerate(zip(members, ros))}  # zip stops at the originals
    return out, {"batch": q, "pad_tokens": int(sum(L - n for n in lens))}


def mode_BR(ch, hd, passes, P, members, ctx, shrink=True, dup=False):
    """Ragged batch: steps end at each distinct suffix length, so every row's last position is a
    real token. shrink=True (BR): finished rows leave through beam_idx, so no padding is ever fed.
    shrink=False (BK): the batch keeps every row; a finished row is fed filler (its last token)
    after its answer was read, which cannot reach any position it reads."""
    if len(members) < 2:
        return mode_S(ch, hd, passes, P, members, ctx)
    ros = [passes[m]["readout"] for m in members]
    lens = [len(r["prompt_ids"]) - P for r in ros]
    ch.fresh()
    beam = _batch_prefix(ch, ros[0]["prompt_ids"], P, len(ros), dup)
    want_fh = any(r["head"]["kind"] == "pointer" for r in ros)
    active = list(range(len(ros)))  # row indices into ros, in the current state's batch order
    lasts, rows = [None] * len(ros), [dict() for _ in ros]
    p, steps = 0, 0
    for stop in sorted(set(lens)):
        while p < stop:
            e = min(p + SPAN, stop)
            ids = np.array([(ros[i]["prompt_ids"] + [ros[i]["prompt_ids"][-1]] * SPAN * 64)[P + p:P + e]
                            for i in active], dtype=np.int64)
            try:
                out0, fh = ch.chunk(ch.reqs, ids, P + p, beam=beam, want_fh=want_fh)
            except Exception as ex:
                raise RuntimeError(f"step {steps} rows {len(active)} beam {beam.tolist()} "
                                   f"span [{P + p},{P + e}): {ex}") from ex
            steps += 1
            for b, i in enumerate(active):
                for pos in hd.need(ros[i]):
                    if P + p <= pos < P + e:
                        rows[i][pos] = fh[b, pos - P - p].copy()
                if lens[i] == e:
                    lasts[i] = out0[b, -1].copy()
            p = e
            beam = np.arange(len(active), dtype=np.int32)
        ch.mem.sample()
        if not shrink:
            continue
        keep = [b for b, i in enumerate(active) if lens[i] > stop]
        beam = np.array(keep, dtype=np.int32)
        active = [active[b] for b in keep]
    out = {m: (hd.run(r, lasts[i], rows[i]), hd.raw(r, lasts[i], rows[i]))
           for i, (m, r) in enumerate(zip(members, ros))}
    return out, {"batch": len(ros), "steps": steps}


def mode_BF(ch, hd, passes, P, members, ctx):
    """B with filler instead of masked padding: the short rows are right-filled with their last
    token and attention_mask stays 1, so the pad is ordinary later tokens, which a causal model
    cannot let reach an earlier position (B's masked pad does: NaN, see the report)."""
    return mode_B(ch, hd, passes, P, members, ctx, mask_pad=False)


def mode_BK(ch, hd, passes, P, members, ctx):
    return mode_BR(ch, hd, passes, P, members, ctx, shrink=False)


def mode_BDF(ch, hd, passes, P, members, ctx):
    return mode_B(ch, hd, passes, P, members, ctx, mask_pad=False, dup=True)


def mode_BDK(ch, hd, passes, P, members, ctx):
    return mode_BR(ch, hd, passes, P, members, ctx, shrink=False, dup=True)


class NotApplicable(Exception):
    pass


MODES = {"S": mode_S, "SP": mode_SP, "H": mode_H, "HE": mode_HE, "R": mode_R, "B": mode_B, "BR": mode_BR, "BK": mode_BK, "BF": mode_BF, "BDF": mode_BDF, "BDK": mode_BDK, "P": mode_P}


# ------------------------------------------------------------------------------------------------
# Commands


def cmd_analyze(a):
    cases = load_jsonl(a.cases)
    al = align(load_jsonl(a.dump), cases)
    sizes, sib, tot, cons_hit, plens, slens, strategies = Counter(), 0, 0, 0, [], [], Counter()
    for c, passes in al:
        for P, mem in fork_groups(a.family, passes):
            sizes[len(mem)] += 1
            tot += len(mem)
            if len(mem) > 1:
                sib += len(mem)
                plens.append(P)
                slens += [len(passes[m]["readout"]["prompt_ids"]) - P for m in mem]
        h, _ = consecutive_sibling_share(passes)
        cons_hit += h
        for p in passes:
            strategies[p.get("strategy", "-")] += 1
    qtypes = Counter(q["type"] for c in cases for q in c["body"]["questions"].values())
    lens = [len(p["readout"]["prompt_ids"]) for _, ps in al for p in ps]
    print(json.dumps({
        "family": a.family, "cases": len(al), "passes": tot, "question_types": qtypes,
        "strategies": strategies, "group_sizes": dict(sorted(sizes.items())),
        "passes_with_sibling": sib, "share_with_sibling": round(sib / tot, 3),
        "consecutive_sibling_share": round(cons_hit / tot, 3),
        "prompt_len": [min(lens), int(np.median(lens)), max(lens)],
        "prefix_len": [min(plens), int(np.median(plens)), max(plens)] if plens else None,
        "suffix_len": [min(slens), int(np.median(slens)), max(slens)] if slens else None,
    }, default=dict))


def cmd_run(a):
    cases = load_jsonl(a.cases)
    al = align(load_jsonl(a.dump), cases)
    if a.limit:
        al = al[:a.limit]
    cache_dir = a.cache_dir
    if cache_dir is None and os.environ.get("LOCALAPPDATA"):
        cache_dir = os.path.join(os.environ["LOCALAPPDATA"], "cascadia", "ov-cache")  # the server's default
    ch = Chain(a.tree, cache_dir=cache_dir)
    ch.recreate_after_restore = not a.no_recreate
    hd = Heads(a.family, a.tree, ch.width)
    log = lambda *x: print(*x, flush=True)
    log(f"SPIKE compiled in {ch.compile_s:.1f}s width={ch.width} last_only={ch.last_only} "
        f"final_hidden={ch.has_fh} gpu_bytes={ch.mem.gpu()} private={proc_private()}")
    api = {}
    try:
        ch.ctx = ch.core.get_default_context("GPU")
        api["core.get_default_context('GPU')"] = f"ok: {type(ch.ctx).__name__}"
    except Exception as e:
        api["core.get_default_context('GPU')"] = f"FAILED: {e}"
    out = open(a.out, "w", encoding="utf-8")
    ref = {}  # case id -> {member: (logits, raw)}
    groups = [(c, passes, fork_groups(a.family, passes)) for c, passes in al]
    base_gpu, base_priv = ch.mem.gpu(), proc_private()
    for mode in a.modes.replace("+", ",").split(","):
        fn = MODES[mode]
        ctx = {"api": api, "pool": [], "fanout": a.fanout}
        ch.mem.reset()
        # Warm-up on the first case (dynamic-shape kernels compile on first sight), untimed.
        try:
            for P, mem in groups[0][2]:
                fn(ch, hd, groups[0][1], P, mem, ctx)
        except NotApplicable as e:
            log(f"SPIKE mode {mode} not applicable: {e}")
            out.write(json.dumps({"mode": mode, "not_applicable": str(e)}) + "\n")
            continue
        except Exception as e:
            log(f"SPIKE mode {mode} FAILED in warm-up: {type(e).__name__}: {e}")
            api[f"mode {mode}"] = f"FAILED: {type(e).__name__}: {e}"
            out.write(json.dumps({"mode": mode, "failed": f"{type(e).__name__}: {e}"}) + "\n")
            continue
        ch.mem.reset()
        t_mode = time.perf_counter()
        for ci, (c, passes, gs) in enumerate(groups):
            res, extras, t = {}, [], time.perf_counter()
            for P, mem in gs:
                r, x = fn(ch, hd, passes, P, mem, ctx)
                res.update(r)
                extras.append(x)
            wall = time.perf_counter() - t
            rec = {"mode": mode, "case": c["id"], "wall_s": wall, "passes": len(passes),
                   "groups": [[P, mem] for P, mem in gs], "extras": extras,
                   "logits": {m: res[m][0] for m in sorted(res)}}
            if mode == "S":
                ref[c["id"]] = {m: res[m][1] for m in res}
            elif c["id"] in ref:
                rec["raw_max_abs"] = max(float(np.max(np.abs(res[m][1] - ref[c["id"]][m]))) for m in res)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            log(f"SPIKE {mode} case {ci + 1}/{len(groups)} {c['id']} {wall:.3f}s "
                f"raw_max_abs={rec.get('raw_max_abs', '-')}")
        mode_s = time.perf_counter() - t_mode
        mrec = {"mode": mode, "summary": True, "mode_s": mode_s,
                "peak_gpu_bytes": ch.mem.max_gpu, "peak_private_bytes": ch.mem.max_priv,
                "base_gpu_bytes": base_gpu, "base_private_bytes": base_priv,
                "pool_requests": len(ctx["pool"]) * len(ch.stages)}
        out.write(json.dumps(mrec) + "\n")
        log(f"SPIKE mode {mode} done in {mode_s:.1f}s peak_gpu={ch.mem.max_gpu} "
            f"peak_private={ch.mem.max_priv}")
        ctx["pool"].clear()
    out.write(json.dumps({"api": api}) + "\n")
    out.close()
    log("SPIKE DONE " + json.dumps(api))


def cmd_report(a):
    """Merge result files (each holds its own S run) into parity, speed and memory tables. Every
    mode is compared with the S of its own file (same process, same GPU state)."""
    cases = {c["id"]: c for c in load_jsonl(a.cases)}
    codec = Codec(a.family, a.tree)
    dump = {c["id"]: ps for c, ps in align(load_jsonl(a.dump), list(cases.values()))}

    def pass_probs(cid, rec):
        """{(member, qid): probs} through the one shared codec function."""
        out = {}
        for m, logits in rec["logits"].items():
            for qid, s0, e0 in dump[cid][int(m)]["parts"]:
                out[(int(m), qid)] = codec.probs(cases[cid]["body"]["questions"][qid], logits[s0:e0])
        return out

    def compare(rs, ref):
        dp, flips = 0.0, []
        for cid, r in rs.items():
            for k, p in pass_probs(cid, r).items():
                s_ = ref[cid][k]
                if not np.all(np.isfinite(p)):
                    dp = float("inf")  # a NaN head output: counted as a failure and a flip
                    flips.append(round(margin(s_), 4))
                    continue
                dp = max(dp, float(np.max(np.abs(p - s_))))
                if int(np.argmax(p)) != int(np.argmax(s_)):
                    flips.append(round(margin(s_), 4))
        return dp, flips

    def nan_passes(rs):
        return sum(any(not math.isfinite(x) for x in v) for r in rs.values() for v in r["logits"].values())

    files, api, na = [], {}, {}
    for path in a.results:
        by, summ = {}, {}
        for r in load_jsonl(path):
            if "api" in r:
                api.update(r["api"])
            elif r.get("summary"):
                summ[r["mode"]] = r
            elif "not_applicable" in r or "failed" in r:
                na[r["mode"]] = r.get("not_applicable") or "FAILED " + r["failed"]
            elif "case" in r:
                by.setdefault(r["mode"], {})[r["case"]] = r
        files.append((path, by, summ))
    S0 = files[0][1]["S"]
    ref0 = {cid: pass_probs(cid, r) for cid, r in S0.items()}
    # What HEAD runs: Kev restores through the host in HEAD's order (HE; H when HE was not run),
    # JEV passes are cold (S).
    head_like = "S"
    if a.family == "kev":
        head_like = "HE" if any("HE" in by for _, by, _ in files) else "H"
    hl = None
    for _, by, _ in files:
        if head_like in by:
            hl = {c: by[head_like][c]["wall_s"] / by["S"][c]["wall_s"] for c in by[head_like]}
    rows, s_runs = {}, []
    for path, by, summ in files:
        S = by["S"]
        ref = {cid: pass_probs(cid, r) for cid, r in S.items()}
        s_runs.append(float(np.median([r["wall_s"] for r in S.values()])))
        dpS, flS = compare(S, ref0)
        rows.setdefault("S", {"run_to_run_vs_first_S": {"max_dp": dpS, "flips": flS}})
        rows["S"].setdefault("median_case_s_per_file", []).append(s_runs[-1])
        for mode, rs in by.items():
            if mode == "S":
                continue
            dp, flips = compare(rs, ref)
            ref_sp = None
            if "SP" in by and mode != "SP":
                refsp = {cid: pass_probs(cid, r) for cid, r in by["SP"].items()}
                ref_sp = compare(rs, refsp)
            raw = max((r.get("raw_max_abs", 0.0) for r in rs.values()), default=0.0)
            ratio = [rs[c]["wall_s"] / S[c]["wall_s"] for c in rs]
            saved = [S[c]["wall_s"] - rs[c]["wall_s"] for c in rs]
            cms = [x for r in rs.values() for e in r["extras"] for x in e.get("copy_ms", [])]
            sm, smS = summ.get(mode, {}), summ.get("S", {})
            vs_head = [ratio[i] / hl[c] for i, c in enumerate(rs)] if hl else None
            rows[mode] = {
                "file": os.path.basename(path), "cases": len(rs), "nan_passes": nan_passes(rs),
                "max_dp": dp, "flips_S_margin": flips, "raw_max_abs": raw,
                "vs_SP": {"max_dp": ref_sp[0], "flips": ref_sp[1]} if ref_sp else None,
                "viable": dp <= 0.05 and all(m < 0.05 for m in flips) and len(rs) == len(S0),
                "median_case_s": float(np.median([r["wall_s"] for r in rs.values()])),
                "median_S_case_s": float(np.median([r["wall_s"] for r in S.values()])),
                "median_ratio_vs_S": float(np.median(ratio)),
                "median_saved_s_vs_S": float(np.median(saved)),
                "median_ratio_vs_HEAD_like": float(np.median(vs_head)) if vs_head else None,
                "copy_ms_per_question_median": float(np.median(cms)) if cms else None,
                "copy_ms_per_question_p90": float(np.percentile(cms, 90)) if cms else None,
                "peak_gpu_over_S_bytes": (sm.get("peak_gpu_bytes", 0) - smS.get("peak_gpu_bytes", 0)) if sm else None,
                "peak_private_over_S_bytes": (sm.get("peak_private_bytes", 0) - smS.get("peak_private_bytes", 0)) if sm else None,
                "peak_gpu_bytes": sm.get("peak_gpu_bytes"), "S_peak_gpu_bytes": smS.get("peak_gpu_bytes"),
                "pool_requests": sm.get("pool_requests"),
            }
    def server_check(refp):
        dp, n, flips, skipped = 0.0, 0, [], 0
        for line in load_jsonl(a.server):
            cid = line["id"]
            if cid not in refp:
                continue
            ppq = {}
            for (m, qid), p in refp[cid].items():
                ppq.setdefault(qid, []).append(p)
            for qid, ans in line["answers"].items():
                if len(ppq.get(qid, [])) != 1:
                    skipped += 1
                    continue
                sp = [1 - ans["noul"], ans["noul"]] if ans["type"] == "noul" else list(ans["probabilities"].values())
                mine = ppq[qid][0]
                dp = max(dp, float(np.max(np.abs(np.asarray(sp) - mine))))
                n += 1
                if int(np.argmax(sp)) != int(np.argmax(mine)):
                    flips.append(round(margin(sp), 4))
        return {"questions": n, "max_dp": dp, "flips_server_margin": flips, "skipped_multi_pass": skipped}

    server = {}
    if a.server:
        # The Step 1 server ran with the prefix cache on: Kev's first pass cuts its span at the
        # shared prefix and later passes restore it (= SP/H here); JEV passes are cold (= S).
        server["S"] = server_check(ref0)
        for _, by, _ in files:
            for mode in ("SP", "H", "HE"):
                if mode in by and mode not in server:
                    server[mode] = server_check({c: pass_probs(c, r) for c, r in by[mode].items()})
    proj = {}
    if a.baseline_p50:
        for mode, r in rows.items():
            if mode == "S" or not r.get("viable"):
                continue
            proj[mode] = {
                "brief_formula_p50_s": a.baseline_p50 * r["median_ratio_vs_S"],
                "head_matched_p50_s": a.baseline_p50 * r["median_ratio_vs_HEAD_like"] if r["median_ratio_vs_HEAD_like"] else None,
            }
            for k in list(proj[mode]):
                if proj[mode][k] is not None:
                    proj[mode][k.replace("_s", "_change_pct")] = 100 * (proj[mode][k] / a.baseline_p50 - 1)
    print(json.dumps({"family": a.family, "head_like_mode": head_like, "baseline_p50_s": a.baseline_p50,
                      "modes": rows, "server_check_S": server, "projection": proj,
                      "not_applicable_or_failed": na, "api": api}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("analyze", "run", "report"):
        s = sub.add_parser(name)
        s.add_argument("--family", choices=["kev", "jev"], required=True)
        s.add_argument("--dump", required=True)
        s.add_argument("--cases", required=True)
        s.add_argument("--tree", required=name != "analyze")
        if name == "run":
            s.add_argument("--out", required=True)
            s.add_argument("--modes", default="S,H,R,B,BR,P")
            s.add_argument("--limit", type=int)
            s.add_argument("--cache-dir")
            s.add_argument("--no-recreate", action="store_true",
                           help="reset_state even after a restore (the engine recreates instead)")
            s.add_argument("--fanout", type=int, default=0,
                           help="B only: repeat each group's rows to this batch (memory probe)")
        if name == "report":
            s.add_argument("--results", required=True, nargs="+")
            s.add_argument("--server")
            s.add_argument("--baseline-p50", type=float, help="HEAD p50 (s) from the Step 1 run file")
    a = ap.parse_args(argv)
    {"analyze": cmd_analyze, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
