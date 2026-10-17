"""IR-level check of the LM-head cut on a device: outputs and the head's share of the last stage.

  # Before cutting: does the device already narrow the head? (Task 48 step 1)
  python tools/decision/head_cut_check.py --profile-only --old TREE/stage7/stage.xml --device GPU
  # After cut_lm_head.py: old (backup) vs new (cut) last stage
  python tools/decision/head_cut_check.py --old TREE/stage7/stage.xml.pre-cut --new TREE/stage7/stage.xml --device GPU

Each IR is compiled on --device with the server's default precision (no config beyond
PERF_COUNT=YES) and fed what the engine's chain_pass_inner feeds a last stage on a cold pass:
stage_hidden [1, T, width] random normal x 0.5 (seed 0), a dummy embeds input (zeros) when the
stage has one, attention_mask ones [1, T], position_ids [rows, 1, T] = arange, beam_idx [0].
Every T starts from a fresh state (reset_state()).

Prints, for T in --tokens (default 1, 7, 256): the logits max |diff| (old vs new) and the
final_hidden max |diff| (trees with that output). Then, at the profile T (default 256), each IR's
summed profiling real_time of the head's nodes (names containing "lm_head", plus the grafted LoRA
branch found structurally; an executed node counts when any original node fused into it does),
the stage total, the head's share, the wall-clock median and the slowest nodes, marked [head].

Exit codes: 0 pass; 1 a gate failed (compare mode: logits max |diff| >= 0.05 at the profile T,
final_hidden not identical at T < 256 or >= 1e-3 at the profile T, head time fell < 90%);
3 (--profile-only) the head is already under --narrowed-pct (2%) of the stage: the device
already narrows it, so the cut would buy nothing.
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "qwen36_surgery"))
import weight_swap as ws  # noqa: E402

MROPE_ROWS = 4  # crates/cascadia-engine-openvino/src/qwen36.rs


def bin_for(xml):
    if xml.endswith(".xml"):
        return xml[:-4] + ".bin"
    if xml.endswith(".xml.pre-cut"):
        return xml[:-len(".xml.pre-cut")] + ".bin.pre-cut"
    raise SystemExit(f"{xml}: expected a .xml or .xml.pre-cut path")


def head_node_names(model):
    """Friendly names of the LM head's nodes: the lm_head MatMul, the cut Slice, and every node of
    a grafted lm_head LoRA branch (the MatMuls reading the head's input, up to the Add)."""
    mm = ws._lm_head_matmul(model)
    names = {mm.get_friendly_name()}
    src = mm.input_value(0)
    if src.get_node().get_friendly_name() == ws.LM_HEAD_CUT:
        names.add(ws.LM_HEAD_CUT)
    for t in src.get_target_inputs():
        node = t.get_node()
        if node.get_type_name() != "MatMul" or node is mm:
            continue
        # x A^T -> (x A^T) B^T -> Add(.., head)
        names.add(node.get_friendly_name())
        for t2 in node.output(0).get_target_inputs():
            n2 = t2.get_node()
            names.add(n2.get_friendly_name())
            for t3 in n2.output(0).get_target_inputs():
                if t3.get_node().get_type_name() == "Add":
                    names.add(t3.get_node().get_friendly_name())
    return names


def feeds(model, T, width, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    out = {}
    for inp in model.inputs:
        nm = inp.get_any_name()
        et = inp.get_element_type().to_dtype()
        if nm == "stage_hidden":
            out[nm] = (rng.normal(size=(1, T, width)) * 0.5).astype(et)
        elif "embed" in nm:
            out[nm] = np.zeros((1, T, width), dtype=et)
        elif "attention_mask" in nm:
            out[nm] = np.ones((1, T), dtype=et)
        elif "position" in nm:
            ps = inp.get_partial_shape()
            rows = ps[0].get_length() if ps[0].is_static else MROPE_ROWS
            out[nm] = np.broadcast_to(np.arange(T, dtype=et), (rows, 1, T)).copy()
        elif "beam" in nm:
            out[nm] = np.zeros((1,), dtype=et)
        else:
            raise SystemExit(f"unexpected stage input {nm!r} (not what the engine feeds)")
    return out


class Stage:
    def __init__(self, core, xml, device):
        import openvino as ov  # noqa: F401
        self.xml = xml
        self.model = core.read_model(xml, bin_for(xml))
        self.head = head_node_names(self.model)
        width = self.model.input("stage_hidden").get_partial_shape()[2]
        if not width.is_static:
            raise SystemExit(f"{xml}: stage_hidden width is not static")
        self.width = width.get_length()
        t0 = time.time()
        self.compiled = core.compile_model(self.model, device, {"PERF_COUNT": "YES"})
        self.req = self.compiled.create_infer_request()
        print(f"{xml}: compiled on {device} in {time.time() - t0:.0f}s; head nodes {sorted(self.head)}",
              flush=True)
        names = [sorted(o.get_names()) for o in self.compiled.outputs]
        self.fh = next((i for i, n in enumerate(names) if "final_hidden" in n), None)
        # Executed node -> the original nodes fused into it (rt_info originalLayersNames), so a head
        # MatMul fused under another name still counts as head time.
        self.fused = {}
        for op in self.compiled.get_runtime_model().get_ops():
            ri = op.get_rt_info()
            orig = ri["originalLayersNames"].astype(str) if "originalLayersNames" in ri else ""
            self.fused[op.get_friendly_name()] = [n for n in orig.split(",") if n]

    def is_head(self, exec_name):
        names = self.fused.get(exec_name) or [exec_name]
        return any("lm_head" in n or n in self.head for n in names + [exec_name])

    def run(self, T):
        self.req.reset_state()
        t0 = time.perf_counter()
        self.req.infer(feeds(self.model, T, self.width))
        dt = time.perf_counter() - t0
        logits = self.req.get_output_tensor(0).data.copy()
        fh = self.req.get_output_tensor(self.fh).data.copy() if self.fh is not None else None
        return logits, fh, dt

    def profile(self, T, reps):
        self.run(T)  # warm-up
        walls, heads, totals, top = [], [], [], {}
        for _ in range(reps):
            _, _, dt = self.run(T)
            walls.append(dt * 1000)
            head = total = 0.0
            for p in self.req.profiling_info:
                ms = p.real_time.total_seconds() * 1000
                total += ms
                if self.is_head(p.node_name):
                    head += ms
                top[p.node_name] = top.get(p.node_name, 0.0) + ms / reps
            heads.append(head)
            totals.append(total)
        return statistics.median(walls), statistics.median(heads), statistics.median(totals), top


def print_profile(label, st, T, reps, show_top):
    wall, head, total, top = st.profile(T, reps)
    share = 100 * head / total if total else float("nan")
    print(f"{label} T={T}: lm_head {head:.2f} ms of {total:.2f} ms profiled ({share:.1f}%), "
          f"wall median {wall:.2f} ms over {reps}", flush=True)
    for name, ms in sorted(top.items(), key=lambda kv: -kv[1])[:show_top]:
        mark = " [head]" if st.is_head(name) else ""
        print(f"    {ms:9.3f} ms  {name}{mark}")
    return head, total, share


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="last-stage IR before the cut (stage.xml or stage.xml.pre-cut)")
    ap.add_argument("--new", help="last-stage IR after the cut")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--tokens", default="1,7,256", help="comma-separated T values for the output comparison")
    ap.add_argument("--profile-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5, help="profiled runs per IR (after one warm-up)")
    ap.add_argument("--top", type=int, default=12, help="print the N slowest profiled nodes per IR")
    ap.add_argument("--profile-only", action="store_true", help="profile --old alone (is the head already narrowed?)")
    ap.add_argument("--narrowed-pct", type=float, default=2.0)
    a = ap.parse_args(argv)
    import openvino as ov
    core = ov.Core()
    print(f"openvino {ov.__version__}, device {a.device}", flush=True)
    T = a.profile_tokens

    if a.profile_only:
        old = Stage(core, a.old, a.device)
        _, _, share = print_profile("old", old, T, a.reps, a.top)
        if share < a.narrowed_pct:
            print(f"VERDICT: head is {share:.1f}% < {a.narrowed_pct}% of the last stage: the device already "
                  "narrows it; the cut would buy nothing. STOP.")
            return 3
        print(f"VERDICT: head is {share:.1f}% of the last stage (>= {a.narrowed_pct}%): the cut is worth trying.")
        return 0

    if not a.new:
        raise SystemExit("--new is required unless --profile-only")
    old, new = Stage(core, a.old, a.device), Stage(core, a.new, a.device)
    fails = []
    for t in [int(x) for x in a.tokens.split(",")]:
        lo, fo, _ = old.run(t)
        ln, fn, _ = new.run(t)
        if ln.shape != lo.shape:
            fails.append(f"T={t}: logits shape {ln.shape} != {lo.shape}")
            continue
        dl = float(np.abs(ln.astype(np.float64) - lo).max())
        same_top = int(np.argmax(lo.reshape(-1))) == int(np.argmax(ln.reshape(-1)))
        msg = f"T={t}: logits max|d| {dl:.6f} (argmax {'same' if same_top else 'DIFFERS'})"
        if t == T and dl >= 0.05:
            fails.append(f"T={t}: logits max|d| {dl:.6f} >= 0.05")
        if fo is not None and fn is not None:
            df = float(np.abs(fn.astype(np.float64) - fo).max())
            msg += f", final_hidden max|d| {df:.6g}"
            if t < T and df != 0.0:
                fails.append(f"T={t}: final_hidden max|d| {df:.6g} != 0")
            if t == T and df >= 1e-3:
                fails.append(f"T={t}: final_hidden max|d| {df:.6g} >= 1e-3")
        elif (fo is None) != (fn is None):
            fails.append(f"T={t}: final_hidden present in one IR only")
        print(msg, flush=True)
    ho, _, so = print_profile("old", old, T, a.reps, a.top)
    hn, _, sn = print_profile("new", new, T, a.reps, a.top)
    fall = 100 * (1 - hn / ho) if ho else float("nan")
    print(f"lm_head time at T={T}: {ho:.2f} -> {hn:.2f} ms ({fall:.1f}% less); share {so:.1f}% -> {sn:.1f}%")
    if not fall >= 90:
        fails.append(f"lm_head time fell {fall:.1f}% < 90%")
    for f in fails:
        print(f"FAIL: {f}")
    print("VERDICT: " + ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
