#!/usr/bin/env python3
"""Conformance checks for the --elastic memory posture (Linux).

Runs the checks below against a built `cascadia` binary:

  C2  output identity      stock vs --elastic, temperature 0 -> byte-identical
  C5  elasticity witness   the big allocations leave the anonymous set
                           (RssAnon collapses vs stock; file-backed grows)
  C3  pressure survival    optional (--pressure-mb N): serve correct text
                           under a cgroup MemoryMax cap with swap off

Guards that keep a result meaningful:

  * gate line - the elastic leg's log must contain "elastic posture active".
    Without it the run is invalid (hook off), not a pass.
  * backing dir - ELASTIC_DIR must be on a disk-backed filesystem. On tmpfs
    the pages cannot be written back, so the posture gives no survival
    benefit under a memory cap even though RssAnon still collapses
    (measured 2026-09-22, B70: tmpfs backing dies at the same caps as stock).
  * cgroup limit - the pressure leg verifies memory.max actually equals the
    requested cap before trusting the result.

Optional legs completing the protocol table:

  C1  floor independence   optional (--scale-model M2): settled committed
                           memory lands in one band at two model scales
                           >=8x apart (file-size ratio, fail-closed)
  C4  co-tenancy           optional (--coten-models A,B --coten-budget-mb N):
                           N co-resident servers under ONE cgroup MemoryMax
                           (a fleet budget, not a per-tenant cap), each
                           serving its own solo-verified text, at a budget
                           below the sum of their solo footprints so survival
                           proves the requirement does not grow with N.

Engine-independent: the interposer sits below the engine and intercepts libc
allocation, so the same suite applies to any engine that allocates through
malloc. Linux-only as written (/proc + systemd cgroups). The hook itself is
verified on Windows (PR #132, Detours: 1329 -> 223 MB private commit), but a
Windows port of this suite needs a private-bytes witness (psapi) and a Job
Object pressure leg -- not /proc.

Usage:
  export INTEL_OPENVINO_DIR=/path/to/ov-genai-sdk    # or LD_LIBRARY_PATH
  python3 tests/conformance/test_elastic.py \
      --bin target/release/cascadia \
      --model /path/to/model-int4-ov \
      --elastic-dir /disk/backed/backing \
      [--pressure-mb 1024] [--scale-model /path/to/model2] \
      [--coten-models A,B --coten-budget-mb 1536] \
      [--max-tokens 64] [--json-out results.json]

Exit code: 0 = all requested checks passed; 1 = a check failed; 2 = preflight.
"""
import argparse
import glob
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request

DEFAULT_PROMPT = ("Explain, step by step, why the sky appears blue during the "
                  "day and red at sunset.")
GATE_LINE = "elastic posture active"


def log(msg):
    print(f"[conformance] {msg}", flush=True)


class Fail(RuntimeError):
    pass


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def free_port(base, used):
    for p in range(base, base + 100):
        if p in used:
            continue
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
        used.add(p)
        return p
    raise Fail(f"no free port in [{base}, {base + 100})")


def wait_port(port, proc, timeout):
    """'up' | 'died' | 'timeout'."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return "died"
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return "up"
        except OSError:
            time.sleep(0.5)
    return "timeout"


def wait_closed(port, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.5)
    return False


def find_pid(port):
    """Serving pid via ss, falling back to a /proc scan (unprivileged ss
    may hide pid=). The elastic re-exec keeps the pid, but never assume it."""
    try:
        out = sh(["ss", "-tlnpH", f"sport = :{port}"]).stdout
        m = re.search(r"pid=(\d+)", out)
        if m:
            return int(m.group(1))
    except OSError:
        pass
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/comm") as f:
                if "cascadia" not in f.read():
                    continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode(errors="replace")
        except OSError:
            continue
        if f":{port}" in cmd and " run " in cmd.replace("\x00", " "):
            return int(pid)
    return None


def proc_mem(pid):
    """RssAnon/RssFile/RssShmem/VmRSS in MB, or None."""
    out = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("RssAnon", "RssFile", "RssShmem", "VmRSS"):
                    out[k] = round(int(v.split()[0]) / 1024, 1)
    except (OSError, ValueError):
        return None
    return out or None


def chat(port, prompt, max_tokens, timeout=900):
    body = json.dumps({
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return (time.monotonic() - t0, resp["usage"]["completion_tokens"],
            resp["choices"][0]["message"]["content"])


def launch(binpath, model, port, elastic, env, logpath, memmax_mb=None,
           unit=None, min_mb=None, so=None):
    cmd = [binpath, "run", model, "--device", "CPU", "--api", f"0.0.0.0:{port}"]
    if elastic:
        cmd.append("--elastic")
        # Research escape hatches over the CLI's fixed (1 MB, pool on) defaults,
        # driven by env so no binary rebuild is needed: MIN_MB is the big-
        # allocation threshold (the interposer reads ELASTIC_MIN_MB directly)
        # and SO forces the interposer past the re-exec (own-loading) path.
        # Both are env-only: `run` has no --elastic-min-mb flag (that is a
        # `worker` flag), and the env var survives because the forced preload
        # skips the re-exec that would otherwise overwrite it with the default.
        if min_mb:
            env["ELASTIC_MIN_MB"] = str(min_mb)
        if so:
            env["CASCADIA_ELASTIC_ACTIVE"] = "1"
            env["LD_PRELOAD"] = so
    if memmax_mb:
        cap_cmd = ["systemd-run", "--user", "--scope", "--quiet"]
        if unit:
            cap_cmd += ["--unit", unit]
        cmd = cap_cmd + ["-p", f"MemoryMax={memmax_mb}M", "-p", "MemorySwapMax=0"] + cmd
    lf = open(logpath, "w")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                         env=env, start_new_session=True)
    return p, lf


def launch_fleet(binpath, specs, env, logpath, memmax_mb, unit,
                 min_mb=None, so=None):
    """N servers as ONE scope under ONE MemoryMax -- the C4 fleet budget.

    A fleet budget, not a per-tenant cap: survival below the sum of the solo
    footprints is the co-tenancy contrast the reservation posture cannot buy.
    """
    parts = []
    for model, port, elastic in specs:
        c = [binpath, "run", model, "--device", "CPU", "--api", f"0.0.0.0:{port}"]
        if elastic:
            c.append("--elastic")
        parts.append(f"{shlex.join(c)} > {logpath}.{port} 2>&1 &")
    # Env-only escape hatches (see launch()): `run` has no --elastic-min-mb.
    if min_mb:
        env["ELASTIC_MIN_MB"] = str(min_mb)
    if so:
        env["CASCADIA_ELASTIC_ACTIVE"] = "1"
        env["LD_PRELOAD"] = so
    inner = " ".join(parts) + " wait"
    cmd = ["systemd-run", "--user", "--scope", "--quiet", "--unit", unit,
           "-p", f"MemoryMax={memmax_mb}M", "-p", "MemorySwapMax=0",
           "bash", "-c", inner]
    lf = open(logpath, "w")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                         env=env, start_new_session=True)
    return p, lf


def stop_fleet(pf, ports):
    proc, lf = pf
    for port in ports:
        stop_by_pid(find_pid(port))
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    lf.close()
    for port in ports:
        wait_closed(port)


def stop(proc, port):
    stop_by_pid(find_pid(port))
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    wait_closed(port)


def cgroup_path(pid):
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            for line in f:
                if line.startswith("0::"):
                    return line.split("::", 1)[1].strip()
    except OSError:
        pass
    return None


def cgroup_file(pid, name):
    cg = cgroup_path(pid)
    if not cg:
        return None
    try:
        with open(f"/sys/fs/cgroup{cg}/{name}") as f:
            return f.read().strip()
    except OSError:
        return None


def fstype(path):
    try:
        r = sh(["findmnt", "-no", "FSTYPE", "-T", path])
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except OSError:
        pass
    try:
        r = sh(["df", "--output=fstype", path])
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        if r.returncode == 0 and len(lines) >= 2:
            return lines[-1].strip()
    except OSError:
        pass
    return "unknown"


def ldd_missing(binpath, env=None):
    try:
        r = sh(["ldd", binpath], env=env)
    except OSError:
        return []
    return [l.strip() for l in r.stdout.splitlines() if "not found" in l]


def dir_size(path):
    """Total bytes under path (model file scale, for the C1 ratio gate)."""
    n = 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                n += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return n


def drop_cache(paths):
    """Evict model files from the page cache before a capped leg.

    Instrument note: page
    cache charged by a previous leg is reparented when that leg's cgroup dies
    and would be free for the next one -- an optimistic bias that makes low
    limits look survivable. DONTNEED forces every leg to fault its own
    weights in and charge them to its own cgroup.
    """
    for root in paths:
        for r, _, names in os.walk(root):
            for name in names:
                try:
                    fd = os.open(os.path.join(r, name), os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                except OSError:
                    pass


def find_elastic_so(binpath):
    """Built interposer next to the binary (target/release/build/...)."""
    d = os.path.dirname(os.path.abspath(binpath))
    hits = sorted(glob.glob(os.path.join(
        d, "build", "cascadia-elastic-*", "out", "libcascadia_elastic.so")))
    return hits[-1] if hits else None


def stop_by_pid(pid):
    if not pid:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(
        description="Elastic-memory posture conformance checks.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="built cascadia binary")
    ap.add_argument("--model", required=True, help="model directory (int4-ov)")
    ap.add_argument("--elastic-dir", default=None,
                    help="disk-backed backing dir (else $ELASTIC_DIR/$TMPDIR//tmp)")
    ap.add_argument("--pressure-mb", type=int, default=None,
                    help="run the C3 pressure leg under this cgroup cap (MB)")
    ap.add_argument("--scale-model", default=None,
                    help="second model dir for the C1 floor-independence leg")
    ap.add_argument("--coten-models", default=None,
                    help="comma-separated model dirs for the C4 co-tenancy leg")
    ap.add_argument("--coten-budget-mb", type=int, default=None,
                    help="fleet MemoryMax for the C4 leg (must exceed the sum "
                         "of per-model floors; see the guard)")
    ap.add_argument("--elastic-min-mb", type=int, default=None,
                    help="override the elastic big-alloc threshold (MB) for "
                         "C1's elastic legs; CLI default is 1")
    ap.add_argument("--elastic-so", default=None,
                    help="force the interposer .so via LD_PRELOAD (skips the "
                         "binary's re-exec); default: auto-discover near --bin")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--settle", type=float, default=6.0,
                    help="seconds to let allocations settle before reading memory")
    ap.add_argument("--port-base", type=int, default=8330)
    ap.add_argument("--json-out", default=None, help="write the result JSON here")
    a = ap.parse_args()

    results = {"suite": "cascadia-elastic-conformance", "protocol":
               "(C2/C5, C1/C4/C3 optional)",
               "bin": os.path.abspath(a.bin), "model": os.path.abspath(a.model),
               "checks": {}, "pass": False}
    checks = results["checks"]

    def record(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": detail}
        log(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" - {detail}" if detail else ""))
        return ok

    def finish(code):
        results["pass"] = code == 0
        if a.json_out:
            with open(a.json_out, "w") as f:
                json.dump(results, f, indent=2)
            log(f"json: {a.json_out}")
        log("")
        log("=== summary ===")
        for name, c in checks.items():
            log(f"  {'PASS' if c['pass'] else 'FAIL'}  {name}")
        log(f"RESULT: {'PASS' if code == 0 else 'FAIL'}")
        sys.exit(code)

    if not sys.platform.startswith("linux"):
        log("FAIL: this suite is Linux-only (uses /proc and systemd cgroups)")
        finish(2)
    for p, what in ((results["bin"], "binary"), (results["model"], "model")):
        if not os.path.exists(p):
            log(f"FAIL: {what} not found: {p}")
            finish(2)
    # --- child environment -------------------------------------------------
    base_env = dict(os.environ)
    base_env.pop("LD_PRELOAD", None)          # pristine: no ambient preloads
    base_env.pop("CASCADIA_ELASTIC_ACTIVE", None)
    sdk = os.environ.get("INTEL_OPENVINO_DIR")
    if sdk:
        lp = base_env.get("LD_LIBRARY_PATH", "")
        if sdk not in lp:
            extra = f"{sdk}/runtime/lib/intel64:{sdk}/runtime/3rdparty/tbb/lib"
            base_env["LD_LIBRARY_PATH"] = f"{extra}:{lp}" if lp else extra

    missing = ldd_missing(results["bin"], base_env)
    if missing:
        log("FAIL: binary has unresolved shared libraries; export INTEL_OPENVINO_DIR "
            "or LD_LIBRARY_PATH matching the build:")
        for m in missing[:8]:
            log(f"  {m}")
        finish(2)

    so_path = a.elastic_so or find_elastic_so(results["bin"])
    results["elastic_so"] = so_path

    # --- backing-dir guard -------------------------------------------------
    edir = (a.elastic_dir or os.environ.get("ELASTIC_DIR")
            or os.environ.get("TMPDIR") or "/tmp")
    ft = fstype(edir)
    results["elastic_dir"] = edir
    results["elastic_fs"] = ft
    if not os.path.isdir(edir):
        record("guard: backing dir usable", False, f"{edir} is not a directory")
        finish(1)
    if not os.access(edir, os.W_OK):
        record("guard: backing dir usable", False, f"{edir} is not writable")
        finish(1)
    if ft in ("tmpfs", "ramfs"):
        record("guard: backing dir disk-backed", False,
               f"{edir} is on {ft}: pages cannot be written back, so the posture "
               f"gives NO survival benefit under a memory cap (measured: dies at "
               f"the same caps as stock). Point --elastic-dir/ELASTIC_DIR at a "
               f"disk-backed directory (ext4/xfs).")
        finish(1)
    record("guard: backing dir disk-backed", True, f"{edir} ({ft})")

    used = set()
    prompt = DEFAULT_PROMPT

    def free_leg(elastic):
        port = free_port(a.port_base, used)
        tag = "elastic" if elastic else "stock"
        logp = f"/tmp/conformance-{tag}-{port}.log"
        env = dict(base_env)
        if elastic:
            env["ELASTIC_DIR"] = edir
        proc, lf = launch(results["bin"], results["model"], port, elastic, env, logp)
        try:
            t0 = time.monotonic()
            st = wait_port(port, proc, 300)
            if st != "up":
                raise Fail(f"{tag} server {st} before serving (log: {logp})")
            load_s = time.monotonic() - t0
            dt, ntok, text = chat(port, prompt, a.max_tokens)
            time.sleep(a.settle)
            pid = find_pid(port)
            mem = proc_mem(pid) if pid else None
            with open(logp) as f:
                logtxt = f.read()
            log(f"  {tag}: {ntok} tok in {dt:.1f}s, load {load_s:.1f}s, "
                f"pid {pid}, anon {mem and mem.get('RssAnon')} MB")
            return {"text": text, "mem": mem, "gate": GATE_LINE in logtxt,
                    "log": logp, "tok_s": round(ntok / dt, 1) if dt else None}
        finally:
            lf.close()
            stop(proc, port)

    log(f"=== C2/C5 on {results['bin']} + {results['model']} ===")
    stock = elas = None
    try:
        stock = free_leg(False)
        elas = free_leg(True)
    except Fail as e:
        record("C2 output identity", False, f"leg failed: {e}")
        record("C5 elasticity witness", False, f"leg failed: {e}")

    if stock is not None and elas is not None:
        record("gate line (" + GATE_LINE + ")", elas["gate"],
               "found" if elas["gate"] else f"missing - hook off? see {elas['log']}")

        ident = len(stock["text"]) > 0 and stock["text"] == elas["text"]
        record("C2 output identity", ident,
               f"byte-identical ({len(stock['text'])} chars)" if ident
               else "outputs differ")
        if not ident:
            log(f"  stock  : {stock['text'][:200]!r}")
            log(f"  elastic: {elas['text'][:200]!r}")

        sm, em = stock["mem"], elas["mem"]
        if not sm or not em or sm.get("RssAnon") is None or em.get("RssAnon") is None:
            record("C5 elasticity witness", False,
                   "RssAnon measurement missing (fail-closed)")
        else:
            ratio = em["RssAnon"] / max(sm["RssAnon"], 1)
            ok = ratio <= 0.5
            record("C5 elasticity witness", ok,
                   f"RssAnon {sm['RssAnon']} -> {em['RssAnon']} MB "
                   f"({ratio:.0%} of stock); RssFile {sm.get('RssFile')} -> "
                   f"{em.get('RssFile')} MB")
            if sm["RssAnon"] < 64:
                log("  WARN: stock RssAnon < 64 MB - model too small for a "
                    "meaningful witness")

    # --- C3: pressure survival (optional) ----------------------------------
    if a.pressure_mb and stock is not None:
        cap = a.pressure_mb
        log(f"=== C3: pressure survival under MemoryMax={cap} MB ===")
        ps, pe = free_port(a.port_base, used), free_port(a.port_base, used)

        plog = f"/tmp/conformance-stock-cap-{ps}.log"
        proc, lf = launch(results["bin"], results["model"], ps, False,
                          base_env, plog, memmax_mb=cap)
        try:
            st = wait_port(ps, proc, 300)
        finally:
            lf.close()
            stop(proc, ps)
        with open(plog) as f:
            ptxt = f.read()
        if st == "up":
            log(f"  stock @ {cap} MB: survived the cap - it is above stock's "
                f"knee on this box; the contrast is not demonstrated")
        elif "Failed to" in ptxt:
            record("C3 pressure survival", False,
                   f"systemd-run failed: {ptxt.strip()[-200:]}")
            finish(1)
        else:
            log(f"  stock @ {cap} MB: died before serving (expected below the knee)")

        elog = f"/tmp/conformance-elastic-cap-{pe}.log"
        env = dict(base_env)
        env["ELASTIC_DIR"] = edir
        proc, lf = launch(results["bin"], results["model"], pe, True,
                          env, elog, memmax_mb=cap)
        ok, detail = False, ""
        try:
            st = wait_port(pe, proc, 300)
            if st != "up":
                with open(elog) as f:
                    etxt = f.read()
                detail = f"server {st}"
                if "Failed to" in etxt:
                    detail += f" - systemd-run failed: {etxt.strip()[-160:]}"
            else:
                dt, ntok, text = chat(pe, prompt, a.max_tokens)
                pid = find_pid(pe)
                mmax = cgroup_file(pid, "memory.max") if pid else None
                ev = cgroup_file(pid, "memory.events") if pid else None
                if mmax is None or not mmax.isdigit():
                    detail = ("could not verify cgroup memory.max - limit not "
                              "enforced? (invalid instrument)")
                elif int(mmax) != cap * 1024 * 1024:
                    detail = (f"memory.max={mmax} != requested {cap * 1024 * 1024} "
                              f"(invalid instrument)")
                elif text != stock["text"]:
                    detail = "served, but text differs from the unconstrained run"
                else:
                    m = re.search(r"oom_kill\s+(\d+)", ev or "")
                    if m and int(m.group(1)) > 0:
                        detail = f"oom_kill fired: {(ev or '').strip()[:160]}"
                    else:
                        ok = True
                        detail = (f"served {ntok} tok in {dt:.1f}s under {cap} MB; "
                                  f"memory.max={mmax}")
        except Exception as e:
            detail = f"generation failed: {type(e).__name__}: {e}"
        finally:
            lf.close()
            stop(proc, pe)
        record("C3 pressure survival", ok, detail)

    # --- C1: floor independence (optional, --scale-model) -------------------
    if a.scale_model:
        big = os.path.abspath(a.scale_model)
        s1 = dir_size(results["model"])
        s2 = dir_size(big)
        ratio = max(s1, s2) / max(min(s1, s2), 1)
        log(f"=== C1: floor independence ({results['model']}: {s1 // 2**20} MB "
            f"vs {big}: {s2 // 2**20} MB, {ratio:.1f}x) ===")
        if ratio < 8:
            record("C1 floor independence", False,
                   f"model scales {ratio:.1f}x apart - the protocol requires "
                   f">=8x (invalid instrument); pick a bigger --scale-model")
        else:
            legs = {}
            live = []
            ok, detail = True, ""
            try:
                for tag, mdl in (("small", results["model"]), ("big", big)):
                    port = free_port(a.port_base, used)
                    logp = f"/tmp/conformance-c1-{tag}-{port}.log"
                    env = dict(base_env)
                    env["ELASTIC_DIR"] = edir
                    proc, lf = launch(results["bin"], mdl, port, True, env, logp,
                                      min_mb=a.elastic_min_mb, so=so_path)
                    live.append((proc, lf, port))
                    st = wait_port(port, proc, 900)
                    if st != "up":
                        raise Fail(f"{tag} ({mdl}) {st} before serving (log: {logp})")
                    dt, ntok, text = chat(port, prompt, a.max_tokens)
                    legs[tag] = {"port": port, "text": text, "dt": dt, "ntok": ntok}
                    log(f"  {tag}: {ntok} tok in {dt:.1f}s ({mdl})")
                time.sleep(a.settle)   # both settled under the same conditions
                for tag, (proc, lf, port) in zip(("small", "big"), live):
                    pid = find_pid(port)
                    legs[tag]["pid"] = pid
                    legs[tag]["mem"] = proc_mem(pid) if pid else None
            except Fail as e:
                ok, detail = False, f"leg failed: {e}"
            finally:
                for proc, lf, port in live:
                    lf.close()
                    stop(proc, port)
            if ok:
                f1 = (legs["small"]["mem"] or {}).get("RssAnon")
                f2 = (legs["big"]["mem"] or {}).get("RssAnon")
                if f1 is None or f2 is None:
                    record("C1 floor independence", False,
                           "RssAnon measurement missing (fail-closed)")
                else:
                    band = max(f1, f2) / max(min(f1, f2), 1)
                    ok = band <= 1.5 and f2 < 256
                    record("C1 floor independence", ok,
                           f"committed floor after a settled request: "
                           f"{results['model'].split('/')[-1]} = {f1} MB vs "
                           f"{big.split('/')[-1]} = {f2} MB at {ratio:.1f}x "
                           f"scale (band {band:.2f}x)")
            else:
                record("C1 floor independence", False, detail)

    # --- C4: co-tenancy (optional, --coten-models) --------------------------
    if a.coten_models:
        cmodels = [os.path.abspath(m) for m in a.coten_models.split(",") if m.strip()]
        log(f"=== C4: co-tenancy ({len(cmodels)} models, one fleet budget) ===")
        if len(cmodels) < 2:
            record("C4 co-tenancy", False,
                   "--coten-models needs >=2 model dirs (invalid instrument)")
        else:
            # Solo baselines first: the budget must sit ABOVE their floor sum
            # (or nothing can live) and well BELOW the naive N x solo footprint
            # (or survival demonstrates nothing). The floors are the per-model
            # solo footprints under --elastic.
            env = dict(base_env)
            env["ELASTIC_DIR"] = edir
            floors, solo_text = [], []
            fail = None
            for mdl in cmodels:
                port = free_port(a.port_base, used)
                logp = f"/tmp/conformance-c4-solo-{port}.log"
                proc, lf = launch(results["bin"], mdl, port, True, env, logp,
                                  min_mb=a.elastic_min_mb, so=so_path)
                try:
                    st = wait_port(port, proc, 900)
                    if st != "up":
                        fail = f"solo leg for {mdl} {st} (log: {logp})"
                        break
                    dt, ntok, text = chat(port, prompt, a.max_tokens)
                    time.sleep(a.settle)
                    pid = find_pid(port)
                    m = proc_mem(pid) if pid else None
                    floors.append(m.get("RssAnon") if m else None)
                    solo_text.append(text)
                    log(f"  solo {mdl}: floor {floors[-1]} MB")
                finally:
                    lf.close()
                    stop(proc, port)

            floor_sum = sum(f for f in floors if f is not None) \
                if all(f is not None for f in floors) else None
            if fail is None and floor_sum is None:
                fail = "RssAnon measurement missing (fail-closed)"
            budget = a.coten_budget_mb
            if fail is None and budget is None:
                fail = ("--coten-budget-mb is required with --coten-models "
                        "(fleet budget, one MemoryMax for the whole scope)")
            if fail is None and not (floor_sum * 1.15 < budget <= floor_sum * 3):
                fail = (f"budget {budget} MB is not bracketed by the floors "
                        f"(sum {floor_sum} MB): pick one in "
                        f"(1.15x, 3x] of the sum or survival demonstrates "
                        f"nothing (invalid instrument)")

            ok, detail = False, ""
            ports = []
            if fail is not None:
                detail = fail
            else:
                env = dict(base_env)
                env["ELASTIC_DIR"] = edir
                specs = []
                for i, mdl in enumerate(cmodels):
                    port = free_port(a.port_base, used)
                    ports.append(port)
                    specs.append((mdl, port, True))
                unit = f"cascadia-conformance-c4-{os.getpid()}"
                plog = f"/tmp/conformance-c4-fleet-{budget}.log"
                drop_cache(cmodels)   # same cache instrument note as C3
                pf = launch_fleet(results["bin"], specs, env, plog, budget, unit,
                                  min_mb=a.elastic_min_mb, so=so_path)
                texts, live_ok = [], True
                try:
                    for i, port in enumerate(ports):
                        st = wait_port(port, pf[0], 600)
                        if st != "up":
                            live_ok = False
                            detail = (f"server {i} ({cmodels[i]}) {st} under the "
                                      f"fleet budget (log: {plog}.{port})")
                            break
                    if live_ok:
                        for i, port in enumerate(ports):
                            dt, ntok, text = chat(port, prompt, a.max_tokens)
                            texts.append(text)
                            log(f"  tenant {i}: {ntok} tok in {dt:.1f}s")
                        # The fleet budget must be the instrument, verified.
                        pid = find_pid(ports[0])
                        mmax = cgroup_file(pid, "memory.max") if pid else None
                        ev = cgroup_file(pid, "memory.events") if pid else None
                        if mmax is None or not mmax.isdigit() or \
                                int(mmax) != budget * 1024 * 1024:
                            detail = (f"memory.max={mmax} != requested "
                                      f"{budget * 1024 * 1024} (invalid "
                                      f"instrument)")
                        elif texts != solo_text:
                            detail = ("served, but a tenant's text differs "
                                      "from its solo run")
                        else:
                            m = re.search(r"oom_kill\s+(\d+)", ev or "")
                            if m and int(m.group(1)) > 0:
                                detail = f"oom_kill fired: {(ev or '').strip()[:160]}"
                            else:
                                ok = True
                                detail = (f"{len(cmodels)} tenants served their "
                                          f"solo text under one {budget} MB fleet "
                                          f"budget (solo floor sum {floor_sum} MB "
                                          f"= {floor_sum / budget:.0%} of it); "
                                          f"memory.max={mmax}")
                except Exception as e:
                    detail = f"generation failed: {type(e).__name__}: {e}"
                finally:
                    stop_fleet(pf, ports)
            record("C4 co-tenancy", ok, detail)

    ok_all = all(c["pass"] for c in checks.values()) and len(checks) >= 3
    finish(0 if ok_all else 1)


if __name__ == "__main__":
    main()
