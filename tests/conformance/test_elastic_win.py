#!/usr/bin/env python3
"""Windows conformance checks for the `--elastic` posture.

The Linux suite (`test_elastic.py`) reads `/proc/<pid>/status` and drives cgroup
v2. Windows has neither, so this port uses what Windows does have:

  * **private bytes** via `GetProcessMemoryInfo` (psapi) — the commit charge.
    This is the witness, because it is what counts against the commit limit and
    what an OOM would kill.
  * **working set** via the same call — reported for context only. It does NOT
    drop without pressure (measured on HunterLaptopSergio: 3101 -> 3116 MB),
    so it is not a pass condition and must not be mistaken for one.
  * the **gate line** `elastic posture active (in-process; ...)`. A run without
    it has the Detours hook compiled out or not installed, and its numbers are
    invalid — not a pass. (Detours must be present at build time via
    `DETOURS_DIR`; note that cargo caches the fingerprint, so after building
    Detours run `cargo clean -p cascadia-elastic` or the build silently keeps
    the hook out.)
  * **Job Objects** (`JOB_OBJECT_LIMIT_PROCESS_MEMORY`) for the pressure leg —
    the Windows analog of a cgroup `MemoryMax`.

Checks
------
  W1 gate line       the posture must report active
  W2 output identity stock vs elastic, temperature 0 -> byte-identical
  W3 elasticity      private bytes collapse vs stock
  W4 pressure (opt)  serve under a Job Object commit cap (--pressure-mb N)

Usage
-----
  python test_elastic_win.py --exe C:\\path\\cascadia.exe \\
      --model C:\\path\\model-ov --setup C:\\ov\\setupvars.bat \\
      [--pressure-mb 512] [--json-out results.json]

Exit code 0 only if every selected check passes.
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

MB = 1 << 20

# ---- psapi: private bytes + working set ----------------------------------


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),          # <- private commit
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_memory(pid):
    """Return (private_bytes, working_set_bytes) for `pid`, or None."""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        c = PROCESS_MEMORY_COUNTERS()
        c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return None
        return int(c.PagefileUsage), int(c.WorkingSetSize)
    finally:
        k32.CloseHandle(h)


# ---- Job Objects: the pressure leg ---------------------------------------


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wt.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wt.DWORD),
        ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
        ("PriorityClass", wt.DWORD),
        ("SchedulingClass", wt.DWORD),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


def make_capped_job(cap_bytes):
    """Create a Job Object that caps a process's commit. Returns the handle."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise OSError("CreateJobObjectW failed")
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = cap_bytes
    ok = k32.SetInformationJobObject(
        job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
        ctypes.sizeof(info))
    if not ok:
        raise OSError("SetInformationJobObject failed")
    return job


def assign_to_job(job, pid):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_SET_QUOTA = 0x0100
    PROCESS_TERMINATE = 0x0001
    h = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
    if not h:
        raise OSError("OpenProcess for job assignment failed")
    try:
        if not k32.AssignProcessToJobObject(job, h):
            raise OSError("AssignProcessToJobObject failed")
    finally:
        k32.CloseHandle(h)


# ---- server driving -------------------------------------------------------


def wait_port(port, proc, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return "died"
        s = socket.socket()
        s.settimeout(1)
        try:
            s.connect(("127.0.0.1", port))
            s.close()
            return "up"
        except OSError:
            pass
        time.sleep(2)
    return "timeout"


def chat(port, prompt, max_tokens, timeout=600):
    body = json.dumps({
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def launch(exe, model, port, elastic, log_path, setup=None):
    args = [exe, "run", model, "--device", "CPU",
            "--api", "127.0.0.1:%d" % port]
    if elastic:
        args.append("--elastic")
    env = dict(os.environ)
    if elastic:
        env["ELASTIC_DIR"] = env.get("ELASTIC_DIR", os.path.join(
            os.path.dirname(log_path), "elastic-backing"))
    if setup:
        # Route through a wrapper .bat: building the equivalent string for
        # `cmd /c` mangles the quoting and the setup call is not found.
        bat = log_path + ".bat"
        with open(bat, "w", newline="\r\n") as f:
            f.write("@echo off\n")
            f.write('call "%s" >nul\n' % setup)
            f.write('"%s"\n' % '" "'.join(args))
        cmd = ["cmd", "/c", bat]
    else:
        cmd = args
    lf = open(log_path, "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return p, lf


def gate_line(log_path):
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "elastic posture active" in line:
                    return line.strip()
    except OSError:
        pass
    return None


def cascadia_pids():
    """PIDs of live cascadia.exe processes, via tasklist."""
    out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq cascadia.exe", "/FO", "CSV", "/NH"],
        capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        parts = [x.strip('"') for x in line.split('","')]
        if len(parts) >= 2 and parts[0].lower() == "cascadia.exe":
            try:
                pids.append(int(parts[1]))
            except ValueError:
                pass
    return pids


def server_pid(proc):
    """The process holding the memory.

    launch() may go through a wrapper .bat, so proc.pid is cmd.exe, whose
    footprint is ~7 MB — measuring it makes W3 compare two cmd processes and
    pass vacuously. Find the actual cascadia.exe instead.
    """
    pids = cascadia_pids()
    if not pids:
        return proc.pid
    # biggest footprint wins, in case a leftover survived a previous leg
    best, best_mem = pids[0], -1
    for pid in pids:
        m = process_memory(pid)
        if m and m[0] > best_mem:
            best, best_mem = pid, m[0]
    return best


def kill(proc):
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/F"],
                       capture_output=True)
    except Exception:
        pass
    for pid in cascadia_pids():
        subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                       capture_output=True)


# ---- checks ---------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--setup", help="setupvars.bat to call before launching")
    ap.add_argument("--prompt", default="Write a haiku about memory.")
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--settle-secs", type=int, default=15)
    ap.add_argument("--pressure-mb", type=int, default=0)
    ap.add_argument("--tmp", default=os.environ.get("TEMP", "."))
    ap.add_argument("--json-out")
    a = ap.parse_args()

    results = []

    def record(ok, name, detail):
        results.append({"name": name, "ok": bool(ok), "detail": detail})
        print("[conformance]   %s  %s - %s" %
              ("PASS" if ok else "FAIL", name, detail))

    print("[conformance] === Windows elastic checks ===")
    print("[conformance]   exe   %s" % a.exe)
    print("[conformance]   model %s" % a.model)

    # ---- stock leg ----
    log_stock = os.path.join(a.tmp, "conf-win-stock.log")
    p, lf = launch(a.exe, a.model, 8021, False, log_stock, a.setup)
    st = wait_port(8021, p, a.load_timeout)
    if st != "up":
        lf.close()
        record(False, "stock leg", "did not serve (%s) — see %s" % (st, log_stock))
        return finish(results, a)
    r_stock = chat(8021, a.prompt, a.max_tokens)
    time.sleep(a.settle_secs)
    mem_stock = process_memory(server_pid(p))
    txt_stock = r_stock["choices"][0]["message"]["content"]
    kill(p)
    lf.close()

    # ---- elastic leg ----
    log_el = os.path.join(a.tmp, "conf-win-elastic.log")
    p, lf = launch(a.exe, a.model, 8022, True, log_el, a.setup)
    st = wait_port(8022, p, a.load_timeout)
    if st != "up":
        lf.close()
        record(False, "elastic leg", "did not serve (%s) — see %s" % (st, log_el))
        return finish(results, a)

    # W1 — gate line. Without it the hook is out and nothing below means anything.
    gl = gate_line(log_el)
    record(gl is not None, "W1 gate line (posture active)", gl or "NOT FOUND")

    r_el = chat(8022, a.prompt, a.max_tokens)
    time.sleep(a.settle_secs)
    mem_el = process_memory(server_pid(p))
    txt_el = r_el["choices"][0]["message"]["content"]

    # W2 — output identity
    record(txt_stock == txt_el, "W2 output identity",
           "byte-identical (%d chars)" % len(txt_stock) if txt_stock == txt_el
           else "DIFFERS")

    # W3 — elasticity witness on private commit
    if mem_stock and mem_el:
        priv_s, ws_s = mem_stock
        priv_e, ws_e = mem_el
        ratio = priv_e / priv_s if priv_s else 9.9
        record(ratio <= 0.5, "W3 elasticity witness",
               "private commit %.0f -> %.0f MB (%.0f%% of stock); "
               "working set %.0f -> %.0f MB" %
               (priv_s / MB, priv_e / MB, ratio * 100, ws_s / MB, ws_e / MB))
    else:
        record(False, "W3 elasticity witness", "could not read process memory")

    # W4 — pressure leg (opt-in)
    if a.pressure_mb and mem_stock and a.pressure_mb * MB >= mem_stock[0]:
        # A cap at or above stock's own private commit proves nothing: stock
        # would have served under it too. Fail closed rather than record a
        # vacuous PASS, and say what cap would be meaningful.
        record(False, "W4 pressure survival",
               "inconclusive: cap %d MB is not below stock's private commit "
               "(%.0f MB); pick --pressure-mb below that" %
               (a.pressure_mb, mem_stock[0] / MB))
    elif a.pressure_mb:
        kill(p)
        lf.close()
        cap = a.pressure_mb * MB
        log_p = os.path.join(a.tmp, "conf-win-pressure.log")
        p2, lf2 = launch(a.exe, a.model, 8023, True, log_p, a.setup)
        job = make_capped_job(cap)
        try:
            assign_to_job(job, p2.pid)
        except OSError as e:
            lf2.close()
            kill(p2)
            record(False, "W4 pressure survival", "job assignment failed: %s" % e)
            return finish(results, a)
        st = wait_port(8023, p2, a.load_timeout)
        if st != "up":
            lf2.close()
            record(False, "W4 pressure survival",
                   "did not serve under a %d MB commit cap (%s)" %
                   (a.pressure_mb, st))
        else:
            try:
                r = chat(8023, a.prompt, a.max_tokens)
                ok = r["choices"][0]["message"]["content"] == txt_stock
                record(ok, "W4 pressure survival",
                       "served identical text under a %d MB commit cap" %
                       a.pressure_mb if ok else "served DIFFERENT text")
            except Exception as e:
                record(False, "W4 pressure survival", "request failed: %s" % e)
        kill(p2)
        lf2.close()
    else:
        kill(p)
        lf.close()

    return finish(results, a)


def finish(results, a):
    print("[conformance] ")
    print("[conformance] === summary ===")
    for r in results:
        print("[conformance]   %s  %s" % ("PASS" if r["ok"] else "FAIL", r["name"]))
    ok = bool(results) and all(r["ok"] for r in results)
    print("[conformance] RESULT: %s" % ("PASS" if ok else "FAIL"))
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"results": results, "ok": ok}, f, indent=2)
        print("[conformance] json: %s" % a.json_out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
