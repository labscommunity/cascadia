#!/usr/bin/env python3
"""Fleet beacon: the ranks find each other without a list of addresses.

Every box runs this next to its rank (cascadia-inkling-beacon.service). Once a
second it announces "rank N of fleet F is here" by UDP broadcast on the wired
port, and it listens for the others. What it hears goes into a managed block
of /etc/hosts:

    192.168.1.37 inkling-rank-4

Workers dial `inkling-rank-<next>:<port>`; the name is resolved again at every
connection attempt, so an address that DHCP rotates is picked up by itself.
When this box's own address or a neighbouring rank's address changes while the
worker is running, the worker is restarted (--on-change): its TCP connections
to the old address are dead, and TCP alone takes minutes to notice.

    beacon.py --rank 4 --total 11        the service
    beacon.py --show                      who is where (any machine on the LAN with Python 3)
    beacon.py --top                       live load of every box (CPU, memory, disk, network, power, iGPU)
    beacon.py --record FILE --wait 600    the same, written to FILE as JSON lines (for a benchmark)

Once a second every box also broadcasts a second, separate packet with its own
load figures and the worker's latest "stage profile" log line (see
CASCADIA_STAGE_PROFILE_SECS), so one machine sees where the whole pipeline's
time goes without a login on the other boxes. Discovery never depends on it.

Same LAN segment only (broadcast does not cross routers), and nothing here is
authenticated: anyone on the LAN can claim a rank. Fine for a demo network.
"""
import argparse
import json
import os
import re
import select
import socket
import subprocess
import sys
import time

MAGIC = "cascadia-inkling-beacon-1"
TELE_MAGIC = "cascadia-inkling-tele-1"  # load figures: a packet of its own, so it can never break discovery
VIRTUAL = ("lo", "docker", "veth", "br-", "virbr", "tailscale", "wl", "ww", "tun", "wg")
BEGIN = "# >>> cascadia-inkling fleet (written by beacon.py; do not edit between the markers) >>>"
END = "# <<< cascadia-inkling fleet <<<"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def local_addrs():
    """[(interface, address, broadcast)], wired ports with link first."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "scope", "global"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    wired, other = [], []
    for line in out.splitlines():
        f = line.split()
        if len(f) < 4 or "brd" not in f:
            continue
        name, ip, brd = f[1], f[3].split("/")[0], f[f.index("brd") + 1]
        try:
            link = open("/sys/class/net/%s/carrier" % name).read().strip() == "1"
        except OSError:
            link = True
        (wired if link and not name.startswith(VIRTUAL) else other).append((name, ip, brd))
    # A box with no wired address yet (or a test on Wi-Fi) still announces itself.
    return wired or [a for a in other if not a[0].startswith(("lo", "docker", "veth", "br-", "virbr"))]


def worker_status(unit="cascadia-inkling.service"):
    """What this box's rank is doing, in a few words, for `--show` on any other machine."""
    st = {"state": "?", "restarts": 0, "phase": ""}
    try:
        out = subprocess.run(["systemctl", "show", unit, "-p", "ActiveState", "-p", "NRestarts", "-p", "MainPID"],
                             capture_output=True, text=True, timeout=5).stdout
        kv = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
        st["state"] = kv.get("ActiveState", "?")
        st["restarts"] = int(kv.get("NRestarts", "0") or 0)
        worker_status.pid = int(kv.get("MainPID", "0") or 0)
        log_ = subprocess.run(["journalctl", "-u", unit, "-n", "400", "-o", "cat", "--no-pager"],
                              capture_output=True, text=True, timeout=5).stdout
        lines = [re.sub(r"\x1b\[[0-9;]*m", "", l).strip() for l in log_.splitlines()]
        lines = [l for l in lines if l and "GPU_MOE_BATCHED" not in l]
        prof = [l for l in lines if " stage profile " in l]
        if prof:
            worker_status.profile = prof[-1]
        lines = [l for l in lines if " stage profile " not in l]
        last_start = max([i for i, l in enumerate(lines) if "worker starting" in l] or [0])
        run = lines[last_start:]
        text = " ".join(run)
        if any(k in text for k in ("entering relay loop", "API serving", "API + dashboard serving", "stream admitted", "task done")):
            phase = "serving"
        elif "upstream peer accepted" in text:
            phase = "loading the model"
        elif "downstream connected" in text:
            phase = "waiting for the previous rank to dial in"
        elif "waiting for downstream peer" in text:
            phase = "waiting for the next rank"
        else:
            phase = ""
        errs = [l for l in run if re.search(r"ERROR|Error:|panicked|exiting for supervisor", l)]
        if errs:
            phase = (phase + "; " if phase else "") + "last error: " + re.sub(r"^\S+Z\s+\w+\s+\S+:\s*", "", errs[-1])[:110]
        elif not phase and run:
            phase = re.sub(r"^\S+Z\s+\w+\s+\S+:\s*", "", run[-1])[:110]
        st["phase"] = phase
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return st


worker_status.pid = 0  # the worker's process, for its CPU time and memory
worker_status.profile = ""  # its latest "stage profile" log line


def _read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def _num(path, default=0):
    try:
        return int(_read(path).split()[0])
    except (ValueError, IndexError):
        return default


class Telemetry:
    """This box's load, from /proc and /sys only (no tools to install, a few hundred microseconds a sample).

    Counters become per-second rates between two samples; every source is optional, a box that lacks one
    simply does not report that figure.
    """

    def __init__(self):
        self.prev = None
        self.ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        self.static_sent = 0.0

    def counters(self, iface, pid):
        import glob
        c = {"t": time.time()}
        f = _read("/proc/stat").split("\n", 1)[0].split()
        if len(f) >= 8:
            v = [int(x) for x in f[1:9]]
            c["cpu_total"], c["cpu_idle"], c["cpu_iow"] = sum(v), v[3], v[4]
        for line in _read("/proc/vmstat").splitlines():
            k, _, v = line.partition(" ")
            if k in ("pgpgin", "pgpgout", "pswpin", "pswpout", "pgmajfault"):
                c[k] = int(v)
        rd = wr = 0
        for line in _read("/proc/diskstats").splitlines():
            f = line.split()
            if len(f) > 9 and re.match(r"^(nvme\d+n\d+|sd[a-z]+|vd[a-z]+)$", f[2]):
                rd += int(f[5])
                wr += int(f[9])
        c["rd_sect"], c["wr_sect"] = rd, wr
        if iface:
            c["rx"] = _num("/sys/class/net/%s/statistics/rx_bytes" % iface)
            c["tx"] = _num("/sys/class/net/%s/statistics/tx_bytes" % iface)
        m = re.search(r"^Tcp: (?!Rto)(.*)$", _read("/proc/net/snmp"), re.M)
        if m:
            f = m.group(1).split()
            if len(f) > 11:
                c["retrans"] = int(f[11])
        if pid:
            f = _read("/proc/%d/stat" % pid).rsplit(")", 1)[-1].split()
            if len(f) > 13:
                c["p_majflt"], c["p_cpu"] = int(f[9]), int(f[11]) + int(f[12])
        c["rapl"] = _num("/sys/class/powercap/intel-rapl:0/energy_uj", -1)
        c["rapl_max"] = _num("/sys/class/powercap/intel-rapl:0/max_energy_range_uj", 0)
        idle = [_num(g, -1) for g in sorted(glob.glob("/sys/class/drm/card*/device/tile*/gt0/gtidle/idle_residency_ms"))]
        c["gpu_idle_ms"] = idle[0] if idle else -1
        c["throttle"] = _num("/sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count", 0)
        return c

    def gauges(self, pid):
        import glob
        g = {}
        mem = {}
        for line in _read("/proc/meminfo").splitlines():
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable", "Cached", "AnonPages", "SwapTotal", "SwapFree", "Mlocked"):
                mem[k] = int(v.split()[0]) // 1024
        if mem:
            g.update(mem_total=mem.get("MemTotal", 0), mem_avail=mem.get("MemAvailable", 0), cached=mem.get("Cached", 0),
                     anon=mem.get("AnonPages", 0), swap=mem.get("SwapTotal", 0) - mem.get("SwapFree", 0))
        if pid:
            for line in _read("/proc/%d/status" % pid).splitlines():
                k, _, v = line.partition(":")
                if k in ("RssAnon", "RssFile", "VmSwap"):
                    g["p_" + k.lower()] = int(v.split()[0]) // 1024
        freqs = [_num(p) for p in glob.glob("/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq")]
        if freqs:
            g["mhz"] = sum(freqs) // len(freqs) // 1000
            g["mhz_max"] = max(freqs) // 1000
        temps = [_num(p) for p in glob.glob("/sys/class/thermal/thermal_zone*/temp")]
        if temps:
            g["temp"] = max(temps) // 1000
        act = [_num(p) for p in sorted(glob.glob("/sys/class/drm/card*/device/tile*/gt0/freq0/act_freq"))]
        if act:
            g["gpu_mhz"] = act[0]
        return g

    def static(self, iface):
        s = {"ncpu": os.cpu_count() or 0, "kernel": os.uname().release}
        m = re.search(r"^model name\s*:\s*(.*)$", _read("/proc/cpuinfo"), re.M)
        if m:
            s["cpu"] = m.group(1).strip()[:48]
        s["gov"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").strip()
        s["epp"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference").strip()
        s["profile"] = _read("/sys/firmware/acpi/platform_profile").strip()
        for i, name in ((0, "pkg"), (1, "psys")):
            z = "/sys/class/powercap/intel-rapl:%d/" % i
            if _read(z + "name").strip():
                s[name + "_pl1_w"] = _num(z + "constraint_0_power_limit_uw") // 1000000
                s[name + "_pl2_w"] = _num(z + "constraint_1_power_limit_uw") // 1000000
        if iface:
            s["nic"], s["nic_mbps"] = iface, _num("/sys/class/net/%s/speed" % iface, -1)
        s["swappiness"] = _num("/proc/sys/vm/swappiness", -1)
        return {k: v for k, v in s.items() if v not in ("", None)}

    def sample(self, iface, pid):
        """One telemetry dict, or None on the first call (rates need two samples)."""
        now = self.counters(iface, pid)
        prev, self.prev = self.prev, now
        if not prev:
            return None
        dt = max(now["t"] - prev["t"], 1e-3)
        d = lambda k: max(now.get(k, 0) - prev.get(k, 0), 0)
        out = {}
        if d("cpu_total"):
            out["cpu"] = round(1 - d("cpu_idle") / d("cpu_total") - d("cpu_iow") / d("cpu_total"), 3)
            out["iowait"] = round(d("cpu_iow") / d("cpu_total"), 3)
        out["rd_mb_s"] = round(d("rd_sect") * 512 / 1e6 / dt, 1)
        out["wr_mb_s"] = round(d("wr_sect") * 512 / 1e6 / dt, 1)
        out["swapin_s"], out["swapout_s"] = round(d("pswpin") / dt), round(d("pswpout") / dt)
        out["majflt_s"] = round(d("pgmajfault") / dt)
        if "rx" in now:
            out["rx_mb_s"], out["tx_mb_s"] = round(d("rx") / 1e6 / dt, 2), round(d("tx") / 1e6 / dt, 2)
        out["retrans_s"] = round(d("retrans") / dt, 1)
        if "p_cpu" in now and "p_cpu" in prev:
            out["p_cores"] = round(d("p_cpu") / self.ticks / dt, 2)
            out["p_majflt_s"] = round(d("p_majflt") / dt)
        if now["rapl"] >= 0 and prev.get("rapl", -1) >= 0:
            e = now["rapl"] - prev["rapl"]
            if e < 0:
                e += now["rapl_max"]
            out["pkg_w"] = round(e / 1e6 / dt, 1)
        if now["gpu_idle_ms"] >= 0 and prev.get("gpu_idle_ms", -1) >= 0:
            out["gpu"] = round(min(max(1 - d("gpu_idle_ms") / 1e3 / dt, 0), 1), 3)
        out["throttle"] = d("throttle")
        out.update(self.gauges(pid))
        if now["t"] - self.static_sent >= 15:
            self.static_sent = now["t"]
            out["static"] = self.static(iface)
        return out


def profile_fields(line):
    """The key=value numbers of a worker's "stage profile" log line, plus its own timestamp."""
    if not line:
        return None
    f = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)\b", line.split(" stage profile ", 1)[-1])}
    if not f:
        return None
    f["at"] = line.split()[0]
    return f


def announce_telemetry(port, payload, tele):
    """Broadcast this box's load figures. Best effort: whatever fails here, discovery goes on."""
    try:
        addrs = local_addrs()[:1]
        if not addrs:
            return
        name, ip, brd = addrs[0]
        t = tele.sample(name, worker_status.pid)
        if t is None:
            return
        msg = {"magic": TELE_MAGIC, "fleet": payload["fleet"], "rank": payload["rank"], "host": payload["host"],
               "t": round(time.time(), 2), "sys": t}
        p = profile_fields(worker_status.profile)
        if p:
            msg["prof"] = p
        data = json.dumps(msg, separators=(",", ":")).encode()
        if len(data) > 1400 and "static" in t:  # stay inside one Ethernet frame
            del t["static"]
            data = json.dumps(msg, separators=(",", ":")).encode()
        if len(data) > 1400:
            msg.pop("prof", None)
            data = json.dumps(msg, separators=(",", ":")).encode()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.bind((ip, 0))
        s.sendto(data, (brd, port))
        s.close()
    except Exception:  # noqa: BLE001 - never let a sensor or a socket take the beacon down
        pass


def listener(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if hasattr(socket, "SO_REUSEPORT"):  # --show next to the service, both hear every broadcast
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.bind(("", port))
    s.setblocking(False)
    return s


def announce(port, payload):
    sent = []
    # One port only: a box with two live ports would otherwise look like two boxes claiming one rank.
    for name, ip, brd in local_addrs()[:1]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.bind((ip, 0))  # so the receivers see this port's address as the source
            try:
                mac = open("/sys/class/net/%s/address" % name).read().strip()
            except OSError:
                mac = payload["host"]
            try:
                files = int(json.load(open("/run/cascadia-inkling/update.json")).get("version", 0))
            except (OSError, ValueError):
                files = 0
            s.sendto(json.dumps(dict(payload, ip=ip, id=mac, files=files))[:1300].encode(), (brd, port))
            s.close()
            sent.append(ip)
        except OSError:
            pass
    return sent


def parse(data, fleet):
    try:
        m = json.loads(data.decode())
        if m.get("magic") == MAGIC and m.get("fleet") == fleet:
            host = str(m.get("host", "?"))
            parse.files[int(m["rank"])] = int(m.get("files", 0) or 0)
            if isinstance(m.get("w"), dict):
                parse.worker[int(m["rank"])] = m["w"]
            return int(m["rank"]), int(m.get("total", 0)), host, str(m.get("id", host))
    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
        pass
    return None


parse.files = {}  # rank -> version of the fleet files its updater has applied (0 = not enrolled)
parse.worker = {}  # rank -> {"state", "restarts", "phase"} as that box reports it


def write_hosts(path, fleet, table):
    lines = [BEGIN] + ["%s %s-rank-%d" % (table[r]["ip"], fleet, r) for r in sorted(table)] + [END]
    try:
        old = open(path).read().splitlines()
    except OSError:
        old = []
    out, skip = [], False
    for l in old:
        if l.strip() == BEGIN:
            skip = True
        elif l.strip() == END:
            skip = False
        elif not skip:
            out.append(l)
    while out and not out[-1].strip():
        out.pop()
    text = "\n".join(out + [""] + lines) + "\n"
    tmp = path + ".inkling-tmp"
    try:
        with open(tmp, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except OSError:  # a bind-mounted /etc/hosts (containers) cannot be replaced, only rewritten
        try:
            os.unlink(tmp)
        except OSError:
            pass
        with open(path, "w") as f:
            f.write(text)


def show(table, dups, total, fleet, me=None):
    total = total or (max(table) + 1 if table else 0)
    now = time.time()
    for r in range(total):
        e = table.get(r)
        if not e:
            print("  rank %2d  --  not heard" % r)
            continue
        age = now - e["seen"]
        note = "  <-- this box" if me == r else ""
        if age > 10:
            note += "  (silent for %d s)" % age
        if r in dups:
            note += "  <-- CLAIMED BY SEVERAL BOXES: %s" % ", ".join(sorted(dups[r]))
        v = parse.files.get(r, 0)
        files = "files %s" % time.strftime("%m-%d %H:%M:%S", time.localtime(v)) if v else "files: not enrolled"
        w = parse.worker.get(r)
        if w:
            clean = lambda t: re.sub(r"[^\x20-\x7e]", "?", str(t))[:140]
            note = "  | worker %s, %s restarts: %s%s" % (clean(w.get("state")), clean(w.get("restarts")), clean(w.get("phase")), note)
        print("  rank %2d  %-15s  %-16s %s%s" % (r, e["ip"], e["host"], files, note))
    if 0 in table:
        print("  API: http://%s:8000   (on a fleet box also http://%s-rank-0:8000)" % (table[0]["ip"], fleet))


def run_show(a):
    sock = listener(a.port)
    table, dups, total = {}, {}, 0
    end = time.time() + a.wait
    while time.time() < end:
        r, _, _ = select.select([sock], [], [], 0.2)
        if not r:
            continue
        data, (src, _) = sock.recvfrom(2048)
        m = parse(data, a.fleet)
        if not m:
            continue
        rank, tot, host, ident = m
        total = max(total, tot)
        if rank in table and table[rank]["id"] != ident:
            dups.setdefault(rank, set()).update([host + "@" + src, table[rank]["host"] + "@" + table[rank]["ip"]])
        table[rank] = {"ip": src, "host": host, "id": ident, "seen": time.time()}
    if not table:
        print("no beacons heard on UDP port %d in %d s (same LAN segment? firewall?)" % (a.port, a.wait))
        return 1
    show(table, dups, total, a.fleet)
    return 2 if dups else 0


def run_watch(a):
    """--top (a table that refreshes) and --record FILE (every packet as a JSON line, profiles once each)."""
    sock = listener(a.port)
    latest, seen_prof = {}, {}
    out = open(a.record, "a") if a.record else None
    end = time.time() + a.wait if (a.record or a.wait != 3.0) else float("inf")
    last_draw = 0.0
    cols = (("cpu", "cpu", 5), ("p_cores", "cores", 5), ("pkg_w", "W", 5), ("mhz", "MHz", 5), ("temp", "C", 3),
            ("gpu", "gpu", 5), ("mem_avail", "avail", 6), ("p_rssanon", "anon", 6), ("cached", "cache", 6),
            ("swap", "swap", 5), ("rd_mb_s", "rdMB/s", 7), ("majflt_s", "majf/s", 6), ("swapin_s", "swi/s", 6),
            ("rx_mb_s", "rxMB/s", 6), ("tx_mb_s", "txMB/s", 6))
    try:
        while time.time() < end:
            r, _, _ = select.select([sock], [], [], 0.5)
            if r:
                data, (src, _) = sock.recvfrom(4096)
                try:
                    m = json.loads(data.decode())
                except (ValueError, UnicodeDecodeError):
                    continue
                if m.get("magic") != TELE_MAGIC or m.get("fleet") != a.fleet:
                    continue
                rank = int(m.get("rank", -1))
                prof = m.get("prof")
                if prof and seen_prof.get(rank) == prof.get("at"):
                    prof = None  # the same log line again: the worker has not written a newer one
                elif prof:
                    seen_prof[rank] = prof.get("at")
                prev = latest.get(rank, {})
                latest[rank] = {"sys": m.get("sys", {}), "prof": prof or prev.get("prof"), "host": m.get("host"), "ip": src,
                                "static": m.get("sys", {}).get("static") or prev.get("static")}
                if out:
                    rec = {"t": m.get("t"), "rank": rank, "host": m.get("host"), "sys": m.get("sys", {})}
                    if prof:
                        rec["prof"] = prof
                    out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    out.flush()
            if a.top and time.time() - last_draw >= 1.0:
                last_draw = time.time()
                lines = ["\x1b[H\x1b[2J%s fleet %s   (Ctrl-C to stop)" % (time.strftime("%H:%M:%S"), a.fleet),
                         "rank " + " ".join(h.rjust(w) for _, h, w in cols) + "  | last stage profile: util  ms/row  wait%  cache miss%"]
                for rank in sorted(latest):
                    e = latest[rank]
                    row = "%4d " % rank + " ".join(str(e["sys"].get(k, "-")).rjust(w) for k, _, w in cols)
                    p = e.get("prof")
                    if p and p.get("window_ms"):
                        busy = sum(p.get(k, 0) for k in ("recv_ms", "compute_ms", "prefill_ms", "head_ms", "send_ms", "relay_ms", "emit_ms"))
                        look = p.get("cache_hits", 0) + p.get("cache_misses", 0)
                        row += "  | %4.0f%% %7.1f %5.0f%% %6.1f%%" % (
                            100.0 * busy / p["window_ms"], p.get("compute_ms", 0) / max(p.get("rows", 0), 1),
                            100.0 * p.get("wait_ms", 0) / p["window_ms"], 100.0 * p.get("cache_misses", 0) / max(look, 1))
                    lines.append(row)
                print("\n".join(lines), flush=True)
    except KeyboardInterrupt:
        pass
    if out:
        out.close()
    if not latest:
        print("no telemetry heard on UDP port %d (boxes still on an older beacon.py?)" % a.port)
        return 1
    return 0


def run_service(a):
    sock = listener(a.port)
    host = socket.gethostname()
    payload = {"magic": MAGIC, "fleet": a.fleet, "rank": a.rank, "total": a.total, "host": host}
    table, dups = {}, {}
    try:  # the table survives a restart of this service, so a change that happened meanwhile is still a change
        st = json.load(open(a.state))
        if st.get("fleet") == a.fleet:
            table = {int(k): v for k, v in st.get("table", {}).items()}
    except (OSError, ValueError):
        pass
    pending = {}  # rank -> (new address, first heard) until it has been stable for a moment
    watched = {a.rank - 1, a.rank, a.rank + 1}
    last_send = last_state = last_warn = last_worker = last_tele = 0.0
    tele = Telemetry()
    dirty = bool(table)
    log("beacon for rank %d of %d, fleet %s, UDP %d" % (a.rank, a.total, a.fleet, a.port))
    while True:
        now = time.time()
        if now - last_worker >= 5.0:
            last_worker = now
            payload["w"] = worker_status()
        if now - last_send >= 1.0:
            last_send = now
            if not announce(a.port, payload) and now - last_warn > 30:
                last_warn = now
                log("no network address to announce from yet")
        if a.telemetry and now - last_tele >= 1.0:
            last_tele = now
            announce_telemetry(a.port, payload, tele)
        r, _, _ = select.select([sock], [], [], 0.5)
        if r:
            try:
                data, (src, _) = sock.recvfrom(2048)
            except OSError:
                continue
            m = parse(data, a.fleet)
            if not m:
                continue
            rank, _, peer, ident = m
            now = time.time()
            e = table.get(rank)
            if e and e.get("id", ident) != ident and now - e["seen"] < 10:
                # Two boxes claim one rank: never follow either, or the ranks would flap between them.
                if rank not in dups:
                    log("WARNING: rank %d is claimed by %s (%s) and %s (%s): re-install one of them with its own rank"
                        % (rank, e["host"], e["ip"], peer, src))
                dups[rank] = now
                continue
            if rank in dups and now - dups[rank] > 30:
                del dups[rank]
            if e is None:
                table[rank] = {"ip": src, "host": peer, "id": ident, "seen": now}
                dirty = True
                log("rank %d is at %s (%s)" % (rank, src, peer))
            elif e["ip"] == src:
                e["seen"], e["host"], e["id"] = now, peer, ident
                pending.pop(rank, None)
            else:
                first = pending.setdefault(rank, (src, now))
                if first[0] != src:
                    pending[rank] = (src, now)
                elif now - first[1] >= a.settle:
                    log("rank %d moved from %s to %s" % (rank, e["ip"], src))
                    table[rank] = {"ip": src, "host": peer, "id": ident, "seen": now}
                    pending.pop(rank, None)
                    dirty = True
                    if rank in watched and a.on_change:
                        log("restarting the local worker: %s" % a.on_change)
                        subprocess.Popen(a.on_change, shell=True)
        if dirty:
            dirty = False
            if a.hosts:
                try:
                    write_hosts(a.hosts, a.fleet, table)
                except OSError as err:
                    log("cannot write %s: %s" % (a.hosts, err))
        if time.time() - last_state >= 2.0:
            last_state = time.time()
            try:
                os.makedirs(os.path.dirname(a.state), exist_ok=True)
                tmp = a.state + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"fleet": a.fleet, "rank": a.rank, "total": a.total, "time": last_state,
                               "table": table, "duplicates": sorted(dups)}, f)
                os.replace(tmp, a.state)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", type=int)
    ap.add_argument("--total", type=int, default=0)
    ap.add_argument("--fleet", default="inkling")
    ap.add_argument("--port", type=int, default=9099)
    ap.add_argument("--hosts", default="/etc/hosts", help="hosts file to keep current ('' = none)")
    ap.add_argument("--state", default="/run/cascadia-inkling/fleet.json")
    ap.add_argument("--settle", type=float, default=3.0, help="seconds a new address must persist before it counts")
    ap.add_argument("--on-change", default="", help="command to run when this rank's or a neighbour's address changed")
    ap.add_argument("--show", action="store_true", help="print who is where and exit")
    ap.add_argument("--wait", type=float, default=3.0, help="--show / --record: seconds to listen")
    ap.add_argument("--top", action="store_true", help="live load of every box until Ctrl-C")
    ap.add_argument("--record", default="", help="append every box's load figures to this file as JSON lines for --wait seconds")
    ap.add_argument("--no-telemetry", dest="telemetry", action="store_false", help="service: announce the rank only")
    a = ap.parse_args()
    if a.top or a.record:
        return run_watch(a)
    if a.show:
        return run_show(a)
    if a.rank is None or a.total <= 0:
        ap.error("--rank and --total are required (or --show)")
    return run_service(a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
