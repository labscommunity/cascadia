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
        out = subprocess.run(["systemctl", "show", unit, "-p", "ActiveState", "-p", "NRestarts"],
                             capture_output=True, text=True, timeout=5).stdout
        kv = dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
        st["state"] = kv.get("ActiveState", "?")
        st["restarts"] = int(kv.get("NRestarts", "0") or 0)
        log_ = subprocess.run(["journalctl", "-u", unit, "-n", "400", "-o", "cat", "--no-pager"],
                              capture_output=True, text=True, timeout=5).stdout
        lines = [re.sub(r"\x1b\[[0-9;]*m", "", l).strip() for l in log_.splitlines()]
        lines = [l for l in lines if l and "GPU_MOE_BATCHED" not in l]
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
    last_send = last_state = last_warn = last_worker = 0.0
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
    ap.add_argument("--wait", type=float, default=3.0, help="--show: seconds to listen")
    a = ap.parse_args()
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
