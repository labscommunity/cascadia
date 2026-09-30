#!/usr/bin/env python3
"""All-to-all request/reply traffic between the fleet's boxes, the pattern expert parallelism would put on the LAN.

Runs on every box at once (started from fleet-overrides.env, standard library only). Each box is a server (echoes
every message back) and a client of every other box (sends MSG-byte requests, waits for the MSG-byte reply, paced
so that the box transmits RATE MB/s in total: half as requests, half as the replies it serves). The rate ramps in
steps; rank 0's server is the clock every box schedules the steps on. At the end of each step one line goes to
stdout in the "stage profile" form the beacon relays to rank 0 (integers only): achieved MB/s from the interface
counters, round-trip percentiles of the messages in microseconds, stalls, TCP retransmits, kernel complaints about
the NIC. A box that sees more than 3 % retransmitted segments stops climbing; one whose kernel complains about
USB / the NIC stops sending.

Test hooks (environment): NETTEST_SELF, NETTEST_PEERS=host:port,..., NETTEST_PORT, NETTEST_SCALE (divide all times).
"""
import os, re, socket, struct, subprocess, sys, threading, time

RANK = int(os.environ.get("NETTEST_SELF", os.environ.get("RANK", "0")))
TOTAL = int(os.environ.get("TOTAL", "11"))
FLEET = os.environ.get("FLEET", "inkling")
PORT = int(os.environ.get("NETTEST_PORT", "9200"))
SCALE = float(os.environ.get("NETTEST_SCALE", "1"))
TAG = os.environ.get("NETTEST_TAG", "N18")
MSG = 100_000
STEPS = [(30, 1), (180, 5), (180, 15), (180, 30), (180, 50)]  # (seconds, MB/s transmitted per box)
HDR = struct.Struct("!cQI")
if os.environ.get("NETTEST_PEERS"):
    PEERS = [(i, h.rsplit(":", 1)[0], int(h.rsplit(":", 1)[1])) for i, h in enumerate(os.environ["NETTEST_PEERS"].split(","))]
else:
    PEERS = [(i, "%s-rank-%d" % (FLEET, i), PORT) for i in range(TOTAL)]
T0 = time.monotonic()
lock = threading.Lock()
rtts, state = [], {"rate": 0.0, "stop": False, "held": 0, "abort": 0, "offset": None, "connected": 0}


def recv_exact(s, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return buf


def serve(conn):
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            kind, seq, n = HDR.unpack(recv_exact(conn, HDR.size))
            body = recv_exact(conn, n) if n else b""
            if kind == b"T":  # what time is it on this box's test clock (ms)
                conn.sendall(HDR.pack(b"T", int((time.monotonic() - T0) * 1000), 0))
            else:
                conn.sendall(HDR.pack(b"E", seq, len(body)) + body)
    except Exception:  # noqa: BLE001
        pass
    finally:
        conn.close()


def server():
    ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("0.0.0.0", PORT)); ls.listen(64)
    while True:
        c, _ = ls.accept()
        threading.Thread(target=serve, args=(c,), daemon=True).start()


def dial(host, port, deadline):
    while time.monotonic() < deadline and not state["stop"]:
        try:
            s = socket.create_connection((host, port), timeout=3)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1); s.settimeout(10)
            return s
        except OSError:
            time.sleep(2)
    return None


def client(host, port):
    s = dial(host, port, T0 + 1500 / SCALE)
    if s is None:
        return
    with lock:
        state["connected"] += 1
    payload, seq = b"\xa5" * MSG, 0
    while not state["stop"]:
        rate = state["rate"]
        if rate <= 0 or state["abort"]:
            time.sleep(0.05); continue
        gap = MSG / (rate * 1e6 / 2 / max(1, len(PEERS) - 1))  # seconds between this peer's requests
        t = time.monotonic()
        try:
            s.sendall(HDR.pack(b"E", seq, MSG) + payload)
            recv_exact(s, HDR.size + MSG)
        except Exception:  # noqa: BLE001
            with lock:
                rtts.append(10.0)  # a lost connection counts as a 10 s stall
            s.close(); s = dial(host, port, T0 + 1500 / SCALE)
            if s is None:
                return
            continue
        dt = time.monotonic() - t; seq += 1
        with lock:
            rtts.append(dt)
        time.sleep(max(0.0, gap - dt))


def counters():
    iface = subprocess.run("ip -o -4 route show to default | awk '{print $5; exit}'", shell=True, capture_output=True, text=True).stdout.strip()
    rd = lambda p: int(open(p).read()) if os.path.exists(p) else 0
    tx, rx = rd("/sys/class/net/%s/statistics/tx_bytes" % iface), rd("/sys/class/net/%s/statistics/rx_bytes" % iface)
    snmp = open("/proc/net/snmp").read() if os.path.exists("/proc/net/snmp") else ""
    m = re.search(r"Tcp: (\D[^\n]*)\nTcp: ([^\n]*)", snmp)
    tcp = dict(zip(m.group(1).split(), map(int, m.group(2).split()))) if m else {}
    return tx, rx, tcp.get("OutSegs", 0), tcp.get("RetransSegs", 0)


def kernel_complaints(seconds):
    try:
        out = subprocess.run(["journalctl", "-k", "-q", "--no-pager", "--since", "-%ds" % seconds], capture_output=True, text=True, timeout=8).stdout
    except Exception:  # noqa: BLE001
        return 0
    return len(re.findall(r"xhci|cdc_ncm|NETDEV WATCHDOG|transmit queue|usb [\d.-]+: reset|link is not ready", out, re.I))


def main():
    threading.Thread(target=server, daemon=True).start()
    ref = [p for p in PEERS if p[0] == 0][0]
    if RANK == 0:
        state["offset"] = 0.0
    else:  # schedule on rank 0's clock
        s = dial(ref[1], ref[2], T0 + 1200 / SCALE)
        if s is None:
            print("%sX probe stage profile ph=9 noref=1" % TAG, flush=True); return
        s.sendall(HDR.pack(b"T", 0, 0)); _, ms, _ = HDR.unpack(recv_exact(s, HDR.size)); s.close()
        state["offset"] = ms / 1000.0 - (time.monotonic() - T0)
    for i, h, p in PEERS:
        if i != RANK:
            threading.Thread(target=client, args=(h, p), daemon=True).start()
    t_end, prev_rate = 0.0, 0.0
    for ph, (dur, rate) in enumerate(STEPS):
        t_start, t_end = t_end, t_end + dur / SCALE
        now = time.monotonic() - T0 + state["offset"]
        if now >= t_end:
            continue  # joined late: this step is over on rank 0's clock
        time.sleep(max(0.0, t_start - now))
        state["rate"] = prev_rate if state["held"] else float(rate)
        with lock:
            rtts.clear()
        c0, w0, kerr, t_ph = counters(), counters(), 0, time.monotonic()
        while time.monotonic() - T0 + state["offset"] < t_end:
            time.sleep(min(15.0 / SCALE, max(0.05, t_end - (time.monotonic() - T0 + state["offset"]))))
            w1 = counters()
            if w1[2] - w0[2] > 2000 and (w1[3] - w0[3]) > 0.03 * (w1[2] - w0[2]) and not state["held"]:
                state["held"] = 1; state["rate"] = prev_rate or 1.0  # the LAN is dropping: climb no further
            w0 = w1
            k = kernel_complaints(max(2, int(15 / SCALE)))
            kerr += k
            if k and not state["abort"]:
                state["abort"] = 1
        c1, secs = counters(), max(1e-3, time.monotonic() - t_ph)
        with lock:
            r = sorted(rtts)
        q = lambda f: int(r[min(len(r) - 1, int(f * len(r)))] * 1e6) if r else 0
        line = ("%sP%d probe stage profile ph=%d rate=%d tx_kbs=%d rx_kbs=%d msgs=%d rtt_p50_us=%d rtt_p99_us=%d rtt_max_us=%d "
                "stalls50=%d stalls200=%d retrans=%d segs=%d peers=%d held=%d abort=%d kerr=%d" % (
                    TAG, ph, ph, int(state["rate"]), (c1[0] - c0[0]) / secs / 1e3, (c1[1] - c0[1]) / secs / 1e3, len(r), q(0.5), q(0.99),
                    int(r[-1] * 1e6) if r else 0, sum(x > 0.05 for x in r), sum(x > 0.2 for x in r), c1[3] - c0[3], c1[2] - c0[2],
                    state["connected"], state["held"], state["abort"], kerr))
        for _ in range(3):
            print(line, flush=True); time.sleep(2 / SCALE)
        if not state["held"]:
            prev_rate = float(rate)
    state["rate"] = 0.0
    time.sleep(60 / SCALE)  # keep echoing for boxes that started later
    state["stop"] = True


if __name__ == "__main__":
    main()
