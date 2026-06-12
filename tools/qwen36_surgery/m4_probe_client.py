# A-side (rank 0): drives the sequence, measures everything.
import socket, struct, sys, time, statistics, json
HOST = sys.argv[1]
def recv_exact(c, n):
    b = b""
    while len(b) < n:
        d = c.recv(n - len(b))
        if not d: raise SystemExit("peer closed")
        b += d
    return b
def recv_frame(c):
    hdr = recv_exact(c, 12)
    kind, epoch, ln = struct.unpack("<III", hdr)
    return kind, epoch, recv_exact(c, ln)
def send_frame(c, kind, epoch, payload=b""):
    c.sendall(struct.pack("<III", kind, epoch, len(payload)) + payload)
HANDSHAKE, RESET, RESET_ACK, POS, ACT, TOKEN, STALE_TEST, DROPPED = range(8)
c = socket.create_connection((HOST, 19998), timeout=30)
c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
# 1. handshake
hs = json.dumps({"manifest": "qwen36-2stage-fake-hash", "ov": "2026.2", "dtype": "f32"}).encode()
t0 = time.perf_counter()
send_frame(c, HANDSHAKE, 0, hs)
k, e, p = recv_frame(c)
assert k == HANDSHAKE and p == hs, "handshake mismatch"
print(f"handshake ok {1000*(time.perf_counter()-t0):.1f} ms", flush=True)
# 2. RESET/ACK
epoch = 1
t0 = time.perf_counter()
send_frame(c, RESET, epoch)
k, e, _ = recv_frame(c)
assert k == RESET_ACK and e == epoch
print(f"reset/ack ok {1000*(time.perf_counter()-t0):.1f} ms", flush=True)
# 3. prefill: 4 chunks of [pos][2MB act]; timed individually (with TOKEN ack on last only is realistic,
#    but ack each to measure delivery time honestly)
chunk = b"a" * (256*2048*4)  # 2 MiB
pre = []
t_pre = time.perf_counter()
for i in range(4):
    t0 = time.perf_counter()
    send_frame(c, POS, epoch, struct.pack("<q", i*256))
    send_frame(c, ACT, epoch, chunk)
    k, e, p = recv_frame(c)
    assert k == TOKEN
    pre.append(time.perf_counter() - t0)
t_pre = time.perf_counter() - t_pre
print(f"prefill 4x2MB: total {t_pre:.2f} s per-chunk {[f'{x:.2f}' for x in pre]}", flush=True)
# 4. decode: 64 tokens of [pos][8KB act] -> token
act = b"d" * (2048*4)
lat = []
for i in range(64):
    t0 = time.perf_counter()
    send_frame(c, POS, epoch, struct.pack("<q", 1024+i))
    send_frame(c, ACT, epoch, act)
    k, e, p = recv_frame(c)
    assert k == TOKEN and e == epoch
    lat.append((time.perf_counter()-t0)*1000)
lat.sort()
p50, p95 = statistics.median(lat), lat[60]
print(f"decode 64x8KB: p50 {p50:.2f} ms p95 {p95:.2f} ms min {lat[0]:.2f} max {lat[-1]:.2f}", flush=True)
# 5. stale epoch drop
send_frame(c, STALE_TEST, epoch-1)
k, e, _ = recv_frame(c)
assert k == DROPPED, "stale frame NOT dropped"
print("stale-epoch drop ok", flush=True)
# gates
g1 = t_pre < 5.0
g2 = p95 < 25.0
print(f"GATE prefill<5s: {'PASS' if g1 else 'FAIL'} ({t_pre:.2f}s) | decode p95<25ms: {'PASS' if g2 else 'FAIL'} ({p95:.2f}ms)", flush=True)
print("DAY0_PROBE_" + ("PASS" if g1 and g2 else "FAIL"), flush=True)
