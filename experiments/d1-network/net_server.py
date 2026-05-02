import socket, time, sys
PORT = 51234
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", PORT))
s.listen(1)
print(f"server listening on :{PORT}", flush=True)
conn, addr = s.accept()
conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
print(f"connected from {addr}", flush=True)
# bandwidth test: receive 1 GB
total = 1024 * 1024 * 1024
got = 0
t0 = time.perf_counter()
while got < total:
    data = conn.recv(min(1024*1024, total-got))
    if not data: break
    got += len(data)
dt = time.perf_counter() - t0
print(f"BW_RX: {got/1e6/dt:.1f} MB/s ({got*8/1e9/dt:.2f} Gbps) over {dt:.2f}s")

# Latency test: 1000 ping-pongs of 8 bytes
n = 1000
t0 = time.perf_counter()
for _ in range(n):
    conn.send(b"01234567")
    conn.recv(8)
dt = time.perf_counter() - t0
print(f"LATENCY: {dt/n*1000:.3f} ms/round-trip ({n} round-trips in {dt:.2f}s)")
conn.close(); s.close()
