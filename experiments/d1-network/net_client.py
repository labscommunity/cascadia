import socket, time, sys
PORT = 51234
HOST = sys.argv[1]
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
s.connect((HOST, PORT))
print(f"connected to {HOST}:{PORT}", flush=True)
# send 1 GB
total = 1024 * 1024 * 1024
chunk = b"X" * (1024 * 1024)
sent = 0
while sent < total:
    s.send(chunk)
    sent += len(chunk)
print("done sending 1 GB", flush=True)
# latency: 1000 ping-pongs
n = 1000
for _ in range(n):
    msg = s.recv(8)
    s.send(msg)
print("done with latency probes", flush=True)
s.close()
