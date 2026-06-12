# B-side (rank 1): speaks the proposed M4' frame sequence.
import socket, struct
def recv_exact(c, n):
    b = b""
    while len(b) < n:
        d = c.recv(n - len(b))
        if not d: raise SystemExit("peer closed")
        b += d
    return b
def recv_frame(c):
    hdr = recv_exact(c, 12)          # [u32 kind][u32 epoch][u32 len]
    kind, epoch, ln = struct.unpack("<III", hdr)
    return kind, epoch, recv_exact(c, ln)
def send_frame(c, kind, epoch, payload=b""):
    c.sendall(struct.pack("<III", kind, epoch, len(payload)) + payload)
HANDSHAKE, RESET, RESET_ACK, POS, ACT, TOKEN, STALE_TEST, DROPPED = range(8)
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 19998)); s.listen(1)
print("listening", flush=True)
c, _ = s.accept()
c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
current_epoch = 0
while True:
    kind, epoch, payload = recv_frame(c)
    if kind == HANDSHAKE:
        # echo back: peer compares manifest-hash/ov-version/dtype
        send_frame(c, HANDSHAKE, epoch, payload)
    elif kind == RESET:
        current_epoch = epoch
        send_frame(c, RESET_ACK, epoch)
    elif kind == POS:
        pass  # position prefix, no reply
    elif kind == ACT:
        if epoch != current_epoch:
            send_frame(c, DROPPED, epoch)
        else:
            send_frame(c, TOKEN, epoch, struct.pack("<I", 369))
    elif kind == STALE_TEST:
        send_frame(c, DROPPED if epoch != current_epoch else TOKEN, epoch)
