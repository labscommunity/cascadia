#!/usr/bin/env python3
"""Role swap, step 1: two boxes give each other the model files of their pipeline roles, over the LAN, verified.

    role_sync.py --prefix /opt/cascadia-inkling --fleet inkling --box 0 --peer 8 [--per 6] [--port 9203]
                 [--mbps 60] [--reserve-gb 70] [--hours 3]

Runs on BOTH boxes of the pair (started by the fleet overrides inside the worker's unit; standard library only).
Nothing here changes what a box runs: `run.sh` swaps the roles later, and only on a box that holds the marker this
script writes last (`<prefix>/role-swap/ready-<role>`).

What each box does:
  * serves `<prefix>/model` read-only on --port (weights, tokenizer, IRs: nothing secret lives there; `<prefix>`
    itself, which holds the fleet key, is never served), plus `model/.role-swap/`:
      files-<role>.txt   "sha256 size path" of what the peer needs for THIS box's installed role: the layers'
                         shells, experts and attention IRs, and the embedding on role 0. Fused-MoE and dense IRs
                         are not copied: run.sh regenerates them from the experts before the worker starts.
      verified           appears when this box holds the peer's role, every byte checked
  * downloads the peer's list into the same model folder (other layer numbers: no path collides), each file to
    `<name>.part`, hashed while it streams, renamed when the hash matches; present files with the right hash are
    kept (resumable), nothing is ever deleted or overwritten with different bytes
  * refuses to start if the disk cannot hold the files plus --reserve-gb for the IRs to be generated
  * writes `role-swap/ready-<peer role>` once BOTH boxes are verified, and says where it is on a "stage profile"
    line every 30 s (the beacon relays those): SW<box> ... files= done= gb_total= gb_done= verified= peer_verified=
    ready= free_gb= err=
"""
import argparse, functools, hashlib, http.server, json, os, socketserver, sys, threading, time, urllib.request

DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # fleet traffic never goes through a web proxy
STATE = dict(files=0, done=0, gb_total=0, gb_done=0, verified=0, peer_verified=0, ready=0, free_gb=0, err=0)
ERRORS = {1: "peer list unavailable", 2: "disk too small", 3: "hash mismatch", 4: "download failed", 5: "own list failed"}


def log(msg):
    print(time.strftime("%H:%M:%S"), "role sync:", msg, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def role_files(model, role, per):
    """Relative paths a box needs to play `role`, beyond what every box already holds."""
    out = []
    for layer in range(role * per, role * per + per):
        out.append("shells/layer_%02d.safetensors" % layer)
        for sub in ("experts/layer_%02d" % layer, "attn_ov/layer_%02d" % layer):
            for root, _, names in os.walk(os.path.join(model, sub)):
                out += [os.path.relpath(os.path.join(root, n), model) for n in sorted(names) if not n.endswith(".part")]
    if role == 0:
        out.append("embed.safetensors")
    return [p for p in out if os.path.isfile(os.path.join(model, p))]


def write_own_list(model, role, per):
    dst = os.path.join(model, ".role-swap", "files-%d.txt" % role)
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    files = role_files(model, role, per)
    if not any(p.startswith("experts/") for p in files):
        raise RuntimeError("no expert files found for role %d under %s" % (role, model))
    t0 = time.time()
    lines = ["%s %d %s" % (sha256_file(os.path.join(model, p)), os.path.getsize(os.path.join(model, p)), p) for p in files]
    with open(dst + ".tmp", "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(dst + ".tmp", dst)
    log("own list: %d files of role %d hashed in %.0f s" % (len(lines), role, time.time() - t0))


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.send_error(405)

    do_PUT = do_DELETE = do_POST

    def list_directory(self, path):   # no browsing: the peer knows the names it wants
        self.send_error(403)
        return None


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(model, port):
    srv = Server(("0.0.0.0", port), functools.partial(Quiet, directory=model))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def fetch_text(url, timeout=15):
    with DIRECT.open(url, timeout=timeout) as r:
        return r.read().decode()


def download(url, dst, want_sha, want_size, mbps):
    part = dst + ".part"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    h, n, t0 = hashlib.sha256(), 0, time.time()
    with DIRECT.open(url, timeout=60) as r, open(part, "wb") as f:
        while True:
            chunk = r.read(4 << 20)
            if not chunk:
                break
            f.write(chunk); h.update(chunk); n += len(chunk)
            if mbps > 0:   # leave the wire to the pipeline's frames
                ahead = n / (mbps * 1e6) - (time.time() - t0)
                if ahead > 0:
                    time.sleep(min(ahead, 1.0))
        f.flush(); os.fsync(f.fileno())
        try:   # 50 GB of fresh page cache is memory the iGPU's pages compete with
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    if n != want_size or h.hexdigest() != want_sha:
        os.remove(part)
        return False
    os.replace(part, dst)
    return True


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def reporter(box, role):
    while True:
        line = "SW%d probe stage profile box=%d role=%d %s" % (box, box, role, " ".join("%s=%d" % kv for kv in STATE.items()))
        print(line, flush=True)
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True); ap.add_argument("--fleet", required=True)
    ap.add_argument("--box", type=int, required=True, help="this box's INSTALLED rank")
    ap.add_argument("--peer", type=int, required=True, help="the other box's installed rank = the role to receive")
    ap.add_argument("--per", type=int, default=6, help="layers per rank")
    ap.add_argument("--port", type=int, default=9203); ap.add_argument("--mbps", type=float, default=60)
    ap.add_argument("--reserve-gb", type=float, default=70); ap.add_argument("--hours", type=float, default=3)
    ap.add_argument("--peer-url", default="", help="override http://<fleet>-rank-<peer>:<port> (tests)")
    a = ap.parse_args()
    model = os.path.join(a.prefix, "model")
    ready = os.path.join(a.prefix, "role-swap", "ready-%d" % a.peer)
    peer = a.peer_url or "http://%s-rank-%d:%d" % (a.fleet, a.peer, a.port)
    deadline = time.time() + a.hours * 3600
    STATE["free_gb"] = int(free_gb(model)); STATE["ready"] = int(os.path.exists(ready))
    threading.Thread(target=reporter, args=(a.box, a.peer), daemon=True).start()
    serve(model, a.port)
    try:
        write_own_list(model, a.box, a.per)
    except Exception as e:   # keep serving nothing useful, say so
        STATE["err"] = 5; log("cannot list own role: %s" % e)
    # ---- the peer's role ----
    flag = os.path.join(model, ".role-swap", "verified")
    while time.time() < deadline and not os.path.exists(flag):
        try:
            rows = [l.split(" ", 2) for l in fetch_text(peer + "/.role-swap/files-%d.txt" % a.peer).splitlines() if l.strip()]
        except Exception:
            STATE["err"] = 1; time.sleep(20); continue
        STATE["err"] = 0
        total = sum(int(r[1]) for r in rows); STATE["files"] = len(rows); STATE["gb_total"] = int(total / 1e9)
        todo, done_bytes = [], 0
        for sha, size, rel in rows:
            dst = os.path.join(model, rel)
            if os.path.isfile(dst) and os.path.getsize(dst) == int(size) and sha256_file(dst) == sha:
                done_bytes += int(size); STATE["done"] += 1; STATE["gb_done"] = int(done_bytes / 1e9)
            else:
                todo.append((sha, int(size), rel))
        need = sum(t[1] for t in todo) / 1e9
        STATE["free_gb"] = int(free_gb(model))
        if free_gb(model) < need + a.reserve_gb:
            STATE["err"] = 2; log("disk: %.0f GB free, %.0f GB to copy + %.0f GB for the IRs: not starting" % (free_gb(model), need, a.reserve_gb))
            time.sleep(300); STATE["done"] = 0; continue
        bad = 0
        for sha, size, rel in todo:
            if os.path.exists(os.path.join(model, rel)):   # same name, other bytes: never overwrite
                bad += 1; STATE["err"] = 3; log("refusing to replace %s (exists with other contents)" % rel); continue
            ok = False
            for _try in range(3):
                try:
                    ok = download(peer + "/" + urllib.request.quote(rel), os.path.join(model, rel), sha, size, a.mbps)
                except Exception as e:
                    log("%s: %s" % (rel, e)); time.sleep(5)
                if ok:
                    break
            if ok:
                done_bytes += size; STATE["done"] += 1; STATE["gb_done"] = int(done_bytes / 1e9)
            else:
                bad += 1; STATE["err"] = 4
        if bad == 0:
            open(flag, "w").write(json.dumps({"role": a.peer, "files": len(rows), "bytes": total, "time": time.time()}))
            log("role %d verified: %d files, %.1f GB" % (a.peer, len(rows), total / 1e9))
        else:
            time.sleep(60); STATE["done"] = 0
    STATE["verified"] = int(os.path.exists(flag))
    # ---- both verified -> this box may play the peer's role ----
    while time.time() < deadline:
        if STATE["verified"] and not STATE["ready"]:
            try:
                fetch_text(peer + "/.role-swap/verified")
                STATE["peer_verified"] = 1
                os.makedirs(os.path.dirname(ready), exist_ok=True)
                open(ready, "w").write("role %d data verified on both boxes %s\n" % (a.peer, time.strftime("%F %T")))
                STATE["ready"] = 1; log("ready: both boxes hold each other's role")
            except Exception:
                pass
        elif STATE["ready"]:
            STATE["peer_verified"] = 1
        time.sleep(20)   # keep serving: the peer may still be copying


if __name__ == "__main__":
    main()
