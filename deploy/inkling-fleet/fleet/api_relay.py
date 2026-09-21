#!/usr/bin/env python3
"""Role swap, step 2: the entry box keeps answering on :8000 while another box plays pipeline rank 0.

    api_relay.py --listen 8000 --upstream inkling-rank-8:8000 [--telemetry /run/cascadia-inkling/telemetry.json]

The operator tunnel, the Tailscale address and every bookmark point at the entry box's port 8000. After a role swap
the API, the dashboard and the scheduler run on the box that plays rank 0; this relay keeps the old address working:
every connection is piped to the upstream's :8000, streaming responses (SSE) included.

One request is answered here: `GET /api/fleet/telemetry`. That file is written by the beacon of the box INSTALLED as
rank 0 (this one), so the API process on the other box does not have it. To make the check apply to every request,
the relay asks for one request per connection (`Connection: close` towards the upstream). Standard library only.
"""
import argparse, asyncio, os, sys, time

HEAD_LIMIT = 65536


def log(msg):
    print(time.strftime("%H:%M:%S"), "api relay:", msg, flush=True)


async def pipe(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError, OSError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


def one_request_per_connection(head):
    """The request head with `Connection: close` (replacing any Connection/Keep-Alive header)."""
    lines = head.split(b"\r\n")
    kept = [lines[0]] + [l for l in lines[1:] if l and not l.lower().startswith((b"connection:", b"keep-alive:", b"proxy-connection:"))]
    return b"\r\n".join(kept + [b"Connection: close", b"", b""])


async def handle(reader, writer, a):
    up_writer = None
    try:
        head = b""
        while b"\r\n\r\n" not in head and len(head) < HEAD_LIMIT:
            chunk = await asyncio.wait_for(reader.read(8192), timeout=30)
            if not chunk:
                return
            head += chunk
        if b"\r\n\r\n" not in head:
            writer.write(b"HTTP/1.1 431 Request Header Fields Too Large\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            return
        cut = head.index(b"\r\n\r\n")
        first = head[:cut].split(b"\r\n", 1)[0].split(b" ")
        path = first[1] if len(first) > 1 else b""
        if first[0] == b"GET" and path.split(b"?", 1)[0] == b"/api/fleet/telemetry":
            try:
                with open(a.telemetry, "rb") as f:
                    body = f.read()
            except OSError:
                body = b"{}"
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\n"
                         b"Connection: close\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            await writer.drain()
            return
        host, port = a.upstream.rsplit(":", 1)
        try:
            up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(host, int(port)), timeout=10)
        except (OSError, asyncio.TimeoutError) as e:
            body = ("the box that plays rank 0 (%s) does not answer: %s\n" % (a.upstream, e)).encode()
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: text/plain\r\nConnection: close\r\nContent-Length: "
                         + str(len(body)).encode() + b"\r\n\r\n" + body)
            await writer.drain()
            return
        up_writer.write(one_request_per_connection(head[:cut]) + head[cut + 4:])
        await up_writer.drain()
        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))
    except (asyncio.TimeoutError, ConnectionError, OSError):
        pass
    finally:
        for w in (writer, up_writer):
            try:
                if w is not None:
                    w.close()
            except Exception:
                pass


async def main_async(a):
    server = await asyncio.start_server(lambda r, w: handle(r, w, a), "0.0.0.0", a.listen, backlog=1024)
    log("listening on :%d -> %s (telemetry from %s)" % (a.listen, a.upstream, a.telemetry))
    async with server:
        await server.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8000)
    ap.add_argument("--upstream", required=True, help="host:port of the box that plays pipeline rank 0")
    ap.add_argument("--telemetry", default=os.environ.get("CASCADIA_FLEET_TELEMETRY_FILE", "/run/cascadia-inkling/telemetry.json"))
    a = ap.parse_args()
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 16384), hard), hard))   # two descriptors per request
    except Exception:
        pass
    try:
        asyncio.run(main_async(a))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
