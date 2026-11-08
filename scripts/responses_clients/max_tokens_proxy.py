"""Test harness for the hardware Codex run (a real model, not the mock): forwards to the Cascadia API, adding
`max_output_tokens` to every /v1/responses body that lacks it (Codex never sends one, and an unset limit can be
too short for a real model's reasoning plus tool call). Streams the response through unchanged. It proves
nothing by itself; responses-clients-check.sh does not use it (the mock needs no limit).
Usage: python3 max_tokens_proxy.py LISTEN_PORT UPSTREAM_BASE MAX_OUTPUT_TOKENS [REQUEST_LOG]   (e.g. 18090 http://127.0.0.1:18100 8192)
REQUEST_LOG (optional): append each request path + body (no headers) as a JSON line.
"""
import http.client, http.server, json, sys, urllib.parse

PORT, UP, CAP = int(sys.argv[1]), urllib.parse.urlparse(sys.argv[2]), int(sys.argv[3])
LOG = sys.argv[4] if len(sys.argv) > 4 else None

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def _fwd(self, method):
        n = int(self.headers.get("content-length", 0)); body = self.rfile.read(n) if n else b""
        if method == "POST" and self.path.endswith("/responses") and body:
            obj = json.loads(body)
            obj.setdefault("max_output_tokens", CAP)
            body = json.dumps(obj).encode()
            if LOG:
                with open(LOG, "a") as f: f.write(json.dumps({"path": self.path, "body": obj}) + "\n")
        c = http.client.HTTPConnection(UP.hostname, UP.port, timeout=1800)
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
        hdrs["content-length"] = str(len(body))
        c.request(method, self.path, body=body or None, headers=hdrs)
        r = c.getresponse()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"): self.send_header(k, v)
        self.send_header("transfer-encoding", "chunked"); self.send_header("connection", "close"); self.end_headers()
        while True:
            chunk = r.read1(65536)
            if not chunk: break
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
    def do_POST(self): self._fwd("POST")
    def do_GET(self): self._fwd("GET")
    def log_message(self, *a): pass

http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
