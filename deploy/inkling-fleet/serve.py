#!/usr/bin/env python3
"""Serve this SSD over the fleet's LAN, for the boxes that cannot read it.

The SSD is ext4 and Windows cannot mount it (it offers to format it: say no).
Plug the SSD into one Ubuntu box on the switch and run

    python3 /media/$USER/<ssd>/inkling-deploy/serve.py

then, on each Windows box, type the two lines this prints. Read-only HTTP on
port 8080: the kit (inkling-deploy/), the model export (inkling/out/) and
/filelist.txt ("size<TAB>path" for every file; the Windows installer reads it
instead of directory listings). Nothing is written to the SSD. Ctrl-C stops it.
"""
import argparse, os, socket, subprocess, sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

SERVED = ("inkling-deploy/", "inkling/out/")
SKIP_DIRS = {"build", "__pycache__"}


def file_list(root):
    lines = []
    for top in SERVED:
        for d, dirs, files in os.walk(os.path.join(root, top)):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
            for f in sorted(files):
                if f.startswith("."):
                    continue
                p = os.path.join(d, f)
                lines.append("%d\t%s" % (os.path.getsize(p), os.path.relpath(p, root).replace(os.sep, "/")))
    return ("\n".join(lines) + "\n").encode()


class Handler(SimpleHTTPRequestHandler):
    listing = b""

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/filelist.txt":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(self.listing)))
            self.end_headers()
            self.wfile.write(self.listing)
            return
        if not path.lstrip("/").startswith(SERVED) or "/../" in path:
            self.send_error(404)
            return
        super().do_GET()

    def list_directory(self, path):  # names come from /filelist.txt
        self.send_error(404)

    def log_message(self, fmt, *args):
        if not (len(args) > 1 and str(args[1]) == "200"):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


def addresses():
    """(interface, address) of this box's real network ports (no containers, bridges or VPNs)."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "scope", "global"], capture_output=True, text=True).stdout
        found = [(l.split()[1], l.split()[3].split("/")[0]) for l in out.splitlines() if len(l.split()) > 3]
        found = [x for x in found if not x[0].startswith(("docker", "br-", "veth", "virbr", "tailscale", "wg", "tun"))]
    except OSError:
        found = []
    return found or [("?", socket.gethostbyname(socket.gethostname()))]


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="folder holding inkling-deploy/ and inkling/out/ (default: the SSD this script is on)")
    a = ap.parse_args()
    for top in SERVED:
        if not os.path.isdir(os.path.join(a.root, top)):
            sys.exit("missing %s under %s" % (top, a.root))
    print("indexing %s ..." % a.root, flush=True)
    Handler.listing = file_list(a.root)
    print("%d files. Serving on port %d; leave this running while the Windows boxes install." % (Handler.listing.count(b"\n"), a.port))
    print("\nOn each Windows box, in PowerShell as administrator (use the address on the fleet's switch):\n")
    for nic, ip in addresses():
        url = "http://%s:%d" % (ip, a.port)
        print("  # via %s" % nic)
        print("  Set-ExecutionPolicy -Scope Process Bypass -Force")
        print("  curl.exe -s -o $env:TEMP\\bootstrap.ps1 %s/inkling-deploy/bootstrap.ps1; & $env:TEMP\\bootstrap.ps1 -Rank <rank> -Server %s\n" % (url, url))
    ThreadingHTTPServer(("0.0.0.0", a.port), partial(Handler, directory=a.root)).serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
