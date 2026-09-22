#!/usr/bin/env python3
"""Read-only completed state captures, restricted to this directory and flat filenames."""
import argparse
import functools
import http.server
import pathlib
import re
import socketserver
import urllib.parse


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        name = urllib.parse.urlsplit(self.path).path.removeprefix('/')
        if name != 'index.jsonl' and not re.fullmatch(r'r[0-9]+-s[0-9]+-[0-9]+-[0-9]+-[0-9]+\.bin', name):
            self.send_error(404)
            return
        root = pathlib.Path(self.directory).resolve()
        path = root / name
        if path.is_symlink() or path.resolve().parent != root or not path.is_file():
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        self.send_error(405)

    def list_directory(self, path):
        self.send_error(404)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--port', type=int, default=9204)
    ap.add_argument('--bind', default='0.0.0.0')
    a = ap.parse_args()
    pathlib.Path(a.dir).mkdir(parents=True, exist_ok=True)
    with Server((a.bind, a.port), functools.partial(Handler, directory=a.dir)) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
