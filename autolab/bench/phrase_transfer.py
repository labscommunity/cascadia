#!/usr/bin/env python3
"""Move the old entry box's learned phrase table to the new pipeline head once."""
import argparse
import functools
import hashlib
import http.server
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.request
import ngram_merge

DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
LIMIT = 16 + ngram_merge.MAX_CONTEXTS * 61


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        paths = {'/entry.bin': self.server.root / 'entry.bin', '/manifest.json': self.server.root / 'manifest.json'}
        path = paths.get(self.path)
        if path is None:
            self.send_error(404)
            return
        with path.open('rb') as f:
            self.send_response(200)
            self.send_header('Content-Length', str(path.stat().st_size))
            self.send_header('Connection', 'close')
            self.end_headers()
            shutil.copyfileobj(f, self.wfile)


def serve(source, root, port, seconds=600):
    ngram_merge.read(source)  # Never serve an unchecked or unrelated file.
    root.mkdir(parents=True, exist_ok=True)
    dest = root / 'entry.bin'
    shutil.copyfile(source, dest)
    data = dest.read_bytes()
    (root / 'manifest.json').write_text(json.dumps({'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}))
    with http.server.ThreadingHTTPServer(('0.0.0.0', port), Handler) as server:
        server.root = root
        timer = threading.Timer(seconds, server.shutdown)
        timer.daemon = True; timer.start()
        print('NG0-serve probe stage profile bytes=%d serving=1' % len(data), flush=True)
        try:
            server.serve_forever()
        finally:
            timer.cancel()


def fetch(target, source_url, state, timeout=90):
    state.mkdir(parents=True, exist_ok=True)
    marker = state / 'entry-seed.done'
    if marker.exists():
        print('NG8-skip probe stage profile already_seeded=1', flush=True)
        return
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            with DIRECT.open(source_url + '/manifest.json', timeout=5) as r:
                meta = json.loads(r.read(4096))
            if not 16 <= meta['size'] <= LIMIT:
                raise ValueError('invalid phrase table size')
            with DIRECT.open(source_url + '/entry.bin', timeout=10) as r:
                data = r.read(LIMIT + 1)
            if len(data) != meta['size'] or hashlib.sha256(data).hexdigest() != meta['sha256']:
                raise ValueError('phrase table checksum mismatch')
            seed = state / 'entry.bin'
            seed.write_bytes(data)
            old = ngram_merge.read(target) if target.exists() else {}
            new = ngram_merge.merge(old, ngram_merge.read(seed))
            backup = state / 'before-entry.bin'
            if target.exists() and not backup.exists():
                shutil.copyfile(target, backup)
            ngram_merge.write(target, new)
            marker.write_text(json.dumps(meta))
            print('NG8-done probe stage profile transferred=1 contexts_before=%d contexts_after=%d' % (len(old), len(new)), flush=True)
            return
        except (OSError, ValueError, KeyError) as e:
            last = e
            time.sleep(2)
    raise RuntimeError('phrase transfer timed out: %s' % last)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('serve'); s.add_argument('source'); s.add_argument('root'); s.add_argument('--port', type=int, default=9205)
    s = sub.add_parser('fetch'); s.add_argument('target'); s.add_argument('source_url'); s.add_argument('state')
    a = ap.parse_args()
    if a.mode == 'serve':
        serve(Path(a.source), Path(a.root), a.port)
    else:
        fetch(Path(a.target), a.source_url, Path(a.state))


if __name__ == '__main__':
    main()
