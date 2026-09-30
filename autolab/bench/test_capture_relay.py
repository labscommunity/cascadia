"""Exercise the real read-only server and relay, including existing routes."""
import asyncio
import functools
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

FLEET = Path(__file__).resolve().parents[2] / 'deploy/inkling-fleet/fleet'


def module(name):
    spec = importlib.util.spec_from_file_location(name, FLEET / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


relay, captures = module('api_relay'), module('capture_server')


class CaptureRelayTest(unittest.IsolatedAsyncioTestCase):
    async def test_routes_are_restricted_and_existing_routes_still_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = 'r10-s0-123-1-1.bin'
            (root / name).write_bytes(b'capture bytes')
            (root / 'index.jsonl').write_text(json.dumps({'file': name}) + '\n')
            (root / 'secret').write_bytes(b'private')
            (root / 'r10-s0-123-1-2.bin').symlink_to(root / 'secret')
            telemetry = root / 'telemetry.json'
            telemetry.write_text('{"ranks": {}}')
            server = captures.Server(('127.0.0.1', 0), functools.partial(captures.Handler, directory=str(root)))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            capture_port = server.server_address[1]

            async def upstream(reader, writer):
                await reader.readuntil(b'\r\n\r\n')
                writer.write(b'HTTP/1.1 200 OK\r\nConnection: close\r\n\r\ndata: hello\n\n')
                await writer.drain()
                writer.close()

            app = await asyncio.start_server(upstream, '127.0.0.1', 0)
            args = types.SimpleNamespace(upstream='127.0.0.1:%d' % app.sockets[0].getsockname()[1],
                                         telemetry=str(telemetry), capture_fleet='fixture', capture_port=capture_port)
            front = await asyncio.start_server(lambda r, w: relay.handle(r, w, args), '127.0.0.1', 0)
            port = front.sockets[0].getsockname()[1]
            original_connect = asyncio.open_connection

            async def connect(host, port, *args, **kwargs):
                if host == 'fixture-rank-10':
                    host = '127.0.0.1'
                return await original_connect(host, port, *args, **kwargs)

            async def request(path, method='GET', direct=False):
                r, w = await original_connect('127.0.0.1', capture_port if direct else port)
                w.write(('%s %s HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\n\r\n' % (method, path)).encode())
                await w.drain()
                body = await asyncio.wait_for(r.read(), 3)
                w.close()
                await w.wait_closed()
                return body

            try:
                with patch.object(relay.asyncio, 'open_connection', connect):
                    response = await request('/api/fleet/capture/10/' + name)
                    self.assertIn(b'200 OK', response)
                    self.assertTrue(response.endswith(b'capture bytes'))
                    for path in ('/api/fleet/capture/11/' + name, '/api/fleet/capture/10/../secret',
                                 '/api/fleet/capture/10/%2e%2e/secret', '/api/fleet/capture/10/r10-s0-123-1-2.bin'):
                        self.assertIn(b'404', await request(path))
                    self.assertIn(b'404', await request('/api/fleet/capture/10/' + name, 'POST'))
                    for path in ('/', '/secret', '/../secret', '/r10-s0-123-1-2.bin'):
                        self.assertIn(b'404', await request(path, direct=True))
                    self.assertIn(b'405', await request('/' + name, 'HEAD', direct=True))
                    self.assertTrue((await request('/api/fleet/telemetry')).endswith(b'{"ranks": {}}'))
                    self.assertTrue((await request('/v1/chat/completions', 'POST')).endswith(b'data: hello\n\n'))
                    args.capture_fleet = ''
                    self.assertIn(b'404', await request('/api/fleet/capture/10/' + name))
            finally:
                front.close(); app.close()
                await front.wait_closed(); await app.wait_closed()
                await asyncio.to_thread(server.shutdown)
                server.server_close(); worker.join()


if __name__ == '__main__':
    unittest.main()
