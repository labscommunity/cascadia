"""Loopback HTTP lifecycle checks with a deliberately blocked executor."""
import http.client
import http.server
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

FIXTURES=Path(__file__).resolve().parents[3]/'crates/cascadia-diffusion/tests/fixtures'
SOURCE=Path(__file__).resolve().parents[1]/'src'
TOKEN='contract-test-token-with-at-least-32-bytes'
CHILD=r'''
import json,sys,time
from pathlib import Path
from cascadia_diffusion import runtime
from cascadia_diffusion.contracts import decode
root=Path(sys.argv[1])
class FakeEngine:
    manifest={'components':{'unet':{'inputs':{},'outputs':{}}}}
    def __init__(self,*args):pass
    def execute(self,raw):
        decode(raw)
        (root/'entered').write_text('1')
        deadline=time.monotonic()+10
        while not (root/'finish').exists():
            if time.monotonic()>deadline:raise TimeoutError()
            time.sleep(.01)
        return raw
runtime.Engine=FakeEngine
runtime.frame=lambda header,values,contract:json.dumps(header).encode()
runtime.unframe=lambda raw,contract:(json.loads(raw),{})
class FakePipeline:
    def __init__(self,*args):self.invoke=args[3]
    def generate(self,body):
        if body.get('prompt')=='call a stage':self.invoke('unet',{})
        if body.get('prompt')=='reject after a stage':
            self.invoke('unet',{})
            raise ValueError('rejected output')
        if body.get('prompt')=='crash after a stage':
            self.invoke('unet',{})
            raise RuntimeError('engine crashed')
        if body.get('prompt')=='crash':raise RuntimeError('engine crashed')
        return {'prompt_bytes':len(body['prompt'].encode()),'negative_bytes':len(body.get('negative_prompt','').encode())}
runtime.Pipeline=FakePipeline
import socketserver
_write=socketserver._SocketWriter.write
def slow_write(self,data):
    # Hold the handler after each write so a lock kept across the reply shows up.
    written=_write(self,data)
    time.sleep(.05)
    return written
socketserver._SocketWriter.write=slow_write
import os,socket,threading
from cascadia_diffusion.server import BoundedServer,serve
knobs=json.loads(sys.argv[2]) if len(sys.argv)>2 else {}
for name in ('auth_deadline','auth_read_timeout'):
    if name in knobs:setattr(BoundedServer,name,knobs[name])
if knobs.get('nowake'):
    # Emulate Winsock: the deadline's shutdown does not wake a blocked read.
    _shutdown=socket.socket.shutdown
    socket.socket.shutdown=lambda self,how:None if how==socket.SHUT_RDWR else _shutdown(self,how)
# Publish the held slot count so a test can wait for every accept.
held=threading.Lock()
def report(server):
    with held:
        (root/'held.tmp').write_text(str(server.connections-server.slots._value))
        os.replace(root/'held.tmp',root/'held')
_process,_thread=BoundedServer.process_request,BoundedServer.process_request_thread
def process_request(self,*args):
    _process(self,*args)
    report(self)
def process_request_thread(self,*args):
    try:_thread(self,*args)
    finally:report(self)
BoundedServer.process_request,BoundedServer.process_request_thread=process_request,process_request_thread
serve(root/'config.json')
'''

def free_ports(count):
    # Hold every socket until all are bound, so the ports differ.
    socks=[socket.socket() for _ in range(count)]
    try:
        for sock in socks:sock.bind(('127.0.0.1',0))
        return [sock.getsockname()[1] for sock in socks]
    finally:
        for sock in socks:sock.close()

class Echo(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        raw=self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200)
        self.send_header('Content-Length',str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def log_message(self,*args):pass

class Server(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.port,self.callback_port=free_ports(2)
        config={'artifact':str(self.root),'manifest':'a'*64,'components':['unet'],'device':'CPU','cache':str(self.root/'cache'),'port':self.port,
                'coordinator':{'mode':'components','callback_url':f'http://127.0.0.1:{self.callback_port}/invoke','callback_boot':'00000000-0000-0000-0000-000000000001'}}
        (self.root/'config.json').write_text(json.dumps(config))
        self.launch()
    def launch(self,*args):
        env=os.environ|{'PYTHONPATH':str(SOURCE),'CASCADIA_DIFFUSION_TOKEN':TOKEN,'CASCADIA_DIFFUSION_CALLBACK_TOKEN':TOKEN}
        self.process=subprocess.Popen([sys.executable,'-c',CHILD,str(self.root),*args],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        self.wait(lambda:self.ready())
    def stop(self):
        self.process.terminate()
        return self.process.communicate(timeout=5)[1]
    def tearDown(self):
        (self.root/'finish').write_text('1')
        if self.process.returncode is None:self.stop()
        self.temp.cleanup()
    def request(self,method,path,body=None,token=TOKEN):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=2)
        try:
            conn.request(method,path,body,headers={'Authorization':'Bearer '+token})
            response=conn.getresponse()
            return response.status,response.read()
        finally:conn.close()
    def ready(self):
        if self.process.poll() is not None:
            raise RuntimeError(self.process.stderr.read().decode())
        try:return self.request('GET','/health')[0]==200
        except OSError:return False
    def wait(self,predicate,limit=5):
        deadline=time.monotonic()+limit
        while not predicate():
            if time.monotonic()>deadline:self.fail('condition timed out')
            time.sleep(.01)
    def test_auth_framing_and_finite_tensor_validation(self):
        self.assertEqual(self.request('GET','/health',token='wrong')[0],401)
        self.assertEqual(self.request('POST','/execute',b'invalid')[0],400)
        raw=(FIXTURES/'frame.bin').read_bytes()
        import struct
        self.assertEqual(self.request('POST','/execute',raw[:-4]+struct.pack('<f',float('inf')))[0],400)
        self.assertFalse((self.root/'entered').exists())
        sock=socket.create_connection(('127.0.0.1',self.port),2)
        try:
            sock.sendall((f'POST /execute HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TOKEN}\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nx').encode())
            self.assertIn(b' 400 ',sock.recv(4096))
        finally:sock.close()
    def test_disconnected_client_keeps_compute_slot_until_completion(self):
        raw=(FIXTURES/'frame.bin').read_bytes()
        sock=socket.create_connection(('127.0.0.1',self.port),2)
        sock.sendall((f'POST /execute HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TOKEN}\r\nContent-Length: {len(raw)}\r\n\r\n').encode()+raw)
        self.wait(lambda:(self.root/'entered').exists())
        sock.close()
        self.assertEqual(self.request('POST','/execute',raw)[0],503)
        (self.root/'finish').write_text('1')
        self.wait(lambda:self.request('POST','/execute',raw)[0]==200)

    def test_generation_envelope_accepts_a_maximum_public_request(self):
        request={'model':'sdxl-layers-512','prompt':'"'*16384,'negative_prompt':''}
        remaining=65536-len(json.dumps(request,separators=(',',':')).encode())
        request['negative_prompt']='"'*(remaining//2)+'x'*(remaining%2)
        self.assertEqual(len(json.dumps(request,separators=(',',':')).encode()),65536)
        self.assertLessEqual(len(request['negative_prompt'].encode()),16384)
        request.update(size='512x512',n=1,num_inference_steps=20,response_format='b64_json')
        payload=json.dumps({'job':'00000000-0000-0000-0000-000000000001','request':request},separators=(',',':')).encode()
        self.assertGreater(len(payload),65536)
        status,body=self.request('POST','/generate',payload)
        self.assertEqual(status,200)
        self.assertEqual(json.loads(body),{'prompt_bytes':16384,'negative_bytes':len(request['negative_prompt'])})
        # Reject the advertised size before reading a body. Sending a large
        # body here would race the server's early close against client writes.
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=2)
        try:
            conn.putrequest('POST','/generate')
            conn.putheader('Authorization','Bearer '+TOKEN)
            conn.putheader('Content-Length',str(256*1024+1))
            conn.endheaders()
            response=conn.getresponse()
            self.assertEqual(response.status,413)
            response.read()
        finally:conn.close()
    def test_generate_separates_bad_requests_from_failed_stage_exchanges(self):
        def generate(job,prompt):
            return self.request('POST','/generate',json.dumps({'job':job,'request':{'prompt':prompt}}).encode())
        job='00000000-0000-0000-0000-000000000002'
        self.assertEqual(generate(job,'call a stage'),(502,b'{"error":"stage exchange failed"}'))
        self.assertEqual(generate('not-a-uuid','x')[0],400)
        self.assertEqual(generate(job,'x')[0],200)
    def test_failure_after_a_completed_stage_is_uncertain(self):
        echo=http.server.HTTPServer(('127.0.0.1',self.callback_port),Echo)
        threading.Thread(target=echo.serve_forever,daemon=True).start()
        try:
            def generate(prompt):
                return self.request('POST','/generate',json.dumps({'job':'00000000-0000-0000-0000-000000000003','request':{'prompt':prompt}}).encode())
            for prompt in ('reject after a stage','crash after a stage'):
                self.assertEqual(generate(prompt),(502,b'{"error":"stage exchange failed"}'))
            self.assertEqual(generate('crash'),(500,b'{"error":"execution failed"}'))
            self.assertEqual(generate('call a stage')[0],200)
        finally:
            echo.shutdown()
            echo.server_close()
    def test_lock_is_free_when_the_reply_arrives(self):
        (self.root/'finish').write_text('1')
        raw=(FIXTURES/'frame.bin').read_bytes()
        for _ in range(3):self.assertEqual(self.request('POST','/execute',raw)[0],200)
    def relaunch(self,**knobs):
        self.stop()
        self.launch(json.dumps(knobs))
    def held(self):
        try:return int((self.root/'held').read_text())
        except (FileNotFoundError,ValueError):return 0
    def health(self):
        try:return self.request('GET','/health')[0]
        except OSError:return None
    def hold_every_slot(self):
        from cascadia_diffusion.server import BoundedServer
        # Connect one at a time and wait for each accept: a burst overflows the
        # listen backlog, and a socket still queued there holds no slot.
        self.wait(lambda:self.held()==0)
        idle=[]
        for count in range(1,BoundedServer.connections+1):
            idle.append(socket.create_connection(('127.0.0.1',self.port),2))
            # Half stall inside the request line, half send nothing.
            if count%2:idle[-1].sendall(b'G')
            self.wait(lambda:self.held()==count)
        return idle
    def assert_slots_free_within(self,limit):
        idle=self.hold_every_slot()
        try:
            with self.assertRaises(OSError):self.request('GET','/health')
            self.wait(lambda:self.health()==200,limit)
        finally:
            for sock in idle:sock.close()
    def test_idle_unauthorized_connections_cannot_hold_every_slot(self):
        # A long read timeout, so only the deadline can free the slots.
        self.relaunch(auth_deadline=2,auth_read_timeout=10)
        self.assert_slots_free_within(3)
        time.sleep(.2)
        self.assertNotIn(b'Traceback',self.stop())
    def test_read_timeout_frees_slots_when_shutdown_does_not_wake_a_read(self):
        self.relaunch(auth_deadline=.2,auth_read_timeout=1.5,nowake=True)
        self.assert_slots_free_within(2.5)
    def test_authorized_body_may_arrive_slower_than_the_pre_auth_read_timeout(self):
        self.relaunch(auth_deadline=.3,auth_read_timeout=.3)
        body=json.dumps({'job':'00000000-0000-0000-0000-000000000005','request':{'prompt':'x'}}).encode()
        sock=socket.create_connection(('127.0.0.1',self.port),5)
        try:
            sock.sendall(f'POST /generate HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TOKEN}\r\nContent-Length: {len(body)}\r\n\r\n'.encode())
            for part in (body[:10],body[10:20],body[20:]):
                time.sleep(.6)
                sock.sendall(part)
            self.assertIn(b' 200 ',sock.recv(4096))
        finally:sock.close()

if __name__=='__main__':unittest.main()
