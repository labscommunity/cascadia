"""Authenticated loopback sidecar. Fleet admission and leases live in Rust.

A request thread retains its compute lock after its client disconnects. The
supervisor must quarantine an ambiguous exchange until this process is stopped.
"""
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import socketserver
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from .contracts import MAX_FRAME, exact, integer, parse_json, require


class BoundedServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads=False
    block_on_close=True
    request_queue_size=8
    def __init__(self,*args,**kwargs):
        self.slots=threading.BoundedSemaphore(8)
        super().__init__(*args,**kwargs)
    def process_request(self,request,address):
        if not self.slots.acquire(False):
            self.shutdown_request(request)
            return
        try:super().process_request(request,address)
        except BaseException:
            self.slots.release()
            raise
    def process_request_thread(self,request,address):
        try:super().process_request_thread(request,address)
        finally:self.slots.release()


def serve(config_path: Path):
    from .runtime import Engine, Pipeline, frame, unframe
    config=parse_json(config_path.read_bytes())
    exact(config, ('artifact','manifest','components','device','cache','port','coordinator'))
    integer(config['port'],1024,65535)
    token=os.environ.pop('CASCADIA_DIFFUSION_TOKEN','')
    require(len(token)>=32 and token.isascii(), 'supervisor token required')
    engine=Engine(Path(config['artifact']),config['manifest'],config['components'],config['device'],Path(config['cache']))
    execute_slot=threading.Lock()
    generate_slot=threading.Lock()
    boot=str(uuid.uuid4())
    coordinator=config['coordinator']
    pipeline=None
    job=None
    sequence=0
    if coordinator is not None:
        exact(coordinator,('mode','callback_url','callback_boot'))
        callback=urllib.parse.urlparse(coordinator['callback_url'])
        require(callback.scheme=='http' and callback.hostname=='127.0.0.1' and callback.port and callback.path=='/invoke' and not callback.username and not callback.query and not callback.fragment, 'callback must be an explicit loopback /invoke URL')
        require(str(uuid.UUID(coordinator['callback_boot']))==coordinator['callback_boot'], 'invalid coordinator boot')
        callback_token=os.environ.pop('CASCADIA_DIFFUSION_CALLBACK_TOKEN','')
        require(len(callback_token)>=32 and callback_token.isascii(), 'callback token required')
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        def invoke(name,values):
            nonlocal sequence
            sequence+=1
            require(job is not None, 'generation has no admitted job')
            header={'manifest':config['manifest'],'component':name,'lease':{'boot':coordinator['callback_boot'],'id':job},'sequence':sequence}
            contract=engine.manifest['components'][name]
            raw=frame(header,values,contract['inputs'])
            request=urllib.request.Request(coordinator['callback_url'],data=raw,headers={'Authorization':'Bearer '+callback_token,'Content-Type':'application/octet-stream'})
            with opener.open(request,timeout=3600) as response:
                result=response.read(MAX_FRAME+1)
            reply,output=unframe(result,contract['outputs'])
            require(all(reply[k]==header[k] for k in header), 'callback returned another execution')
            return output
        pipeline=Pipeline(Path(config['artifact']),config['manifest'],coordinator['mode'],invoke)
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def setup(self):
            super().setup()
            self.connection.settimeout(30)
        def log_message(self,*args):pass
        def respond(self,status,body,binary=False):
            raw=body if binary else json.dumps(body,separators=(',',':')).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/octet-stream' if binary else 'application/json')
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Connection','close')
            self.end_headers()
            self.close_connection=True
            try:self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError,TimeoutError):pass
        def authorized(self):
            values=self.headers.get_all('Authorization',[])
            return len(values)==1 and hmac.compare_digest(values[0].encode(),('Bearer '+token).encode())
        def do_GET(self):
            if not self.authorized():return self.respond(401,{'error':'unauthorized'})
            if self.path!='/health':return self.respond(404,{'error':'unknown route'})
            self.respond(200,{'ready':True,'boot':boot,'manifest':config['manifest'],'components':config['components'],'coordinator':pipeline is not None})
        def do_POST(self):
            nonlocal job,sequence
            if not self.authorized():return self.respond(401,{'error':'unauthorized'})
            if self.path not in ('/execute','/generate'):return self.respond(404,{'error':'unknown route'})
            lengths=self.headers.get_all('Content-Length',[])
            cap=MAX_FRAME if self.path=='/execute' else 65536
            if self.headers.get('Transfer-Encoding') or len(lengths)!=1 or not lengths[0].isascii() or not lengths[0].isdecimal():return self.respond(400,{'error':'invalid framing'})
            length=int(lengths[0])
            if not 0<length<=cap:return self.respond(413,{'error':'body exceeds limit'})
            lock=execute_slot if self.path=='/execute' else generate_slot
            if not lock.acquire(False):return self.respond(503,{'error':'busy'})
            try:
                raw=self.rfile.read(length)
                require(len(raw)==length, 'truncated request')
                if self.path=='/execute':
                    result=engine.execute(raw)
                    self.respond(200,result,True)
                else:
                    require(pipeline is not None, 'coordinator disabled')
                    body=parse_json(raw)
                    exact(body,('job','request'))
                    require(str(uuid.UUID(body['job']))==body['job'], 'invalid job')
                    job=body['job']
                    sequence=0
                    result=pipeline.generate(body['request'])
                    self.respond(200,result)
            except ValueError:
                self.respond(400,{'error':'invalid stage request'})
            except Exception as error:
                print(json.dumps({'event':'execution_failed','type':type(error).__name__}),flush=True)
                self.respond(500,{'error':'execution failed'})
            finally:
                if self.path=='/generate':job=None
                lock.release()
    server=BoundedServer(('127.0.0.1',config['port']),Handler)
    print(json.dumps({'event':'ready','boot':boot,'manifest':config['manifest'],'port':config['port']}),flush=True)
    try:server.serve_forever(poll_interval=.25)
    finally:server.server_close()
