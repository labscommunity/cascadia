"""OpenVINO stage engine and an SDXL scheduler with explicit stage executors."""
from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import time
from .contracts import decode, encode, require, validate_sdxl_metadata, verify_bundle

DTYPES = {'f32':'<f4', 'f16':'<f2', 'i64':'<i8', 'i32':'<i4', 'u8':'u1'}
VERSIONS = {'numpy':'2.4.6', 'openvino':'2026.4.1', 'optimum-intel':'2.2.0', 'diffusers':'0.39.0', 'transformers':'5.5.4', 'torch':'2.15.0', 'Pillow':'13.0.0', 'tokenizers':'0.22.2', 'optimum':'2.3.0'}

def check_versions(coordinator=False):
    import importlib.metadata
    names=VERSIONS if coordinator else ('numpy','openvino')
    for name in names:
        actual=importlib.metadata.version(name).split('+',1)[0]
        require(actual==VERSIONS[name], f'unsupported {name} version: {actual}; expected {VERSIONS[name]}')



def arrays(values, ports):
    import numpy as np
    require(set(values) == {p['name'] for p in ports}, 'input names differ from artifact')
    result = []
    for port in ports:
        value = np.asarray(values[port['name']], dtype=DTYPES[port['dtype']])
        if value.shape == () and port['shape'] == [1]:
            value = value.reshape(1)
        require(list(value.shape) == port['shape'], 'input shape differs from artifact')
        require(bool(np.isfinite(value).all()), 'nonfinite input')
        result.append(np.ascontiguousarray(value))
    return result


def frame(header, values, ports):
    prepared = arrays(values, ports)
    return encode(header | {'tensors':ports}, [memoryview(v).cast('B') for v in prepared])


def unframe(raw, ports):
    import numpy as np
    header, values = decode(raw, ports)
    return header, {port['name']: np.frombuffer(value, dtype=DTYPES[port['dtype']]).reshape(port['shape']) for port,value in zip(ports,values)}


class Engine:
    def __init__(self, root: Path, manifest: str, components, device: str, cache: Path):
        check_versions()
        import openvino as ov
        require(not cache.resolve().is_relative_to(root.resolve()), 'cache must be outside immutable artifact')
        require(device in ('CPU','GPU'), 'supported devices are CPU and GPU')
        self.manifest = verify_bundle(root, manifest, components)
        self.id = manifest
        self.compiled = {}
        core = ov.Core()
        from .export import schema
        for name in components:
            contract = self.manifest['components'][name]
            model = core.read_model(root/contract['xml'], root/contract['weights'])
            require(schema(model) == {k:contract[k] for k in ('inputs','outputs')}, 'IR schema differs from manifest')
            # Keep accumulation precision explicit across full and cut graphs.
            # Quantized constants stay quantized; this is not dequantization.
            compiled = core.compile_model(model, device, {'CACHE_DIR':str(cache/manifest/device/name), 'PERFORMANCE_HINT':'LATENCY', 'INFERENCE_PRECISION_HINT':'f32', 'NUM_STREAMS':'1'})
            self.compiled[name] = compiled
            print(json.dumps({'event':'component_ready','component':name,'device':device,'manifest':manifest}),flush=True)

    def infer(self, name, values):
        import numpy as np
        require(name in self.compiled, 'component is not loaded')
        contract = self.manifest['components'][name]
        inputs = {p['name']:v for p,v in zip(contract['inputs'], arrays(values,contract['inputs']))}
        compiled = self.compiled[name]
        request = compiled.create_infer_request()
        outputs = request.infer(inputs, share_inputs=True)
        result = {p['name']:np.array(outputs[compiled.output(p['name'])],copy=True) for p in contract['outputs']}
        arrays(result, contract['outputs'])
        return result

    def execute(self, raw):
        header, _ = decode(raw)
        require(header['manifest']==self.id and header['component'] in self.compiled, 'unavailable manifest or component')
        name = header['component']
        contract = self.manifest['components'][name]
        _, values = unframe(raw, contract['inputs'])
        started=time.monotonic()
        output=self.infer(name,values)
        encoded=frame(header,output,contract['outputs'])
        print(json.dumps({'event':'stage_complete','component':name,'sequence':header['sequence'],'compute_ms':round((time.monotonic()-started)*1000),'input_bytes':len(raw),'output_bytes':len(encoded)}),flush=True)
        return encoded


def schema_model(contract, name):
    import numpy as np
    import openvino as ov
    inputs=[]
    for item in contract['inputs']:
        parameter=ov.opset13.parameter(item['shape'],np.dtype(DTYPES[item['dtype']]),name=item['name'])
        parameter.output(0).get_tensor().set_names({item['name']})
        inputs.append(parameter)
    outputs=[]
    for item in contract['outputs']:
        zero=ov.opset13.constant(np.array(0,dtype=DTYPES[item['dtype']]))
        value=ov.opset13.broadcast(zero,np.array(item['shape'],dtype=np.int64))
        value.output(0).get_tensor().set_names({item['name']})
        outputs.append(value.output(0))
    return ov.Model(outputs,inputs,name+'_schema_only')


class Pipeline:
    """Tokenization and scheduler only; every model call needs an executor."""
    def __init__(self, root: Path, manifest: str, mode: str, invoke):
        check_versions(coordinator=True)
        import numpy as np
        from diffusers import EulerDiscreteScheduler
        from optimum.intel import OVStableDiffusionXLPipeline
        from transformers import CLIPTokenizer
        require(mode in ('components','layers'), 'invalid placement mode')
        self.manifest=verify_bundle(root,manifest,[])
        validate_sdxl_metadata(root)
        self.mode=mode
        self.invoke=invoke
        schemas=self.manifest['components']
        required={'text_encoder','text_encoder_2','vae_decoder','unet'}
        require(required <= schemas.keys(), 'missing SDXL schema')
        if mode=='layers':
            require({'unet_first','unet_second'} <= schemas.keys(), 'missing denoiser partitions')
        models={name:schema_model(schemas[name],name) for name in required}
        self.pipe=OVStableDiffusionXLPipeline(scheduler=EulerDiscreteScheduler.from_pretrained(root/'scheduler',local_files_only=True),
            tokenizer=CLIPTokenizer.from_pretrained(root/'tokenizer',local_files_only=True),
            tokenizer_2=CLIPTokenizer.from_pretrained(root/'tokenizer_2',local_files_only=True),
            **models,model_save_dir=root,device='CPU',compile=False,dynamic_shapes=False,add_watermarker=False)
        pipeline=self
        class Output:
            def __init__(self,name,values):self.name,self.values=name,values
            def __getitem__(self,key):return self.values[models[self.name].output(key).get_any_name()] if isinstance(key,int) else self.values[key]
            def to_dict(self):return {p:self.values[p.get_any_name()] for p in models[self.name].outputs}
        class Proxy:
            def __init__(self,name):self.name=name
            def __call__(self,inputs,**kwargs):
                values={k:np.asarray(v) for k,v in inputs.items()}
                if self.name=='unet' and mode=='layers':
                    first={p['name']:values[p['name']] for p in schemas['unet_first']['inputs']}
                    boundary=pipeline.invoke('unet_first',first)
                    combined=values|boundary
                    second={p['name']:combined[p['name']] for p in schemas['unet_second']['inputs']}
                    output=pipeline.invoke('unet_second',second)
                else:
                    output=pipeline.invoke(self.name,values)
                return Output(self.name,output)
        for name in models:
            part=getattr(self.pipe,name)
            part.request=Proxy(name)
            def guard_compile(part=part):
                require(part.request is not None, 'metadata-only graph has no executor')
            part.compile=guard_compile
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, body):
        import torch
        require(isinstance(body,dict) and set(body) <= {'model','prompt','negative_prompt','n','size','num_inference_steps','guidance_scale','rng_seed','response_format'}, 'invalid generation fields')
        require(isinstance(body.get('prompt'),str) and 0<len(body['prompt'].strip().encode())<=16384, 'invalid prompt')
        negative=body.get('negative_prompt')
        require(negative is None or isinstance(negative,str) and len(negative.encode())<=16384, 'invalid negative prompt')
        steps=body.get('num_inference_steps',20)
        seed=body.get('rng_seed',42)
        guidance=body.get('guidance_scale',7.5)
        from .contracts import integer
        integer(steps,1,50)
        integer(seed,0,2**32-1)
        require(type(guidance) in (int,float) and 1<guidance<=30, 'SDXL fixed batch requires guidance in (1,30]')
        width,height=self.manifest['width'],self.manifest['height']
        require(body.get('size',f'{width}x{height}')==f'{width}x{height}', 'size differs from artifact')
        require(body.get('n',1)==1 and body.get('response_format','b64_json')=='b64_json', 'one base64 PNG is supported')
        image=self.pipe(prompt=body['prompt'],negative_prompt=negative,height=height,width=width,num_inference_steps=steps,guidance_scale=guidance,generator=torch.Generator().manual_seed(seed)).images[0]
        buffer=io.BytesIO()
        image.save(buffer,format='PNG')
        return {'data':[{'b64_json':base64.b64encode(buffer.getvalue()).decode('ascii')}]}
