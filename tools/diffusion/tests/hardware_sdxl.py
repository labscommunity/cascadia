"""Opt-in hardware qualification. Run only on an authorized, idle test host."""
import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from cascadia_diffusion.runtime import Engine, Pipeline
from cascadia_diffusion.contracts import canonical

parser=argparse.ArgumentParser()
parser.add_argument('--artifact',type=Path,required=True)
parser.add_argument('--manifest',required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--mode',choices=('components','layers'),required=True)
parser.add_argument('--device',default='GPU')
parser.add_argument('--steps',type=int,default=20)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
components=['text_encoder','text_encoder_2','vae_decoder']+(['unet'] if args.mode=='components' else ['unet_first','unet_second'])
started=time.monotonic()
engine=Engine(args.artifact,args.manifest,components,args.device,args.output.parent/'cache')
calls=[]
def invoke(name,values):
    start=time.monotonic()
    result=engine.infer(name,values)
    calls.append({'component':name,'seconds':time.monotonic()-start,'input_bytes':sum(v.nbytes for v in values.values()),'output_bytes':sum(v.nbytes for v in result.values())})
    return result
pipe=Pipeline(args.artifact,args.manifest,args.mode,invoke)
report={'manifest':args.manifest,'mode':args.mode,'device':args.device,'components':components,'startup_seconds':time.monotonic()-started,'versions':{n:importlib.metadata.version(n) for n in ('openvino','optimum-intel','diffusers','transformers','torch','numpy','Pillow')},'runs':[]}
body={'prompt':'Black and white manga illustration, a young adult explorer standing beside a quiet forest shrine, detailed ink linework, screentone shading, balanced composition, no text','negative_prompt':'color, blurry, text, watermark','rng_seed':4242,'num_inference_steps':args.steps,'guidance_scale':7.5}
for i in range(3):
    calls.clear()
    start=time.monotonic()
    result=pipe.generate(body)
    raw=base64.b64decode(result['data'][0]['b64_json'],validate=True)
    (args.output/f'image-{i}.png').write_bytes(raw)
    record={'index':i,'cold':i==0,'seconds':time.monotonic()-start,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'calls':list(calls)}
    try:
        import psutil
        record['memory']=psutil.Process().memory_info()._asdict()
    except ImportError:pass
    report['runs'].append(record)
    (args.output/'report.json').write_bytes(canonical(report)+b'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='calls'}),flush=True)
if report['runs'][1]['sha256']!=report['runs'][2]['sha256']:
    raise RuntimeError('warm seeded repeat differs')
print(json.dumps({'passed':True,'report':str(args.output/'report.json')}),flush=True)
