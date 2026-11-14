import argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(prog='cascadia-diffusion')
    commands=parser.add_subparsers(dest='command',required=True)
    export=commands.add_parser('export-sdxl')
    export.add_argument('--source',type=Path,required=True)
    export.add_argument('--output',type=Path,required=True)
    export.add_argument('--revision',required=True)
    export.add_argument('--size',type=int,choices=(256,512,1024),default=512)
    for command in ('verify','pack'):
        sub=commands.add_parser(command)
        sub.add_argument('--source',type=Path,required=True)
        sub.add_argument('--manifest',required=True)
        sub.add_argument('--components',nargs='*',default=[])
        if command=='pack':sub.add_argument('--output',type=Path,required=True)
    generate=commands.add_parser('generate')
    generate.add_argument('--source',type=Path,required=True)
    generate.add_argument('--manifest',required=True)
    generate.add_argument('--mode',choices=('components','layers'),default='components')
    generate.add_argument('--device',choices=('CPU','GPU'),default='GPU')
    generate.add_argument('--cache',type=Path,required=True)
    generate.add_argument('--request',type=Path,required=True)
    generate.add_argument('--output',type=Path,required=True)
    serve=commands.add_parser('serve')
    serve.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='export-sdxl':
        from .export import export_sdxl
        export_sdxl(args.source,args.output,args.revision,args.size)
    elif args.command=='pack':
        from .export import pack
        pack(args.source,args.output,args.manifest,args.components)
    elif args.command=='generate':
        import base64
        from .contracts import parse_json, require
        from .runtime import Engine, Pipeline
        require(not args.output.exists(), 'output already exists')
        require(args.request.stat().st_size <= 65536, 'request exceeds limit')
        body=parse_json(args.request.read_bytes())
        components=['text_encoder','text_encoder_2','vae_decoder']+(['unet'] if args.mode=='components' else ['unet_first','unet_second'])
        engine=Engine(args.source,args.manifest,components,args.device,args.cache)
        pipeline=Pipeline(args.source,args.manifest,args.mode,engine.infer)
        result=pipeline.generate(body)
        with args.output.open('xb') as output:
            output.write(base64.b64decode(result['data'][0]['b64_json'],validate=True))
    elif args.command=='verify':
        from .contracts import verify_bundle
        verify_bundle(args.source,args.manifest,args.components)
        print('Verified')
    else:
        from .server import serve
        serve(args.config)

if __name__=='__main__':main()
