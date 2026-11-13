"""Fixed-shape SDXL export with an exact, live-tensor U-Net graph cut."""
from pathlib import Path
import json
import os
import shutil
import tempfile
from .contracts import canonical, file_hash, require, validate_manifest, verify_bundle


def schema(model):
    def port(value):
        return {'name': value.get_any_name(), 'shape': list(value.shape), 'dtype': value.get_element_type().get_type_name()}
    return {'inputs': [port(p) for p in model.inputs], 'outputs': [port(p) for p in model.outputs]}


def partition(denoiser):
    import numpy as np
    import openvino as ov
    nodes = denoiser.get_ordered_ops()
    position = {node.get_instance_id(): i for i, node in enumerate(nodes)}
    def nbytes(port):
        return int(np.prod(port.shape, dtype=np.int64)) * port.get_element_type().size
    constants = np.array([sum(nbytes(p) for p in node.outputs()) if node.get_type_name() == 'Constant' else 0 for node in nodes], dtype=np.int64)
    cumulative = np.cumsum(constants)
    total = int(cumulative[-1])
    require(total > 0, 'denoiser contains no weights')
    best = None
    for cut in {int(np.searchsorted(cumulative, total*f)) for f in np.linspace(.45, .55, 21)}:
        frontier = []
        for index, node in enumerate(nodes[:cut+1]):
            if node.get_type_name() in ('Constant', 'Parameter'):
                continue
            for port in node.outputs():
                if any(position[target.get_node().get_instance_id()] > cut for target in port.get_target_inputs()):
                    frontier.append((index, port.get_index(), port))
        score = (sum(nbytes(port) for _, _, port in frontier), abs(int(cumulative[cut])-total//2))
        if frontier and (best is None or score < best[0]):
            best = (score, frontier)
    require(best is not None, 'no valid graph cut')
    frontier = best[1]
    results = []
    for index, (_, _, port) in enumerate(frontier):
        result = ov.opset13.result(port)
        result.output(0).get_tensor().set_names({f'boundary_{index}'})
        results.append(result)
    first = ov.Model(results, denoiser.get_parameters(), 'unet_first')
    second_model = denoiser.clone()
    second_nodes = second_model.get_ordered_ops()
    new_parameters = []
    for index, (node_index, port_index, _) in enumerate(frontier):
        node = second_nodes[node_index]
        require(node.get_friendly_name() == nodes[node_index].get_friendly_name(), 'clone graph changed ordering')
        port = node.output(port_index)
        parameter = ov.opset13.parameter(port.shape, port.get_element_type(), name=f'boundary_{index}')
        port.replace(parameter.output(0))
        parameter.output(0).get_tensor().set_names({f'boundary_{index}'})
        new_parameters.append(parameter)
    second = ov.Model(second_model.get_results(), second_model.get_parameters()+new_parameters, 'unet_second')
    # Remove unused original parameters. The coordinator forwards only declared
    # inputs; it never guesses which skip connections are needed by a partition.
    for model in (first, second):
        for parameter in list(model.get_parameters()):
            if not parameter.output(0).get_target_inputs():
                model.remove_parameter(parameter)
        model.validate_nodes_and_infer_types()
    return first, second


def export_sdxl(source: Path, output: Path, revision: str, size: int):
    import openvino as ov
    require(type(size) is int and 64 <= size <= 1024 and size % 8 == 0, 'invalid size')
    require(not output.exists(), 'output already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='.cascadia-export-', dir=output.parent))
    try:
        manifest = {'version': 1, 'architecture': 'sdxl', 'source_revision': revision, 'width': size, 'height': size, 'components': {}, 'files': {}}
        for name in ('model_index.json', 'scheduler', 'tokenizer', 'tokenizer_2'):
            src = source/name
            if src.is_dir():
                shutil.copytree(src, root/name, ignore=shutil.ignore_patterns('*.xml', '*.bin', '*.safetensors', '*.onnx'))
            else:
                shutil.copy2(src, root/name)
        for name in ('README.md', 'LICENSE', 'LICENSE.md', 'LICENSE.txt', 'cascadia-model-source.json'):
            if (source/name).is_file():
                shutil.copy2(source/name, root/name)
        core = ov.Core()
        def save(name, model):
            folder = root/name
            folder.mkdir()
            ov.save_model(model, folder/'openvino_model.xml', compress_to_fp16=False)
            # Read the serialized graph: serialization can alter port aliases.
            saved = core.read_model(folder/'openvino_model.xml')
            manifest['components'][name] = schema(saved) | {'xml': f'{name}/openvino_model.xml', 'weights': f'{name}/openvino_model.bin'}
            print(json.dumps({'exported': name, 'weight_bytes': (folder/'openvino_model.bin').stat().st_size}), flush=True)
        denoiser = None
        for name in ('text_encoder', 'text_encoder_2', 'unet', 'vae_decoder'):
            model = core.read_model(source/name/'openvino_model.xml')
            if name.startswith('text_encoder'):
                shapes = {port: [1, 77] for port in model.inputs}
            elif name == 'vae_decoder':
                shapes = {model.input('latent_sample'): [1, 4, size//8, size//8]}
            else:
                shapes = {'sample': [2,4,size//8,size//8], 'timestep': [1], 'encoder_hidden_states': [2,77,2048], 'text_embeds': [2,1280], 'time_ids': [2,6]}
            model.reshape(shapes)
            if name.startswith('text_encoder'):
                # SDXL uses the penultimate hidden state and pooled output. Keep
                # the last two states so Optimum's [-2] indexing stays correct.
                hidden = [p for p in model.outputs if any(n.startswith('hidden_states') for n in p.names)]
                require(len(hidden) >= 2, 'SDXL encoder lacks hidden states')
                selected = [model.outputs[0]]
                if len(model.outputs)>1 and 'pooler_output' in model.outputs[1].get_any_name():
                    selected.append(model.outputs[1])
                for port in hidden[-2:]:
                    if port not in selected:
                        selected.append(port)
                model = ov.Model(selected, model.get_parameters(), name)
            save(name, model)
            shutil.copy2(source/name/'config.json', root/name/'config.json')
            if name == 'unet':
                denoiser = model
        first, second = partition(denoiser)
        save('unet_first', first)
        save('unet_second', second)
        for path in sorted(root.rglob('*')):
            if path.is_file():
                manifest['files'][path.relative_to(root).as_posix()] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
        mid = validate_manifest(manifest)
        (root/'manifest.json').write_bytes(canonical(manifest)+b'\n')
        verify_bundle(root, mid, list(manifest['components']))
        os.rename(root, output)
        print(json.dumps({'manifest': mid, 'output': str(output)}), flush=True)
        return mid
    finally:
        if root.exists():
            shutil.rmtree(root)


def pack(source: Path, output: Path, expected: str, selected):
    manifest = verify_bundle(source, expected, selected)
    require(not output.exists(), 'output already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='.cascadia-pack-', dir=output.parent))
    try:
        needed = {name for name in manifest['files'] if not name.endswith(('.xml', '.bin'))}
        for name in selected:
            needed.update((manifest['components'][name]['xml'], manifest['components'][name]['weights']))
        for name in needed | {'manifest.json'}:
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source/name, target)
        verify_bundle(root, expected, selected)
        os.rename(root, output)
    finally:
        if root.exists():
            shutil.rmtree(root)
