"""Version-one contracts shared with the cascadia-diffusion Rust crate.

Tensor payloads are little endian; frame header length is big endian. No model
code, pickle, or objects are accepted from the data plane.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import uuid

MAX_TENSOR_BYTES = 256 * 1024 * 1024
MAX_HEADER = 64 * 1024
MAX_FRAME = 8 + MAX_HEADER + MAX_TENSOR_BYTES
WIDTH = {'f32': 4, 'f16': 2, 'i64': 8, 'i32': 4, 'u8': 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_sdxl_metadata(root: Path):
    """Reject pipeline settings this fixed implementation would otherwise ignore."""
    def config(name):
        path = root / name
        require(path.stat().st_size <= 1024 * 1024, 'pipeline config exceeds limit')
        value = parse_json(path.read_bytes())
        require(isinstance(value, dict), 'pipeline config must be an object')
        return value

    index = config('model_index.json')
    scheduler = config('scheduler/scheduler_config.json')
    require(index.get('_class_name') in ('StableDiffusionXLPipeline', 'OVStableDiffusionXLPipeline'),
            'only the SDXL text-to-image pipeline is supported')
    require(index.get('scheduler') == ['diffusers', 'EulerDiscreteScheduler']
            and scheduler.get('_class_name') == 'EulerDiscreteScheduler',
            'only EulerDiscreteScheduler is supported; scheduler substitution is forbidden')
    require(index.get('force_zeros_for_empty_prompt', True) is True
            and index.get('requires_aesthetics_score', False) is False,
            'unsupported SDXL conditioning configuration')
    for name in ('tokenizer', 'tokenizer_2'):
        require(index.get(name) == ['transformers', 'CLIPTokenizer']
                and config(f'{name}/tokenizer_config.json').get('model_max_length') == 77,
                'only the 77-token CLIP tokenizer profile is supported')


def exact(value, keys):
    require(isinstance(value, dict) and set(value) == set(keys), 'invalid object fields')


def integer(value, lower, upper):
    require(type(value) is int and lower <= value <= upper, 'integer out of range')


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_./:-]{1,128}', value), 'invalid identifier')


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r'[a-f0-9]{64}', value), 'invalid SHA-256')


def relative_path(value):
    require(isinstance(value, str) and 0 < len(value) <= 240, 'invalid path')
    for part in value.split('/'):
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', part) and part not in ('.', '..') and not part.endswith('.'), 'unsafe artifact path')
        stem = part.split('.')[0].upper()
        require(stem not in ('CON', 'PRN', 'AUX', 'NUL') and not re.fullmatch(r'(COM|LPT)[0-9]', stem), 'reserved artifact path')


def port_bytes(port):
    exact(port, ('name', 'dtype', 'shape'))
    identifier(port['name'])
    require(isinstance(port['dtype'], str) and port['dtype'] in WIDTH, 'unknown dtype')
    require(isinstance(port['shape'], list) and len(port['shape']) <= 8, 'invalid rank')
    size = WIDTH[port['dtype']]
    for dim in port['shape']:
        integer(dim, 1, MAX_TENSOR_BYTES)
        size *= dim
        require(size <= MAX_TENSOR_BYTES, 'tensor exceeds budget')
    return size


def ports_bytes(ports):
    require(isinstance(ports, list) and 0 < len(ports) <= 128, 'invalid tensor count')
    total = sum(port_bytes(p) for p in ports)
    require(len({p['name'] for p in ports}) == len(ports), 'duplicate tensor name')
    require(total <= MAX_TENSOR_BYTES, 'combined tensors exceed budget')
    return total


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def parse_json(raw):
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def validate_manifest(m):
    exact(m, ('version', 'architecture', 'source_revision', 'width', 'height', 'components', 'files'))
    integer(m['version'], 1, 1)
    require(m['architecture'] == 'sdxl', 'unsupported architecture')
    require(isinstance(m['source_revision'], str) and re.fullmatch(r'[a-f0-9]{40}', m['source_revision']), 'immutable source revision required')
    for dimension in (m['width'], m['height']):
        integer(dimension, 64, 1024)
        require(dimension % 8 == 0, 'dimensions must be multiples of eight')
    require(isinstance(m['components'], dict) and 0 < len(m['components']) <= 16, 'invalid components')
    require(isinstance(m['files'], dict) and 0 < len(m['files']) <= 256, 'invalid artifacts')
    for path, artifact in m['files'].items():
        relative_path(path)
        exact(artifact, ('sha256', 'bytes'))
        digest(artifact['sha256'])
        integer(artifact['bytes'], 1, 2**64 - 1)
    for name, component in m['components'].items():
        identifier(name)
        exact(component, ('inputs', 'outputs', 'xml', 'weights'))
        ports_bytes(component['inputs'])
        ports_bytes(component['outputs'])
        for path in (component['xml'], component['weights']):
            relative_path(path)
            require(path in m['files'], 'unhashed component artifact')
        require(component['xml'].endswith('.xml') and component['weights'].endswith('.bin') and component['xml'][:-4] == component['weights'][:-4], 'invalid IR pair')
    return hashlib.sha256(canonical(m)).hexdigest()


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_bundle(root: Path, expected: str, selected):
    digest(expected)
    require(not root.is_symlink(), 'bundle root is a symlink')
    path = root / 'manifest.json'
    require(not path.is_symlink() and path.stat().st_size <= 1024 * 1024, 'invalid manifest file')
    manifest = parse_json(path.read_bytes())
    require(validate_manifest(manifest) == expected, 'manifest digest mismatch')
    require(len(set(selected)) == len(selected) and set(selected) <= manifest['components'].keys(), 'invalid component selection')
    needed = {path for path in manifest['files'] if not path.endswith(('.xml', '.bin'))}
    for name in selected:
        component = manifest['components'][name]
        needed.update((component['xml'], component['weights']))
    for name in needed:
        path = root
        for part in name.split('/'):
            path /= part
            require(not path.is_symlink(), 'artifact symlink')
        artifact = manifest['files'][name]
        require(path.is_file() and path.stat().st_size == artifact['bytes'], 'artifact length mismatch')
        require(file_hash(path) == artifact['sha256'], 'artifact hash mismatch')
    # Tokenizer libraries auto-discover optional files. Unpinned files must not
    # influence execution. Runtime caches and config belong outside this root.
    for count, path in enumerate(root.rglob('*'), 1):
        require(count <= 1024, 'bundle has too many entries')
        require(not path.is_symlink(), 'bundle symlink')
        if path.is_file():
            require(path.relative_to(root).as_posix() in manifest['files'] or path == root / 'manifest.json', 'unlisted artifact')
    return manifest


def validate_header(header):
    exact(header, ('manifest', 'component', 'lease', 'sequence', 'tensors'))
    digest(header['manifest'])
    identifier(header['component'])
    exact(header['lease'], ('boot', 'id'))
    for value in header['lease'].values():
        require(isinstance(value, str) and str(uuid.UUID(value)) == value, 'noncanonical lease UUID')
    integer(header['sequence'], 1, 2**32 - 1)
    return ports_bytes(header['tensors'])


def finite(port, raw):
    fmt = {'f32': '<f', 'f16': '<e'}.get(port['dtype'])
    if fmt:
        # Vectorization is optional; the portable validator has no NumPy dependency.
        try:
            import numpy as np
        except ImportError:
            require(all(math.isfinite(v[0]) for v in struct.iter_unpack(fmt, raw)), 'nonfinite tensor')
        else:
            require(bool(np.isfinite(np.frombuffer(raw, dtype={'f32': '<f4', 'f16': '<f2'}[port['dtype']])).all()), 'nonfinite tensor')


def encode(header, values):
    total = validate_header(header)
    require(len(values) == len(header['tensors']), 'tensor count mismatch')
    raw = canonical(header)
    require(len(raw) <= MAX_HEADER, 'header exceeds budget')
    result = bytearray(8 + len(raw) + total)
    result[:8] = b'CDF1' + struct.pack('>I', len(raw))
    result[8:8+len(raw)] = raw
    offset = 8 + len(raw)
    for port, value in zip(header['tensors'], values):
        size = port_bytes(port)
        require(len(value) == size, 'tensor byte mismatch')
        finite(port, value)
        result[offset:offset+size] = value
        offset += size
    return result


def decode(raw, expected=None):
    require(8 <= len(raw) <= MAX_FRAME and raw[:4] == b'CDF1', 'invalid frame prefix or size')
    size = struct.unpack('>I', raw[4:8])[0]
    require(0 < size <= MAX_HEADER and len(raw) >= 8+size, 'invalid header size')
    header = parse_json(bytes(raw[8:8+size]))
    total = validate_header(header)
    require(len(raw) == 8+size+total, 'payload length mismatch')
    if expected is not None:
        require(header['tensors'] == expected, 'tensor schema mismatch')
    values = []
    offset = 8+size
    view = memoryview(raw)
    for port in header['tensors']:
        size = port_bytes(port)
        value = view[offset:offset+size]
        finite(port, value)
        values.append(value)
        offset += size
    return header, values
