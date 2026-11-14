"""Reuse a verified-layout, sharded FLUX 3 text encoder from the local HF cache."""
import json
import logging
import os
import struct
import tempfile
from pathlib import Path

LOG = logging.getLogger('flux.encoder-cache')
SOURCE_REVISION = '09561231f44e41381c461caf53d5452005a60ac9'
REQUIRED_SHAPES = {
    'model.visual.deepstack_merger_list.0.norm.weight': [4096],
    'model.visual.merger.linear_fc2.weight': [2560, 4096],
    'model.language_model.embed_tokens.weight': [151936, 2560],
    'model.language_model.layers.0.self_attn.q_proj.weight': [4096, 2560],
}


def _read_header(path):
    size = path.stat().st_size
    with path.open('rb') as source:
        raw = source.read(8)
        if len(raw) != 8:
            raise ValueError(f'Cabeçalho incompleto: {path.name}')
        length = struct.unpack('<Q', raw)[0]
        if not 1 <= length <= 100_000_000 or 8 + length > size:
            raise ValueError(f'Tamanho de cabeçalho inválido: {path.name}')
        header = json.loads(source.read(length))
    if not isinstance(header, dict):
        raise ValueError('Cabeçalho safetensors inválido.')
    ranges = []
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        if not isinstance(tensor, dict) or not isinstance(tensor.get('dtype'), str) or not isinstance(tensor.get('shape'), list):
            raise ValueError(f'Tensor inválido: {name}')
        offsets = tensor.get('data_offsets')
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(v) is int for v in offsets):
            raise ValueError(f'Offsets inválidos: {name}')
        start, end = offsets
        if start < 0 or end < start or 8 + length + end > size:
            raise ValueError(f'Tensor incompleto: {name}')
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError(f'Dados safetensors não contíguos: {path.name}')
        cursor = end
    if 8 + length + cursor != size:
        raise ValueError(f'Tamanho de dados incorreto: {path.name}')
    return header, 8 + length


def merge_shards(shards, index_path, destination, required_shapes=REQUIRED_SHAPES):
    """Merge tensors by streaming; never replace an existing destination."""
    index = json.loads(index_path.read_text(encoding='utf-8'))
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError('Índice do encoder ausente ou inválido.')
    shard_by_name = {p.name: p for p in shards}
    if set(weight_map.values()) != set(shard_by_name):
        raise ValueError('Faltam partes do encoder no cache.')
    tensors = {}
    offsets = {}
    for name, path in shard_by_name.items():
        header, data_start = _read_header(path)
        offsets[name] = data_start
        for key, value in header.items():
            if key == '__metadata__':
                continue
            if key in tensors or weight_map.get(key) != name:
                raise ValueError(f'Índice inconsistente para tensor {key}.')
            tensors[key] = (name, value)
    if set(tensors) != set(weight_map):
        raise ValueError('Tensores do encoder não correspondem ao índice.')
    for key, shape in required_shapes.items():
        if key not in tensors or tensors[key][1]['shape'] != shape:
            raise ValueError(f'Encoder incompatível: {key}.')
    out_header = {'__metadata__': {'format': 'pt'}}
    cursor = 0
    for key, (_, desc) in tensors.items():
        length = desc['data_offsets'][1] - desc['data_offsets'][0]
        out_header[key] = {'dtype': desc['dtype'], 'shape': desc['shape'], 'data_offsets': [cursor, cursor + length]}
        cursor += length
    header_bytes = json.dumps(out_header, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    header_bytes += b' ' * (-len(header_bytes) % 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.encoder-merge-', dir=destination.parent, delete=False) as out:
            temporary = Path(out.name)
            out.write(struct.pack('<Q', len(header_bytes)))
            out.write(header_bytes)
            for key, (shard_name, desc) in tensors.items():
                start, end = desc['data_offsets']
                with shard_by_name[shard_name].open('rb') as source:
                    source.seek(offsets[shard_name] + start)
                    remaining = end - start
                    while remaining:
                        chunk = source.read(min(4 * 1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError(f'Tensor truncado durante cópia: {key}.')
                        out.write(chunk)
                        remaining -= len(chunk)
            out.flush()
            os.fsync(out.fileno())
        _read_header(temporary)
        # Link is atomic and refuses to replace a file created by another process.
        os.link(temporary, destination)
        return True
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def reuse_cached_encoder(hf_home, destination):
    cache = Path(hf_home) / 'hub' / 'models--black-forest-labs--flux-3-image-klein' / 'snapshots' / SOURCE_REVISION / 'text_encoder'
    marker = destination.with_name(destination.name + '.reused.json')
    if destination.is_file() and marker.is_file():
        try:
            data = json.loads(marker.read_text(encoding='utf-8'))
            header, _ = _read_header(destination)
            if data.get('source_revision') == SOURCE_REVISION and all(header.get(k, {}).get('shape') == v for k, v in REQUIRED_SHAPES.items()):
                LOG.info('Usando encoder BF16 reconstruído do cache local.')
                return True
        except (OSError, ValueError, TypeError):
            LOG.warning('Encoder reconstruído não passou na validação; será verificado no Hugging Face.')
        return False
    if destination.exists():
        return False
    index = cache / 'model.safetensors.index.json'
    shards = [cache / 'model-00001-of-00002.safetensors', cache / 'model-00002-of-00002.safetensors']
    if not index.is_file() or not all(p.is_file() for p in shards):
        return False
    try:
        LOG.info('Reaproveitando encoder BF16 do cache; unindo arquivos em disco (pode demorar).')
        merge_shards(shards, index, destination)
        marker.write_text(json.dumps({'source_revision': SOURCE_REVISION}), encoding='utf-8')
        LOG.info('Encoder BF16 reconstruído a partir do cache local.')
        return True
    except FileExistsError:
        LOG.info('Encoder já apareceu durante a reconstrução; usando download normal.')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOG.warning('Não foi possível reaproveitar encoder do cache: %s. Usando download normal.', exc)
    return False
