import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from reuse_encoder import _read_header, merge_shards, reuse_cached_encoder


def write_shard(path, entries):
    header, payload = {}, b''
    for key, value in entries.items():
        header[key] = {'dtype': 'U8', 'shape': [len(value)], 'data_offsets': [len(payload), len(payload) + len(value)]}
        payload += value
    content = json.dumps(header, separators=(',', ':')).encode()
    content += b' ' * (-len(content) % 8)
    path.write_bytes(struct.pack('<Q', len(content)) + content + payload)


class ReuseTests(unittest.TestCase):
    def test_merge_preserves_tensors_without_overwriting_sources(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first, second, index, out = (root / n for n in ('one.safetensors', 'two.safetensors', 'index.json', 'out.safetensors'))
            write_shard(first, {'alpha': b'abc', 'beta': b'xy'})
            write_shard(second, {'gamma': b'1234'})
            index.write_text(json.dumps({'weight_map': {'alpha': first.name, 'beta': first.name, 'gamma': second.name}}))
            before = [p.read_bytes() for p in (first, second)]
            self.assertTrue(merge_shards([first, second], index, out, required_shapes={'gamma': [4]}))
            header, start = _read_header(out)
            self.assertEqual(out.read_bytes()[start:], b'abcxy1234')
            self.assertEqual(header['gamma']['data_offsets'], [5, 9])
            self.assertEqual([p.read_bytes() for p in (first, second)], before)
            with self.assertRaises(FileExistsError):
                merge_shards([first, second], index, out, required_shapes={'gamma': [4]})
            self.assertEqual(list(root.glob('.encoder-merge-*')), [])

    def test_rejects_incomplete_or_mismatched_cache(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first, second, index, out = (root / n for n in ('one', 'two', 'index', 'out'))
            write_shard(first, {'alpha': b'abc'})
            write_shard(second, {'gamma': b'1234'})
            index.write_text(json.dumps({'weight_map': {'alpha': first.name, 'gamma': second.name}}))
            with self.assertRaises(ValueError):
                merge_shards([first, second], index, out, required_shapes={'gamma': [999]})
            self.assertFalse(out.exists())
            second.write_bytes(second.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                merge_shards([first, second], index, out, required_shapes={})
            self.assertFalse(out.exists())
            self.assertFalse(reuse_cached_encoder(root, out))

if __name__ == '__main__':
    unittest.main()
