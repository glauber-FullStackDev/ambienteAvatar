import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GraphContract(unittest.TestCase):
    def test_audio_and_keyframes_reach_sampler(self):
        g = json.loads((ROOT / 'h3_first_last_audio_api.json').read_text())
        self.assertEqual(g['8']['inputs']['first_frame'], ['5', 0])
        self.assertEqual(g['8']['inputs']['last_frame'], ['6', 0])
        self.assertEqual(g['9']['inputs']['audio'], ['7', 0])
        self.assertEqual(g['9']['inputs']['positive'], ['8', 0])
        self.assertEqual(g['9']['inputs']['latent'], ['8', 1])
        self.assertEqual(g['10']['inputs']['conditioning'], ['9', 0])
        self.assertEqual(g['14']['inputs']['guider'], ['10', 0])
        self.assertEqual(g['14']['inputs']['latent_image'], ['8', 1])
        self.assertEqual(g['12']['inputs']['sampler_name'], 'res_multistep')
        self.assertEqual(g['13']['inputs']['scheduler'], 'simple')
        self.assertEqual(g['13']['inputs']['steps'], 20)
        self.assertEqual(g['8']['inputs']['length'], 124)
        for node in g.values():
            for value in node['inputs'].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], g)

class TurboContract(unittest.TestCase):
    def test_fl2va_lora_and_sigma_are_connected(self):
        g = json.loads((ROOT / 'h3_first_last_audio_turbo4_api.json').read_text())
        self.assertIn('fl2v_turbo_4step', g['19']['inputs']['lora_name'])
        self.assertEqual(g['20']['inputs']['model'], ['19', 0])
        self.assertEqual(g['20']['inputs']['shift_video'], 6.0)
        self.assertEqual(g['20']['inputs']['shift_audio'], 3.0)
        self.assertEqual(g['10']['inputs']['model'], ['20', 0])
        self.assertEqual(g['13']['inputs']['model'], ['20', 0])
        self.assertEqual(g['13']['inputs']['steps'], 4)
        self.assertEqual(g['12']['inputs']['sampler_name'], 'euler')


class QuantizationContract(unittest.TestCase):
    def test_both_graphs_and_ui_use_q4(self):
        for name in ('h3_first_last_audio_api.json', 'h3_first_last_audio_turbo4_api.json'):
            graph = json.loads((ROOT / name).read_text())
            self.assertEqual(graph['1']['class_type'], 'UnetLoaderGGUF')
            self.assertEqual(graph['1']['inputs']['unet_name'], 'minimax_h3_fl2va_pruned-Q4_K.gguf')
            self.assertEqual(graph['2']['class_type'], 'CLIPLoaderGGUF')
        ui = json.loads((ROOT / 'h3_first_last_audio_ui.json').read_text())
        models = {node['id']: node for node in ui['nodes']}
        self.assertEqual(models[1]['type'], 'UnetLoaderGGUF')
        self.assertEqual(models[1]['widgets_values'][0], 'minimax_h3_fl2va_pruned-Q4_K.gguf')
        self.assertEqual(models[2]['type'], 'CLIPLoaderGGUF')

    def test_download_list_excludes_int8_diffusion(self):
        import runpy
        import sys
        import types
        from unittest.mock import patch
        downloads = []
        fake = types.ModuleType('huggingface_hub')
        fake.hf_hub_download = lambda **kwargs: downloads.append(kwargs['filename'])
        with patch.dict(sys.modules, {'huggingface_hub': fake}), patch.object(sys, 'argv', ['baixar_modelos.py']):
            runpy.run_path(str(ROOT / 'baixar_modelos.py'), run_name='__main__')
        self.assertIn('minimax_h3_fl2va_pruned-Q4_K.gguf', downloads)
        self.assertFalse(any('fl2va_pruned_int8' in filename for filename in downloads))


if __name__ == '__main__': unittest.main()
