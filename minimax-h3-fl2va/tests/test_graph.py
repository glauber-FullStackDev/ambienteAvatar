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


if __name__ == '__main__': unittest.main()

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
