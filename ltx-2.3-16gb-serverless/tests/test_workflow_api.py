from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ltx-2.3-16gb-serverless" / "src"))
sys.path.insert(0, str(ROOT / "ltx-2.3-16gb" / "scripts"))

from workflow_api import (  # noqa: E402
    DECODE_ID,
    IMAGE_STRENGTH_ID,
    PERSONAL_LORA_ID,
    PROMPT_ENHANCE_ID,
    build_job_workflow,
    compile_workflow,
)
from handler import DEFAULT_PROMPT  # noqa: E402
from personal_workflows import add_last_frame_export, validate_personal_lora  # noqa: E402
from gguf_convert import convert_to_gguf  # noqa: E402

SOURCE_PATH = ROOT / "ltx-2.3-16gb" / "assets" / "video_ltx2_3_ia2v_personal_source.json"


class WorkflowApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Mesmos passos do Dockerfile: pessoal -> GGUF 16 GB -> formato API.
        workflow = add_last_frame_export(json.loads(SOURCE_PATH.read_text(encoding="utf-8")))
        validate_personal_lora(workflow)
        cls.template = compile_workflow(convert_to_gguf(workflow))

    def test_compiles_gguf_api_nodes_and_output_nodes(self) -> None:
        self.assertEqual(self.template["317"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(
            self.template["317"]["inputs"]["unet_name"],
            "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
        )
        self.assertEqual(self.template["318"]["class_type"], "DualCLIPLoaderGGUF")
        self.assertEqual(self.template["335"]["class_type"], "VAELoaderKJ")
        self.assertEqual(self.template["269"]["class_type"], "LoadImage")
        self.assertEqual(self.template["276"]["class_type"], "LoadAudio")
        self.assertEqual(self.template["9001"]["class_type"], "SaveVideo")
        self.assertEqual(self.template["9001"]["inputs"]["format"], "auto")
        self.assertEqual(self.template["9003"]["class_type"], "SaveImage")
        self.assertNotIn("300", self.template)  # Reroute removido

    def test_decode_tiled_defaults_match_the_validated_16gb_profile(self) -> None:
        decode = self.template[DECODE_ID]["inputs"]
        self.assertEqual(decode["tile_size"], 512)
        self.assertEqual(decode["overlap"], 64)
        self.assertEqual(decode["temporal_size"], 48)
        self.assertEqual(decode["temporal_overlap"], 16)

    def test_job_values_are_applied(self) -> None:
        workflow = build_job_workflow(self.template, self._values(), "a")
        self.assertEqual(workflow[PERSONAL_LORA_ID]["inputs"]["strength_model"], 0.7)
        self.assertEqual(workflow[IMAGE_STRENGTH_ID]["inputs"]["strength"], 1.0)
        self.assertEqual(workflow["269"]["inputs"]["image"], "jobs/a/input.png")
        self.assertEqual(workflow["276"]["inputs"]["audio"], "jobs/a/audio.wav")
        self.assertEqual(workflow["319"]["inputs"]["value"], "glauberavatar speaking naturally")
        self.assertEqual(workflow["330"]["inputs"]["value"], 704)
        self.assertEqual(workflow["324"]["inputs"]["value"], 1280)
        self.assertEqual(workflow["286"]["inputs"]["noise_seed"], 42)
        self.assertIn("jobs/a", workflow["9001"]["inputs"]["filename_prefix"])
        self.assertIn("jobs/a", workflow["9003"]["inputs"]["filename_prefix"])

    def test_decode_values_are_wired_from_the_job(self) -> None:
        workflow = build_job_workflow(
            self.template,
            self._values(decode_tile_size=384, temporal_size=32, temporal_overlap=8),
            "a",
        )
        decode = workflow[DECODE_ID]["inputs"]
        self.assertEqual(decode["tile_size"], 384)
        self.assertEqual(decode["overlap"], 64)  # fixo, nunca vem do job
        self.assertEqual(decode["temporal_size"], 32)
        self.assertEqual(decode["temporal_overlap"], 8)

    def test_prompt_enhance_is_off_by_default_and_togglable(self) -> None:
        workflow = build_job_workflow(self.template, self._values(), "a")
        self.assertIs(workflow[PROMPT_ENHANCE_ID]["inputs"]["value"], False)
        workflow = build_job_workflow(self.template, self._values(enable_prompt_enhance=True), "a")
        self.assertIs(workflow[PROMPT_ENHANCE_ID]["inputs"]["value"], True)

    def test_default_prompt_contract(self) -> None:
        workflow = build_job_workflow(self.template, self._values(prompt=DEFAULT_PROMPT), "a")
        self.assertEqual(workflow["319"]["inputs"]["value"], DEFAULT_PROMPT)
        self.assertIn("glauberavatar remains in the exact composition", DEFAULT_PROMPT)
        self.assertIn("never touches, holds or bites anything", DEFAULT_PROMPT)

    def _values(self, **overrides):
        values = {
            "image_filename": "jobs/a/input.png",
            "audio_filename": "jobs/a/audio.wav",
            "prompt": "glauberavatar speaking naturally",
            "width": 704,
            "height": 1280,
            "duration_seconds": 12,
            "fps": 24,
            "audio_start_seconds": 1,
            "seed": 42,
            "lora_strength": 0.7,
            "image_strength": 1.0,
            "decode_tile_size": 512,
            "temporal_size": 48,
            "temporal_overlap": 16,
            "enable_prompt_enhance": False,
        }
        values.update(overrides)
        return values


if __name__ == "__main__":
    unittest.main()
