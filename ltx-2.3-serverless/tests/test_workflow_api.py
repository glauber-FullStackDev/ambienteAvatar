from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ltx-2.3-serverless" / "src"))
from workflow_api import (  # noqa: E402
    IMAGE_STRENGTH_ID,
    PERSONAL_LORA_ID,
    build_job_workflow,
    compile_workflow,
)


class WorkflowApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = json.loads(
            (ROOT / "ltx-2.3" / "assets" / "video_ltx2_3_ia2v_personal_source.json").read_text()
        )
        cls.template = compile_workflow(source)

    def test_compiles_api_nodes_and_output_nodes(self) -> None:
        self.assertEqual(self.template["269"]["class_type"], "LoadImage")
        self.assertEqual(self.template["276"]["class_type"], "LoadAudio")
        self.assertEqual(self.template["9001"]["class_type"], "SaveVideo")
        self.assertEqual(self.template["9003"]["class_type"], "SaveImage")

    def test_job_values_are_applied_only_to_supported_controls(self) -> None:
        workflow = build_job_workflow(
            self.template,
            {
                "image_filename": "jobs/a/input.png",
                "audio_filename": "jobs/a/audio.wav",
                "prompt": "glauberavatar speaking naturally",
                "width": 720,
                "height": 1280,
                "duration_seconds": 12,
                "fps": 24,
                "audio_start_seconds": 1,
                "seed": 42,
                "lora_strength": 1.25,
                "image_strength": 0.6,
            },
            "a",
        )
        self.assertEqual(workflow[PERSONAL_LORA_ID]["inputs"]["strength_model"], 1.25)
        self.assertEqual(workflow[IMAGE_STRENGTH_ID]["inputs"]["strength"], 0.6)
        self.assertEqual(workflow["269"]["inputs"]["image"], "jobs/a/input.png")
        self.assertIn("jobs/a", workflow["9001"]["inputs"]["filename_prefix"])


if __name__ == "__main__":
    unittest.main()
