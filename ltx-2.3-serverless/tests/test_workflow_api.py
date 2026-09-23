from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ltx-2.3-serverless" / "src"))
sys.path.insert(0, str(ROOT / "ltx-2.3" / "scripts"))
from workflow_api import (  # noqa: E402
    IMAGE_STRENGTH_ID,
    PERSONAL_LORA_ID,
    PROMPT_ENHANCE_ID,
    build_job_workflow,
    compile_workflow,
)
from build_ltx25_ia2v_api_workflow import (  # noqa: E402
    UNET_NAME,
    UNET_NAME_BF16,
    build_template,
    validate_template,
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
        self.assertEqual(self.template["9001"]["inputs"]["format"], "auto")
        self.assertEqual(self.template["9003"]["class_type"], "SaveImage")
        self.assertNotIn("300", self.template)
        self.assertEqual(self.template["295"]["inputs"]["vae"], ["317", 2])

    def test_job_values_are_applied_only_to_supported_controls(self) -> None:
        workflow = build_job_workflow(
            self.template,
            {
                "image_filename": "jobs/a/input.png",
                "audio_filename": "jobs/a/audio.wav",
                "prompt": "glauberavatar speaking naturally",
                "width": 704,
                "height": 1280,
                "duration_seconds": 12,
                "fps": 24,
                "audio_start_seconds": 1,
                "seed": 42,
                "lora_strength": 1.25,
                "image_strength": 0.6,
                "enable_prompt_enhance": False,
            },
            "a",
        )
        self.assertEqual(workflow[PERSONAL_LORA_ID]["inputs"]["strength_model"], 1.25)
        self.assertEqual(workflow[IMAGE_STRENGTH_ID]["inputs"]["strength"], 0.6)
        self.assertIs(workflow[PROMPT_ENHANCE_ID]["inputs"]["value"], False)
        self.assertEqual(workflow["269"]["inputs"]["image"], "jobs/a/input.png")
        self.assertIn("jobs/a", workflow["9001"]["inputs"]["filename_prefix"])


class Ltx25Ia2vWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template = build_template()
        validate_template(cls.template)

    def _values(self, **overrides):
        values = {
            "image_filename": "jobs/a/input.png",
            "audio_filename": "jobs/a/audio.wav",
            "prompt": "subtle movement only",
            "negative_prompt": "",
            "width": 704,
            "height": 1280,
            "duration_seconds": 10.0,
            "fps": 24,
            "audio_start_seconds": 0.0,
            "seed": 42,
            "lora_strength": 0.7,
            "first_frame_strength": 1.0,
            "cfg": 1.0,
            "decode_tile_size": 512,
            "enable_prompt_enhance": None,
            "base_sigmas": None,
            "refine_sigmas": None,
        }
        values.update(overrides)
        return values

    def test_template_marker_routes_to_ltx25_builder(self) -> None:
        workflow = build_job_workflow(self.template, self._values(), "a")
        self.assertIn("183", workflow)
        self.assertNotIn("9001", workflow)

    def test_job_values_are_applied(self) -> None:
        workflow = build_job_workflow(
            self.template,
            self._values(lora_strength=0.0, seed=7, cfg=1.5, decode_tile_size=1024),
            "a",
        )
        self.assertEqual(workflow["132"]["inputs"]["strength_model"], 0.0)
        self.assertEqual(workflow["109"]["inputs"]["value"], 7)
        self.assertEqual(workflow["163"]["inputs"]["cfg"], 1.5)
        self.assertEqual(workflow["176"]["inputs"]["cfg"], 1.5)
        self.assertEqual(workflow["196"]["inputs"]["value"], 1024)
        self.assertEqual(workflow["180"]["inputs"]["tile_size"], ["196", 0])
        self.assertEqual(workflow["100"]["inputs"]["image"], "jobs/a/input.png")
        self.assertEqual(workflow["101"]["inputs"]["audio"], "jobs/a/audio.wav")
        self.assertEqual(workflow["183"]["inputs"]["filename_prefix"], "video/jobs/a/LTX_2.5_ia2v")
        self.assertEqual(
            workflow["185"]["inputs"]["filename_prefix"],
            "images/last_frame/jobs/a/LTX_2.5_ia2v",
        )

    def test_official_wiring_without_iclora_or_guides(self) -> None:
        prompt = self.template["prompt"]
        self.assertEqual(prompt["132"]["inputs"]["model"], ["130", 0])
        self.assertEqual(prompt["153"]["inputs"]["latent"], ["152", 0])
        self.assertEqual(prompt["163"]["inputs"]["model"], ["132", 0])
        self.assertEqual(prompt["163"]["inputs"]["positive"], ["150", 0])
        self.assertEqual(prompt["163"]["inputs"]["negative"], ["150", 1])
        self.assertEqual(prompt["164"]["inputs"]["video_latent"], ["153", 0])
        self.assertEqual(prompt["170"]["inputs"]["samples"], ["166", 0])
        self.assertEqual(prompt["176"]["inputs"]["positive"], ["150", 0])
        self.assertEqual(prompt["176"]["inputs"]["model"], ["132", 0])
        for gone in ("111", "115", "116", "117", "118", "119", "131", "151", "154", "156", "167"):
            self.assertNotIn(gone, prompt)

    def test_int_casts_feed_int_inputs(self) -> None:
        prompt = self.template["prompt"]
        self.assertEqual(prompt["193"]["class_type"], "LTXFloatToInt")
        self.assertEqual(prompt["194"]["class_type"], "LTXFloatToInt")
        self.assertEqual(prompt["195"]["class_type"], "LTXFloatToInt")
        self.assertEqual(prompt["193"]["inputs"]["a"], ["190", 0])
        self.assertEqual(prompt["152"]["inputs"]["width"], ["193", 0])
        self.assertEqual(prompt["152"]["inputs"]["height"], ["194", 0])
        self.assertEqual(prompt["152"]["inputs"]["length"], ["195", 0])

    def test_prompt_falls_back_to_template_preset(self) -> None:
        workflow = build_job_workflow(self.template, self._values(prompt=""), "a")
        self.assertIn(
            "composition and framing of the initial frame",
            workflow["102"]["inputs"]["value"],
        )
        self.assertIn("precise lip sync", workflow["102"]["inputs"]["value"])
        self.assertEqual(
            workflow["102"]["inputs"]["value"],
            self.template["prompt"]["102"]["inputs"]["value"],
        )

    def test_prompt_enhance_toggle(self) -> None:
        workflow = build_job_workflow(self.template, self._values(enable_prompt_enhance=True), "a")
        self.assertIs(workflow["144"]["inputs"]["value"], True)
        workflow = build_job_workflow(self.template, self._values(enable_prompt_enhance=False), "a")
        self.assertIs(workflow["144"]["inputs"]["value"], False)

    def test_optional_sigmas_overrides(self) -> None:
        workflow = build_job_workflow(
            self.template,
            self._values(base_sigmas="1.0, 0.5, 0.0", refine_sigmas="0.4, 0.0"),
            "a",
        )
        self.assertEqual(workflow["160"]["inputs"]["sigmas"], "1.0, 0.5, 0.0")
        self.assertEqual(workflow["173"]["inputs"]["sigmas"], "0.4, 0.0")

    def test_default_template_uses_int8_unet(self) -> None:
        self.assertEqual(
            self.template["prompt"]["130"]["inputs"]["unet_name"], UNET_NAME
        )

    def test_bf16_template_points_unet_loader_to_bf16_checkpoint(self) -> None:
        template = build_template(UNET_NAME_BF16)
        validate_template(template, unet_name=UNET_NAME_BF16)
        self.assertEqual(template["prompt"]["130"]["inputs"]["unet_name"], UNET_NAME_BF16)
        # Everything else stays identical to the int8 template.
        self.assertEqual(template["prompt"]["134"]["inputs"]["clip_name"], "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors")
        workflow = build_job_workflow(template, self._values(), "a")
        self.assertEqual(workflow["130"]["inputs"]["unet_name"], UNET_NAME_BF16)
        self.assertEqual(workflow["183"]["inputs"]["filename_prefix"], "video/jobs/a/LTX_2.5_ia2v")


if __name__ == "__main__":
    unittest.main()
