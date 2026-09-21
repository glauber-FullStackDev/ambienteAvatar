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
from build_ltx25_ia2v_iclora_api_workflow import (  # noqa: E402
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


class Ltx25IcloraWorkflowTests(unittest.TestCase):
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
            "guiding_strength": 0.8,
            "iclora_strength": 0.9,
            "cfg": 1.0,
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
            self._values(lora_strength=0.0, guiding_strength=1.0, seed=7, cfg=1.5),
            "a",
        )
        self.assertEqual(workflow["132"]["inputs"]["strength_model"], 0.0)
        self.assertEqual(workflow["111"]["inputs"]["value"], 1.0)
        self.assertEqual(workflow["109"]["inputs"]["value"], 7)
        self.assertEqual(workflow["163"]["inputs"]["cfg"], 1.5)
        self.assertEqual(workflow["176"]["inputs"]["cfg"], 1.5)
        self.assertEqual(workflow["100"]["inputs"]["image"], "jobs/a/input.png")
        self.assertEqual(workflow["101"]["inputs"]["audio"], "jobs/a/audio.wav")
        self.assertEqual(workflow["183"]["inputs"]["filename_prefix"], "video/jobs/a/LTX_2.5_ia2v_iclora")
        self.assertEqual(
            workflow["185"]["inputs"]["filename_prefix"],
            "images/last_frame/jobs/a/LTX_2.5_ia2v_iclora",
        )

    def test_first_frame_anchor_and_persistent_guide_wiring(self) -> None:
        prompt = self.template["prompt"]
        self.assertEqual(prompt["153"]["inputs"]["latent"], ["152", 0])
        self.assertEqual(prompt["154"]["inputs"]["latent"], ["153", 0])
        self.assertEqual(prompt["156"]["inputs"]["latent"], ["154", 2])
        self.assertEqual(prompt["164"]["inputs"]["video_latent"], ["156", 2])
        self.assertEqual(prompt["170"]["inputs"]["samples"], ["167", 2])
        self.assertEqual(prompt["132"]["inputs"]["model"], ["131", 0])
        self.assertEqual(prompt["131"]["inputs"]["model"], ["130", 0])
        self.assertEqual(prompt["163"]["inputs"]["model"], ["132", 0])

    def test_int_casts_feed_int_inputs(self) -> None:
        prompt = self.template["prompt"]
        self.assertEqual(prompt["193"]["class_type"], "LTXFloatToInt")
        self.assertEqual(prompt["194"]["class_type"], "LTXFloatToInt")
        self.assertEqual(prompt["195"]["class_type"], "LTXFloatToInt")
        self.assertEqual(prompt["193"]["inputs"]["a"], ["190", 0])
        self.assertEqual(prompt["152"]["inputs"]["width"], ["193", 0])
        self.assertEqual(prompt["152"]["inputs"]["height"], ["194", 0])
        self.assertEqual(prompt["152"]["inputs"]["length"], ["195", 0])
        self.assertEqual(prompt["151"]["inputs"]["amount"], ["195", 0])

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

    def test_reference_guide_bypassed_when_absent(self) -> None:
        workflow = build_job_workflow(self.template, self._values(), "a")
        self.assertNotIn("156", workflow)
        self.assertNotIn("115", workflow)
        self.assertEqual(workflow["163"]["inputs"]["positive"], ["154", 0])
        self.assertEqual(workflow["164"]["inputs"]["video_latent"], ["154", 2])
        self.assertEqual(workflow["167"]["inputs"]["positive"], ["154", 0])

    def test_reference_guide_wired_when_present(self) -> None:
        workflow = build_job_workflow(
            self.template,
            self._values(
                reference_image_filename="jobs/a/reference.png",
                reference_frame_idx=-1,
                reference_guiding_strength=0.6,
            ),
            "a",
        )
        self.assertEqual(workflow["115"]["inputs"]["image"], "jobs/a/reference.png")
        self.assertEqual(workflow["118"]["inputs"]["value"], -1)
        self.assertEqual(workflow["119"]["inputs"]["value"], 0.6)
        self.assertEqual(workflow["156"]["inputs"]["positive"], ["154", 0])
        self.assertEqual(workflow["156"]["inputs"]["latent"], ["154", 2])
        self.assertEqual(workflow["163"]["inputs"]["positive"], ["156", 0])
        self.assertEqual(workflow["164"]["inputs"]["video_latent"], ["156", 2])
        self.assertEqual(workflow["167"]["inputs"]["positive"], ["156", 0])

    def test_reference_guide_defaults_inherit_guiding_strength(self) -> None:
        workflow = build_job_workflow(
            self.template,
            self._values(reference_image_filename="jobs/a/reference.png", guiding_strength=0.5),
            "a",
        )
        self.assertEqual(workflow["119"]["inputs"]["value"], 0.5)

    def test_guiding_disabled_bypasses_persistent_guide(self) -> None:
        workflow = build_job_workflow(self.template, self._values(guiding_strength=0.0), "a")
        for node_id in ("111", "151", "154", "156", "115"):
            self.assertNotIn(node_id, workflow)
        self.assertEqual(workflow["163"]["inputs"]["positive"], ["150", 0])
        self.assertEqual(workflow["163"]["inputs"]["negative"], ["150", 1])
        self.assertEqual(workflow["164"]["inputs"]["video_latent"], ["153", 0])
        self.assertEqual(workflow["167"]["inputs"]["positive"], ["150", 0])
        self.assertEqual(workflow["170"]["inputs"]["samples"], ["167", 2])

    def test_guiding_disabled_keeps_reference_keyframe(self) -> None:
        workflow = build_job_workflow(
            self.template,
            self._values(
                guiding_strength=0.0,
                reference_image_filename="jobs/a/reference.png",
                reference_frame_idx=-1,
                reference_guiding_strength=0.7,
            ),
            "a",
        )
        for node_id in ("111", "151", "154"):
            self.assertNotIn(node_id, workflow)
        self.assertIn("156", workflow)
        self.assertEqual(workflow["156"]["inputs"]["positive"], ["150", 0])
        self.assertEqual(workflow["156"]["inputs"]["negative"], ["150", 1])
        self.assertEqual(workflow["156"]["inputs"]["latent"], ["153", 0])
        self.assertEqual(workflow["119"]["inputs"]["value"], 0.7)
        self.assertEqual(workflow["163"]["inputs"]["positive"], ["156", 0])


if __name__ == "__main__":
    unittest.main()
