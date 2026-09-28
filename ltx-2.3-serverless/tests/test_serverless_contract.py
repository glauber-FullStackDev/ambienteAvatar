from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SERVERLESS = ROOT / "ltx-2.3-serverless"
BF16_SERVERLESS = ROOT / "ltx-2.3-serverless-bf16"


class ServerlessBuildContractTests(unittest.TestCase):
    def test_image_installs_every_custom_node_pack_used_by_the_workflow(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ComfyUI-BFSNodes", dockerfile)
        self.assertIn("ComfyUI-LTXVideo", dockerfile)
        self.assertIn("patch_ltxvideo_kornia.py", dockerfile)

    def test_image_targets_ltx23_pipeline(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("DOWNLOAD_LTX25_MODELS_ON_START=0", dockerfile)
        self.assertIn(
            "WORKFLOW_PATH=/opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_api.json",
            dockerfile,
        )
        self.assertIn("LEGACY_WORKFLOW_PATH=/opt/defaults/workflows/video_ltx2_5_ia2v_api.json", dockerfile)
        self.assertIn("build_ltx25_ia2v_api_workflow.py", dockerfile)
        self.assertIn("video_ltx2_3_ia2v_personal_lora_api.json", dockerfile)

    def test_image_bakes_only_the_int8_unet_template(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("video_ltx2_5_ia2v_api.json", dockerfile)
        self.assertNotIn("video_ltx2_5_ia2v_bf16_api.json", dockerfile)
        self.assertNotIn(
            "ltx-2.5-22b-distilled-transformer-bf16.safetensors", dockerfile
        )

    def test_image_does_not_reference_the_bf16_tree(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("ltx-2.3-serverless-bf16", dockerfile)

    def test_template_uses_official_refine_schedule(self) -> None:
        builder = (ROOT / "ltx-2.3" / "scripts" / "build_ltx25_ia2v_api_workflow.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"', builder)

    def test_bootstrap_downloads_only_the_ltx23_set(self) -> None:
        bootstrap = (SERVERLESS / "src" / "bootstrap_models.py").read_text(encoding="utf-8")
        self.assertIn("ltx-2.3-22b-dev-fp8.safetensors", bootstrap)
        self.assertIn("gemma_3_12B_it_fp4_mixed.safetensors", bootstrap)
        self.assertIn(
            "loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
            bootstrap,
        )
        self.assertIn("loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors", bootstrap)
        self.assertIn("latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors", bootstrap)
        self.assertNotIn(
            "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", bootstrap
        )
        self.assertNotIn(
            "ltx-2.5-22b-distilled-transformer-bf16.safetensors", bootstrap
        )
        self.assertNotIn("ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors", bootstrap)

    def test_handler_enforces_minimum_audio_duration(self) -> None:
        handler = (SERVERLESS / "src" / "handler.py").read_text(encoding="utf-8")
        self.assertIn("MIN_DURATION_SECONDS = 5.0", handler)
        self.assertIn("minimum=MIN_DURATION_SECONDS", handler)

    def test_workflow_publishes_only_the_v2_3_tag(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "publish-ltx23-serverless-image.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("type=raw,value=v2.3", workflow)
        self.assertNotIn("type=raw,value=v2.5\n", workflow)
        self.assertNotIn("value=v2.5-bf16", workflow)
        self.assertNotIn("value=latest", workflow)
        self.assertNotIn("ltx-2.3-serverless-bf16/Dockerfile", workflow)

    def test_bf16_variant_is_a_separate_tree_and_workflow(self) -> None:
        self.assertTrue((BF16_SERVERLESS / "Dockerfile").is_file())
        bf16_workflow = (
            ROOT / ".github" / "workflows" / "publish-ltx23-serverless-bf16-image.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("ltx-2.3-serverless-bf16/Dockerfile", bf16_workflow)
        self.assertIn("type=raw,value=v2.5-bf16", bf16_workflow)
        self.assertNotIn("type=raw,value=v2.5\n", bf16_workflow)

    def test_documented_image_and_example_endpoint_use_v2_3(self) -> None:
        readme = (SERVERLESS / "README.md").read_text(encoding="utf-8")
        endpoint_config = (SERVERLESS / "runpod/endpoint-config.example.json").read_text(
            encoding="utf-8"
        )
        self.assertIn("ambienteavatar-ltx23-serverless:v2.3", readme)
        self.assertIn("ambienteavatar-ltx23-serverless:v2.3", endpoint_config)
        self.assertNotIn("ambienteavatar-ltx23-serverless:v2.5", endpoint_config)
        self.assertNotIn("WORKFLOW_PATH", endpoint_config)
        self.assertNotIn("LTX25_UNET_CHECKPOINT", endpoint_config)

    def test_handler_defaults_to_the_ltx23_workflow(self) -> None:
        handler = (SERVERLESS / "src" / "handler.py").read_text(encoding="utf-8")
        self.assertIn("video_ltx2_3_ia2v_personal_lora_api.json", handler)
        self.assertIn('"/opt/defaults/workflows/video_ltx2_5_ia2v_api.json"', handler)
        self.assertLess(
            handler.index('"/opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_api.json"'),
            handler.index('"/opt/defaults/workflows/video_ltx2_5_ia2v_api.json"'),
        )

    def test_ltx23_template_keeps_prompt_enhance_default_when_job_omits_it(self) -> None:
        sys.path.insert(0, str(SERVERLESS / "src"))
        import workflow_api  # noqa: E402

        source = json.loads(
            (ROOT / "ltx-2.3" / "assets" / "video_ltx2_3_ia2v_personal_source.json").read_text(
                encoding="utf-8"
            )
        )
        template = workflow_api.compile_workflow(source)
        template_default = template[workflow_api.PROMPT_ENHANCE_ID]["inputs"]["value"]
        workflow = workflow_api.build_job_workflow(
            template,
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
                "lora_strength": 0.7,
                "image_strength": 1.0,
                "enable_prompt_enhance": None,
            },
            "a",
        )
        self.assertEqual(template[workflow_api.PROMPT_ENHANCE_ID]["inputs"]["value"], True)
        self.assertEqual(
            workflow[workflow_api.PROMPT_ENHANCE_ID]["inputs"]["value"], template_default
        )


if __name__ == "__main__":
    unittest.main()
