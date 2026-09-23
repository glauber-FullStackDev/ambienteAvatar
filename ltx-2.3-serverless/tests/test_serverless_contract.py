from __future__ import annotations

from pathlib import Path
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

    def test_image_targets_ltx25_pipeline(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("DOWNLOAD_LTX25_MODELS_ON_START=1", dockerfile)
        self.assertIn("WORKFLOW_PATH=/opt/defaults/workflows/video_ltx2_5_ia2v_api.json", dockerfile)
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

    def test_bootstrap_downloads_only_the_int8_set(self) -> None:
        bootstrap = (SERVERLESS / "src" / "bootstrap_models.py").read_text(encoding="utf-8")
        self.assertIn("ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", bootstrap)
        self.assertIn("gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", bootstrap)
        self.assertNotIn(
            "ltx-2.5-22b-distilled-transformer-bf16.safetensors", bootstrap
        )
        self.assertNotIn("ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors", bootstrap)
        self.assertNotIn("ltx-2.3-22b-dev-fp8.safetensors", bootstrap)
        self.assertNotIn("gemma-3-12b-it-abliterated", bootstrap)

    def test_handler_enforces_minimum_audio_duration(self) -> None:
        handler = (SERVERLESS / "src" / "handler.py").read_text(encoding="utf-8")
        self.assertIn("MIN_DURATION_SECONDS = 5.0", handler)
        self.assertIn("minimum=MIN_DURATION_SECONDS", handler)

    def test_workflow_publishes_only_the_v2_5_tag(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "publish-ltx23-serverless-image.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("type=raw,value=v2.5", workflow)
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

    def test_documented_image_and_example_endpoint_use_v2_5(self) -> None:
        readme = (SERVERLESS / "README.md").read_text(encoding="utf-8")
        endpoint_config = (SERVERLESS / "runpod/endpoint-config.example.json").read_text(
            encoding="utf-8"
        )
        self.assertIn("ambienteavatar-ltx23-serverless:v2.5", readme)
        self.assertIn("ambienteavatar-ltx23-serverless:v2.5", endpoint_config)
        self.assertNotIn("ambienteavatar-ltx23-serverless:v2.5-bf16", endpoint_config)
        self.assertNotIn("WORKFLOW_PATH", endpoint_config)
        self.assertNotIn("LTX25_UNET_CHECKPOINT", endpoint_config)


if __name__ == "__main__":
    unittest.main()
