from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SERVERLESS = ROOT / "ltx-2.3-serverless"


class ServerlessBuildContractTests(unittest.TestCase):
    def test_image_installs_every_custom_node_pack_used_by_the_workflow(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ComfyUI-BFSNodes", dockerfile)
        self.assertIn("ComfyUI-LTXVideo", dockerfile)
        self.assertIn("patch_ltxvideo_kornia.py", dockerfile)

    def test_image_targets_ltx25_iclora_pipeline(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("DOWNLOAD_LTX25_MODELS_ON_START=1", dockerfile)
        self.assertIn("WORKFLOW_PATH=/opt/defaults/workflows/video_ltx2_5_ia2v_iclora_api.json", dockerfile)
        self.assertIn("build_ltx25_ia2v_iclora_api_workflow.py", dockerfile)
        self.assertIn("video_ltx2_3_ia2v_personal_lora_api.json", dockerfile)

    def test_bootstrap_downloads_only_the_ltx25_set(self) -> None:
        bootstrap = (SERVERLESS / "src" / "bootstrap_models.py").read_text(encoding="utf-8")
        self.assertIn("ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", bootstrap)
        self.assertIn("gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", bootstrap)
        self.assertIn("ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors", bootstrap)
        self.assertNotIn("ltx-2.3-22b-dev-fp8.safetensors", bootstrap)
        self.assertNotIn("gemma-3-12b-it-abliterated", bootstrap)

    def test_documented_image_and_example_endpoint_use_v2_5(self) -> None:
        readme = (SERVERLESS / "README.md").read_text(encoding="utf-8")
        endpoint_config = (SERVERLESS / "runpod/endpoint-config.example.json").read_text(
            encoding="utf-8"
        )
        self.assertIn("ambienteavatar-ltx23-serverless:v2.5", readme)
        self.assertIn("ambienteavatar-ltx23-serverless:v2.5", endpoint_config)


if __name__ == "__main__":
    unittest.main()
