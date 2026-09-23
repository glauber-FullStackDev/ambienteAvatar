from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
import unittest.mock


ROOT = Path(__file__).resolve().parents[2]
SERVERLESS = ROOT / "ltx-2.3-serverless"


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

    def test_image_bakes_bf16_unet_template_for_dtype_ab_test(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("--unet-name ltx-2.5-22b-distilled-transformer-bf16.safetensors", dockerfile)
        self.assertIn("video_ltx2_5_ia2v_bf16_api.json", dockerfile)

    def test_template_uses_official_refine_schedule(self) -> None:
        builder = (ROOT / "ltx-2.3" / "scripts" / "build_ltx25_ia2v_api_workflow.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"', builder)

    def test_bootstrap_downloads_only_the_ltx25_set(self) -> None:
        bootstrap = (SERVERLESS / "src" / "bootstrap_models.py").read_text(encoding="utf-8")
        self.assertIn("ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", bootstrap)
        self.assertIn("ltx-2.5-22b-distilled-transformer-bf16.safetensors", bootstrap)
        self.assertIn("gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", bootstrap)
        self.assertNotIn("ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors", bootstrap)
        self.assertNotIn("ltx-2.3-22b-dev-fp8.safetensors", bootstrap)
        self.assertNotIn("gemma-3-12b-it-abliterated", bootstrap)

    def test_bootstrap_selects_unet_checkpoint_from_workflow_path(self) -> None:
        sys.path.insert(0, str(SERVERLESS / "src"))
        sys.path.insert(0, str(ROOT / "ltx-2.3" / "scripts"))
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            self.skipTest("huggingface_hub nao instalado no ambiente local")
        import bootstrap_models

        cases = {
            "/opt/defaults/workflows/video_ltx2_5_ia2v_api.json": bootstrap_models.INT8_UNET,
            "/opt/defaults/workflows/video_ltx2_5_ia2v_bf16_api.json": bootstrap_models.BF16_UNET,
        }
        for workflow_path, expected in cases.items():
            with unittest.mock.patch.dict(os.environ, {"WORKFLOW_PATH": workflow_path}):
                self.assertEqual(bootstrap_models.resolve_unet_checkpoint(), expected)
        with unittest.mock.patch.dict(os.environ, {"LTX25_UNET_CHECKPOINT": "bf16"}):
            self.assertEqual(bootstrap_models.resolve_unet_checkpoint(), bootstrap_models.BF16_UNET)
        with unittest.mock.patch.dict(os.environ, {"LTX25_UNET_CHECKPOINT": "int8"}):
            self.assertEqual(bootstrap_models.resolve_unet_checkpoint(), bootstrap_models.INT8_UNET)
        with unittest.mock.patch.dict(os.environ, {"LTX25_UNET_CHECKPOINT": "invalido"}):
            with self.assertRaises(SystemExit):
                bootstrap_models.resolve_unet_checkpoint()

    def test_example_endpoint_is_int8_baseline_without_bf16_workflow(self) -> None:
        endpoint_config = (SERVERLESS / "runpod/endpoint-config.example.json").read_text(
            encoding="utf-8"
        )
        self.assertIn("ambienteavatar-ltx23-serverless:v2.5", endpoint_config)
        self.assertNotIn("video_ltx2_5_ia2v_bf16_api.json", endpoint_config)
        self.assertNotIn("WORKFLOW_PATH", endpoint_config)


if __name__ == "__main__":
    unittest.main()
