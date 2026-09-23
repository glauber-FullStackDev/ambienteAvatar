from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
import unittest.mock


ROOT = Path(__file__).resolve().parents[2]
SERVERLESS = ROOT / "ltx-2.3-serverless-bf16"
INT8_SERVERLESS = ROOT / "ltx-2.3-serverless"


class ServerlessBf16BuildContractTests(unittest.TestCase):
    def test_image_installs_every_custom_node_pack_used_by_the_workflow(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ComfyUI-BFSNodes", dockerfile)
        self.assertIn("ComfyUI-LTXVideo", dockerfile)
        self.assertIn("patch_ltxvideo_kornia.py", dockerfile)

    def test_image_targets_ltx25_bf16_pipeline(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("DOWNLOAD_LTX25_MODELS_ON_START=1", dockerfile)
        self.assertIn("LTX25_UNET_CHECKPOINT=bf16", dockerfile)
        self.assertIn(
            "WORKFLOW_PATH=/opt/defaults/workflows/video_ltx2_5_ia2v_bf16_api.json",
            dockerfile,
        )
        self.assertIn(
            "--unet-name ltx-2.5-22b-distilled-transformer-bf16.safetensors", dockerfile
        )

    def test_image_is_fully_self_contained(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("ltx-2.3-serverless/", dockerfile)
        self.assertNotIn("video_ltx2_5_ia2v_api.json", dockerfile)
        # The personal LoRA is the single shared immutable LFS binary.
        self.assertIn("COPY ltx-2.3/assets/glauberavatar.safetensors", dockerfile)
        for local in ("COPY ltx-2.3-serverless-bf16/", "COPY ltx-2.3/scripts/"):
            self.assertNotIn("COPY ltx-2.3/scripts/", dockerfile)
            self.assertNotIn("COPY ltx-2.3/custom_nodes/", dockerfile)

    def test_bootstrap_downloads_only_the_bf16_set(self) -> None:
        bootstrap = (SERVERLESS / "src" / "bootstrap_models.py").read_text(encoding="utf-8")
        self.assertIn("ltx-2.5-22b-distilled-transformer-bf16.safetensors", bootstrap)
        self.assertIn("gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", bootstrap)
        self.assertNotIn(
            "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", bootstrap
        )
        self.assertNotIn("ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors", bootstrap)
        self.assertNotIn("ltx-2.3-22b-dev-fp8.safetensors", bootstrap)

    def test_handler_defaults_to_bf16_template(self) -> None:
        handler = (SERVERLESS / "src" / "handler.py").read_text(encoding="utf-8")
        self.assertIn("video_ltx2_5_ia2v_bf16_api.json", handler)
        self.assertNotIn("video_ltx2_5_ia2v_api.json", handler)
        self.assertIn("MIN_DURATION_SECONDS = 5.0", handler)

    def test_template_uses_official_refine_schedule(self) -> None:
        builder = (SERVERLESS / "scripts" / "build_ltx25_ia2v_api_workflow.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"', builder)
        self.assertIn(
            'UNET_NAME = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"', builder
        )

    def test_documented_image_and_example_endpoint_use_v2_5_bf16(self) -> None:
        readme = (SERVERLESS / "README.md").read_text(encoding="utf-8")
        endpoint_config = (SERVERLESS / "runpod/endpoint-config.example.json").read_text(
            encoding="utf-8"
        )
        self.assertIn("ambienteavatar-ltx23-serverless:v2.5-bf16", readme)
        self.assertIn("ambienteavatar-ltx23-serverless:v2.5-bf16", endpoint_config)
        self.assertNotIn("WORKFLOW_PATH", endpoint_config)
        self.assertNotIn("LTX25_UNET_CHECKPOINT", endpoint_config)

    def test_int8_worker_sources_do_not_leak_into_bf16_image(self) -> None:
        # Both trees live side by side; the bf16 build must not copy from the
        # int8 directory and vice versa.
        self.assertTrue((SERVERLESS / "src" / "handler.py").is_file())
        self.assertTrue((INT8_SERVERLESS / "src" / "handler.py").is_file())
        int8_dockerfile = (INT8_SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("ltx-2.3-serverless-bf16/", int8_dockerfile)


if __name__ == "__main__":
    unittest.main()
