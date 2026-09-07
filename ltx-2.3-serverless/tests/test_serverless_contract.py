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

    def test_documented_image_and_example_endpoint_use_v2(self) -> None:
        readme = (SERVERLESS / "README.md").read_text(encoding="utf-8")
        endpoint_config = (SERVERLESS / "runpod/endpoint-config.example.json").read_text(
            encoding="utf-8"
        )
        self.assertIn("ambienteavatar-ltx23-serverless:v2", readme)
        self.assertIn("ambienteavatar-ltx23-serverless:v2", endpoint_config)


if __name__ == "__main__":
    unittest.main()
