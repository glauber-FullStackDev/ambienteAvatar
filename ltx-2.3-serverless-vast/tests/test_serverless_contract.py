from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
VAST = ROOT / "ltx-2.3-serverless-vast"


class ServerlessVastBuildContractTests(unittest.TestCase):
    def test_image_installs_custom_node_packs_and_server_stack(self) -> None:
        dockerfile = (VAST / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ComfyUI-BFSNodes", dockerfile)
        self.assertIn("ComfyUI-LTXVideo", dockerfile)
        self.assertIn("patch_ltxvideo_kornia.py", dockerfile)
        self.assertIn("fastapi", (VAST / "requirements.txt").read_text(encoding="utf-8"))
        self.assertIn("uvicorn", (VAST / "requirements.txt").read_text(encoding="utf-8"))

    def test_image_cmd_is_start_server(self) -> None:
        dockerfile = (VAST / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn('CMD ["/opt/serverless-vast/src/start_server.sh"]', dockerfile)

    def test_model_server_exposes_core_routes(self) -> None:
        api = (VAST / "src" / "api.py").read_text(encoding="utf-8")
        for route in ('@app.post("/submit")', '@app.post("/status")', '@app.get("/health")', '@app.get("/benchmark")'):
            self.assertIn(route, api)

    def test_dedicated_pyworker_repo_is_documented(self) -> None:
        readme = (VAST / "README.md").read_text(encoding="utf-8")
        self.assertIn("PYWORKER_REPO", readme)
        self.assertIn("ltx23-vast-pyworker", readme)

    def test_pyworker_mirror_is_complete(self) -> None:
        worker = (VAST / "pyworker" / "worker.py").read_text(encoding="utf-8")
        self.assertIn("WorkerConfig", worker)
        self.assertIn('route="/submit"', worker)
        self.assertIn("BenchmarkConfig", worker)
        self.assertIn("Model server ready", worker)

    def test_webhook_and_correlation_are_implemented(self) -> None:
        readme = (VAST / "README.md").read_text(encoding="utf-8")
        self.assertIn("webhook", readme)
        self.assertIn("project_id", readme)
        self.assertIn("job_id", readme)


if __name__ == "__main__":
    unittest.main()