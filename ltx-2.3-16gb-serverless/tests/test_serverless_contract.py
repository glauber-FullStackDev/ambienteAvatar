from __future__ import annotations

import filecmp
from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SERVERLESS = ROOT / "ltx-2.3-16gb-serverless"
VAST = ROOT / "ltx-2.3-serverless-vast"
sys.path.insert(0, str(SERVERLESS / "src"))

from handler import (  # noqa: E402
    DEFAULTS,
    DEFAULT_PROMPT,
    InputError,
    validate_input,
)
from jobs import JobRecord  # noqa: E402
from webhook import build_payload, url_allowed  # noqa: E402

BASE = {
    "project_id": "projeto-abc",
    "image_url": "https://minio.example.com/input/avatar.png?ok=1",
    "audio_url": "https://minio.example.com/input/fala.wav?ok=1",
    "prompt": "glauberavatar speaking naturally",
    "width": 704,
    "height": 1280,
    "duration_seconds": 12,
    "fps": 24,
    "seed": 42,
}


class WebhookContractParityTests(unittest.TestCase):
    """O contrato de webhook precisa ser o mesmo do worker Vast (byte a byte)."""

    def test_webhook_module_matches_vast_except_documented_delta(self) -> None:
        ours = (SERVERLESS / "src" / "webhook.py").read_text(encoding="utf-8")
        vast = (VAST / "src" / "webhook.py").read_text(encoding="utf-8")

        def without_public_parameters(source: str) -> str:
            lines = source.splitlines()
            start = next(i for i, line in enumerate(lines) if line.startswith("def _public_parameters"))
            end = next(i for i, line in enumerate(lines) if i > start and line.startswith("def "))
            return "\n".join(lines[:start] + lines[end:])

        # Idêntico ao vast exceto pelo bloco `_public_parameters` (únicas
        # chaves públicas novas: enable_prompt_enhance + decode tiled).
        self.assertEqual(
            without_public_parameters(ours).strip(),
            without_public_parameters(vast).strip(),
        )
        for key in ("enable_prompt_enhance", "decode_tile_size", "temporal_size", "temporal_overlap"):
            self.assertIn(f'"{key}"', ours)

    def test_jobs_module_matches_vast(self) -> None:
        self.assertTrue(
            filecmp.cmp(
                VAST / "src" / "jobs.py",
                SERVERLESS / "src" / "jobs.py",
                shallow=False,
            )
        )


class ValidateInputTests(unittest.TestCase):
    def test_project_id_required(self) -> None:
        payload = dict(BASE)
        payload.pop("project_id")
        with self.assertRaises(InputError):
            validate_input(payload)

    def test_job_id_generated_when_absent(self) -> None:
        values = validate_input(dict(BASE))
        self.assertTrue(all(c.isalnum() or c in "-._" for c in values["job_id"]))
        self.assertGreater(len(values["job_id"]), 0)

    def test_caller_job_id_is_preserved(self) -> None:
        payload = {**BASE, "job_id": "job-do-chamador-1"}
        values = validate_input(payload)
        self.assertEqual(values["job_id"], "job-do-chamador-1")

    def test_rejects_hostile_job_id(self) -> None:
        payload = {**BASE, "job_id": "../../etc/passwd"}
        with self.assertRaises(InputError):
            validate_input(payload)

    def test_unwraps_input_envelope(self) -> None:
        values = validate_input({"input": dict(BASE), "ignored": True})
        self.assertEqual(values["project_id"], "projeto-abc")

    def test_prompt_is_optional_and_defaults_to_worker_prompt(self) -> None:
        payload = {key: value for key, value in BASE.items() if key != "prompt"}
        values = validate_input(payload)
        self.assertEqual(values["prompt"], DEFAULT_PROMPT)
        self.assertIn("glauberavatar remains in the exact composition", values["prompt"])

    def test_webhook_optional_and_merged(self) -> None:
        payload = {**BASE, "webhook": {"url": "https://backend.example.com/hook", "extra_params": {"x": 1}}}
        values = validate_input(payload)
        self.assertEqual(values["webhook_url"], "https://backend.example.com/hook")
        self.assertEqual(values["webhook_extra_params"], {"x": 1})

    def test_webhook_url_must_be_http(self) -> None:
        payload = {**BASE, "webhook": {"url": "ftp://backend.example.com/hook"}}
        with self.assertRaises(InputError):
            validate_input(payload)

    def test_runpod_serverless_defaults_win(self) -> None:
        values = validate_input(dict(BASE))
        self.assertEqual(values["lora_strength"], 0.7)
        self.assertEqual(values["image_strength"], 1.0)
        self.assertEqual(DEFAULTS["decode_tile_size"], 512)
        self.assertEqual(DEFAULTS["temporal_size"], 48)
        self.assertEqual(DEFAULTS["temporal_overlap"], 16)
        self.assertFalse(DEFAULTS["enable_prompt_enhance"])

    def test_duration_keeps_the_lipsync_floor(self) -> None:
        payload = {**BASE, "duration_seconds": 3}
        with self.assertRaises(InputError):
            validate_input(payload)

    def test_decode_bounds_are_enforced(self) -> None:
        with self.assertRaises(InputError):
            validate_input({**BASE, "decode_tile_size": 128})
        with self.assertRaises(InputError):
            validate_input({**BASE, "temporal_size": 8})
        with self.assertRaises(InputError):
            validate_input({**BASE, "temporal_size": 16, "temporal_overlap": 16})

    def test_enable_prompt_enhance_off_by_default(self) -> None:
        self.assertFalse(validate_input(dict(BASE))["enable_prompt_enhance"])
        values = validate_input({**BASE, "enable_prompt_enhance": True})
        self.assertTrue(values["enable_prompt_enhance"])


class WebhookPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = JobRecord(
            job_id="job-1",
            project_id="projeto-abc",
            status="completed",
            params=validate_input(dict(BASE)),
            webhook_extra_params={"origin": "client-a"},
            started_at=1.0,
            finished_at=4.0,
        )

    def test_core_keys_win_over_extra_params(self) -> None:
        self.record.webhook_extra_params = {"origin": "client-a", "status": "spoofed"}
        payload = build_payload("job.completed", self.record)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["origin"], "client-a")

    def test_includes_correlation_and_parameters(self) -> None:
        payload = build_payload("job.completed", self.record)
        self.assertEqual(payload["project_id"], "projeto-abc")
        self.assertEqual(payload["job_id"], "job-1")
        self.assertEqual(payload["event"], "job.completed")
        self.assertEqual(payload["parameters"]["width"], 704)
        self.assertEqual(payload["parameters"]["lora_strength"], 0.7)
        self.assertEqual(payload["parameters"]["decode_tile_size"], 512)
        self.assertEqual(payload["parameters"]["temporal_size"], 48)
        self.assertEqual(payload["parameters"]["temporal_overlap"], 16)
        self.assertFalse(payload["parameters"]["enable_prompt_enhance"])
        self.assertEqual(payload["execution_seconds"], 3.0)

    def test_failure_payload_carries_error(self) -> None:
        payload = build_payload("job.failed", self.record, error="timeout")
        self.assertEqual(payload["event"], "job.failed")
        self.assertEqual(payload["error"], "timeout")


class WebhookAllowListTests(unittest.TestCase):
    def test_defaults_allowed_to_env_host(self) -> None:
        previous = os.environ.get("WEBHOOK_URL")
        os.environ["WEBHOOK_URL"] = "https://backend.example.com/hook"
        try:
            self.assertTrue(url_allowed("https://backend.example.com/hook"))
            self.assertFalse(url_allowed("https://evil.example.com/hook"))
        finally:
            if previous is None:
                os.environ.pop("WEBHOOK_URL", None)
            else:
                os.environ["WEBHOOK_URL"] = previous

    def test_default_unset_rejects_unknown(self) -> None:
        self.assertFalse(url_allowed("https://anywhere.invalid/hook"))


class ImageContractTests(unittest.TestCase):
    def test_image_installs_every_custom_node_pack_used_by_the_workflow(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ComfyUI-GGUF", dockerfile)
        self.assertIn("ComfyUI-KJNodes", dockerfile)
        self.assertIn("ComfyUI-BFSNodes", dockerfile)  # ComfySwitchNode (no 348)

    def test_image_targets_the_gguf_16gb_workflow(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("video_ltx2_3_ia2v_personal_lora_16gb_api.json", dockerfile)
        self.assertIn("LTX23_GGUF_QUANT=Q4_K_M", dockerfile)
        self.assertIn("PERSONAL_WORKFLOW_SHA256=a410e101", dockerfile)

    def test_image_does_not_bake_models(self) -> None:
        dockerfile = (SERVERLESS / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn(".gguf\n", dockerfile.replace("GGUF_NODE_COMMIT", ""))
        self.assertNotIn("hf_hub_download", dockerfile)

    def test_handler_enforces_minimum_audio_duration(self) -> None:
        handler = (SERVERLESS / "src" / "handler.py").read_text(encoding="utf-8")
        self.assertIn("MIN_DURATION_SECONDS = 5.0", handler)
        self.assertIn("minimum=MIN_DURATION_SECONDS", handler)

    def test_api_routes_match_the_vast_contract(self) -> None:
        api = (SERVERLESS / "src" / "api.py").read_text(encoding="utf-8")
        self.assertIn('@app.post("/submit")', api)
        self.assertIn('@app.post("/status")', api)
        self.assertIn('@app.get("/health")', api)
        self.assertIn('"job.completed" if record.status == "completed" else "job.failed"', api)


if __name__ == "__main__":
    unittest.main()
