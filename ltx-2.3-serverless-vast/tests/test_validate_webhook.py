from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ltx-2.3-serverless-vast" / "src"))

from validate import InputError, validate_input  # noqa: E402
from webhook import build_payload, default_webhook_url, url_allowed  # noqa: E402
from jobs import JobRecord  # noqa: E402

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

    def test_webhook_optional_and_merged(self) -> None:
        payload = {**BASE, "webhook": {"url": "https://backend.example.com/hook", "extra_params": {"x": 1}}}
        values = validate_input(payload)
        self.assertEqual(values["webhook_url"], "https://backend.example.com/hook")
        self.assertEqual(values["webhook_extra_params"], {"x": 1})

    def test_webhook_url_must_be_http(self) -> None:
        payload = {**BASE, "webhook": {"url": "ftp://backend.example.com/hook"}}
        with self.assertRaises(InputError):
            validate_input(payload)


class WebhookPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = JobRecord(
            job_id="job-1",
            project_id="projeto-abc",
            status="completed",
            params=BASE,
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
        self.assertEqual(payload["execution_seconds"], 3.0)

    def test_failure_payload_carries_error(self) -> None:
        payload = build_payload("job.failed", self.record, error="timeout")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["error"], "timeout")


class WebhookAllowListTests(unittest.TestCase):
    def test_defaults_allowed_to_env_host(self) -> None:
        import os

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


if __name__ == "__main__":
    unittest.main()