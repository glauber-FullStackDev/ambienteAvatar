from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ltx-2.3-serverless-vast" / "src"))

from jobs import JobBusyError, JobManager, JobNotFoundError  # noqa: E402

VALUES = {
    "project_id": "projeto-abc",
    "job_id": "job-1",
    "image_url": "https://minio.example.com/input/avatar.png",
    "audio_url": "https://minio.example.com/input/fala.wav",
    "prompt": "glauberavatar speaking naturally",
    "width": 704,
    "height": 1280,
}


class JobManagerTests(unittest.TestCase):
    def test_lifecycle(self) -> None:
        manager = JobManager()
        record = manager.submit(dict(VALUES))
        self.assertEqual(record.status, "queued")
        self.assertTrue(manager.busy())

        manager.mark_running(record.job_id)
        self.assertEqual(manager.get("job-1").status, "running")

        manager.complete(record.job_id, {"video_url": "https://out/v.mp4"})
        state = manager.get("job-1")
        self.assertEqual(state.status, "completed")
        self.assertEqual(state.result, {"video_url": "https://out/v.mp4"})
        self.assertFalse(manager.busy())

    def test_single_gpu_guard(self) -> None:
        manager = JobManager()
        first = manager.submit(dict(VALUES))
        manager.mark_running(first.job_id)
        second = dict(VALUES)
        second["job_id"] = "job-2"
        with self.assertRaises(JobBusyError):
            manager.submit(second)

    def test_fail_and_release(self) -> None:
        manager = JobManager()
        record = manager.submit(dict(VALUES))
        manager.mark_running(record.job_id)
        manager.fail(record.job_id, "tempo esgotado")
        state = manager.get("job-1")
        self.assertEqual(state.status, "failed")
        self.assertEqual(state.error, "tempo esgotado")
        self.assertFalse(manager.busy())

    def test_unknown_job_raises(self) -> None:
        manager = JobManager()
        with self.assertRaises(JobNotFoundError):
            manager.get("nao-existe")

    def test_next_job_after_completion_reuses_state(self) -> None:
        manager = JobManager()
        first = manager.submit(dict(VALUES))
        manager.mark_running(first.job_id)
        manager.complete(first.job_id, {})
        self.assertFalse(manager.busy())


if __name__ == "__main__":
    unittest.main()