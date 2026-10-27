from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SERVERLESS = ROOT / "ltx-2.3-16gb-serverless"
sys.path.insert(0, str(SERVERLESS / "src"))

import api  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from job_progress import percent_for_phase, tracker  # noqa: E402
from job_queue import JobQueueManager  # noqa: E402
from jobs import JobRecord  # noqa: E402

RUNPOD_INPUT = {
    "image_url": "https://minio.example.com/input/avatar.png?ok=1",
    "audio_url": "https://minio.example.com/input/fala.wav?ok=1",
    "prompt": "glauberavatar speaking naturally",
    "duration_seconds": 12,
}


class PercentTests(unittest.TestCase):
    def test_pesos_por_fase(self) -> None:
        self.assertEqual(percent_for_phase("downloading"), 0.0)
        self.assertEqual(percent_for_phase("queueing"), 3.0)
        self.assertEqual(percent_for_phase("uploading"), 97.0)
        self.assertEqual(percent_for_phase("done"), 100.0)

    def test_amostragem_mapeia_para_a_faixa(self) -> None:
        self.assertAlmostEqual(percent_for_phase("sampling", 0.0), 5.0)
        self.assertAlmostEqual(percent_for_phase("sampling", 1.0), 80.0)
        self.assertAlmostEqual(percent_for_phase("decoding", 0.5), 88.5)
        # clamp de razões fora de [0, 1]
        self.assertAlmostEqual(percent_for_phase("sampling", 1.5), 80.0)
        self.assertAlmostEqual(percent_for_phase("sampling", -1.0), 5.0)


class TrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        tracker.drop("job-1")

    def tearDown(self) -> None:
        tracker.drop("job-1")

    def test_ciclo_de_fase_por_checkpoints(self) -> None:
        tracker.start("job-1", project_id="p", prompt="oi")
        snapshot = tracker.snapshot("job-1")
        self.assertEqual(snapshot["phase"], "downloading")
        tracker.set_phase("job-1", "queueing")
        self.assertEqual(tracker.snapshot("job-1")["phase"], "queueing")
        tracker.finish("job-1", ok=True)
        final = tracker.snapshot("job-1")
        self.assertEqual(final["phase"], "done")
        self.assertEqual(final["percent"], 100.0)

    def test_evento_sampling_progresso(self) -> None:
        tracker.start("job-1")
        tracker.handle_event("job-1", {
            "type": "progress",
            "data": {"value": 10, "max": 20, "node": "344", "prompt_id": "x"},
        })
        snapshot = tracker.snapshot("job-1")
        self.assertEqual(snapshot["phase"], "sampling")
        self.assertAlmostEqual(snapshot["percent"], 5.0 + 75.0 * 0.5)
        self.assertEqual(snapshot["step"], 10)
        self.assertEqual(snapshot["total_steps"], 20)

    def test_evento_decode_vai_para_fase_decoding(self) -> None:
        tracker.start("job-1")
        tracker.handle_event("job-1", {"type": "progress", "data": {"value": 1, "max": 4, "node": "316"}})
        snapshot = tracker.snapshot("job-1")
        self.assertEqual(snapshot["phase"], "decoding")
        self.assertAlmostEqual(snapshot["percent"], percent_for_phase("decoding", 0.25))

    def test_percentual_e_monotonico_entre_samplers(self) -> None:
        tracker.start("job-1")
        tracker.handle_event("job-1", {"type": "progress", "data": {"value": 20, "max": 20, "node": "344"}})
        base_percent = tracker.snapshot("job-1")["percent"]
        tracker.handle_event("job-1", {"type": "progress", "data": {"value": 2, "max": 8, "node": "344"}})
        self.assertGreaterEqual(tracker.snapshot("job-1")["percent"], base_percent)
        self.assertEqual(tracker.snapshot("job-1")["step"], 2)

    def test_eventos_invalidos_sao_ignorados(self) -> None:
        tracker.start("job-1")
        tracker.handle_event("job-1", {"type": "progress", "data": {"value": 1, "max": 0}})
        tracker.handle_event("job-1", {"type": "progress", "data": {}})
        tracker.handle_event("job-1", {"type": "desconhecido", "data": {}})
        snapshot = tracker.snapshot("job-1")
        self.assertEqual(snapshot["phase"], "downloading")

    def test_frames_binarios_e_json_invalido_nao_quebram(self) -> None:
        # exercita _listen indiretamente: eventos não-dict são ignorados
        tracker.handle_event("job-1", "nao-e-dict")  # type: ignore[arg-type]
        self.assertIsNone(tracker.snapshot("job-1"))

    def test_fail_mantem_percentual(self) -> None:
        tracker.start("job-1")
        tracker.handle_event("job-1", {"type": "progress", "data": {"value": 5, "max": 20, "node": "344"}})
        tracker.finish("job-1", ok=False)
        snapshot = tracker.snapshot("job-1")
        self.assertEqual(snapshot["phase"], "failed")
        self.assertLess(snapshot["percent"], 100.0)

    def test_prune_mantem_limite_de_entradas(self) -> None:
        small = tracker.__class__(max_entries=3)
        for index in range(5):
            small.start(f"job-{index}")
        self.assertEqual(len(small._jobs), 3)
        for index in range(5):
            small.drop(f"job-{index}")


class DashboardRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        api.manager = JobQueueManager()
        self.client = TestClient(api.app)
        api.STARTUP_DONE.set()
        tracker.drop("job-1")

    def tearDown(self) -> None:
        tracker.drop("job-1")

    @staticmethod
    def _wait_job(api_module, job_id, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                record = api_module.manager.get(job_id)
            except api_module.JobNotFoundError:
                time.sleep(0.02)
                continue
            if record.status in ("completed", "failed"):
                return record
            time.sleep(0.02)
        raise AssertionError(f"job {job_id} não terminou em {timeout}s")

    def _record(self, status="completed", result=None, job_id="job-1", error=None):
        return JobRecord(
            job_id=job_id,
            project_id="projeto-abc",
            status=status,
            params={"prompt": "glauberavatar oi", "project_id": "projeto-abc"},
            started_at=100.0,
            finished_at=160.0 if result or status == "completed" else None,
            result=result,
            error=error,
        )

    def test_pagina_dashboard_serve_html(self) -> None:
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Fila de Geração", response.text)

    def test_summary_sem_token_configurado(self) -> None:
        response = self.client.get("/dashboard/api/summary")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["current"], None)
        self.assertEqual(body["history"], [])
        self.assertIn("worker", body)

    def test_summary_com_token_via_query(self) -> None:
        with mock.patch.dict(api.os.environ, {"API_TOKEN": "segredo"}):
            negado = self.client.get("/dashboard/api/summary")
            self.assertEqual(negado.status_code, 401)
            ok = self.client.get("/dashboard/api/summary?token=segredo")
            self.assertEqual(ok.status_code, 200)

    def test_summary_separa_atual_de_historico(self) -> None:
        finished = self._record(status="completed", result={"video_url": "https://minio/v.mp4"})
        with api.manager._lock:
            api.manager._jobs["job-1"] = finished
        tracker.start("job-2", project_id="projeto-abc")
        tracker.handle_event("job-2", {"type": "progress", "data": {"value": 5, "max": 10, "node": "344"}})
        running = self._record(status="running", job_id="job-2")
        with api.manager._lock:
            api.manager._jobs["job-2"] = running

        try:
            body = self.client.get("/dashboard/api/summary").json()
        finally:
            tracker.drop("job-2")

        self.assertEqual(body["current"]["job_id"], "job-2")
        self.assertEqual(body["current"]["progress"]["phase"], "sampling")
        self.assertEqual(len(body["history"]), 1)
        self.assertEqual(body["history"][0]["job_id"], "job-1")
        self.assertEqual(body["history"][0]["duration_seconds"], 60.0)
        self.assertEqual(body["history"][0]["video_url"], "https://minio/v.mp4")

    def _values(self, job_id: str) -> dict:
        return {
            "project_id": "projeto-abc",
            "job_id": job_id,
            "image_url": "https://minio/i.png",
            "audio_url": "https://minio/a.wav",
            "prompt": "glauberavatar oi",
            "webhook_url": "",
            "webhook_extra_params": {},
        }

    def test_summary_mostra_fila_com_posicoes(self) -> None:
        # dispatcher pausado: os enfileirados precisam continuar na fila
        with mock.patch.object(api.manager, "next_queued", return_value=None):
            api.manager.enqueue(self._values("job-q1"))
            api.manager.enqueue(self._values("job-q2"))
            body = self.client.get("/dashboard/api/summary").json()

        self.assertEqual(body["current"], None)
        self.assertEqual([q["job_id"] for q in body["queue"]], ["job-q1", "job-q2"])
        self.assertEqual(body["queue"][0]["queue_position"], 1)
        self.assertEqual(body["queue"][1]["queue_position"], 2)
        self.assertEqual(body["counts"]["queued"], 2)
        self.assertEqual(body["counts"]["running"], 0)
        self.assertEqual(body["worker"]["queue_size"], 2)

    def test_summary_counts_completos(self) -> None:
        completed = self._record(status="completed", result={"video_url": "https://minio/v.mp4"})
        failed = self._record(status="failed", error="boom", job_id="job-f")
        with api.manager._lock:
            api.manager._jobs["job-1"] = completed
            api.manager._jobs["job-f"] = failed

        with mock.patch.object(api.manager, "next_queued", return_value=None):
            api.manager.enqueue(self._values("job-q1"))
            body = self.client.get("/dashboard/api/summary").json()

        counts = body["counts"]
        self.assertEqual(counts["completed"], 1)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["queued"], 1)
        self.assertEqual(counts["running"], 0)
        self.assertEqual(len(body["history"]), 2)

    def test_summary_marca_cancelados(self) -> None:
        cancelled = self._record(status="completed", result=None, job_id="job-c")
        with api.manager._lock:
            api.manager._jobs["job-c"] = cancelled
        body = self.client.get("/dashboard/api/summary").json()
        self.assertTrue(body["history"][0]["cancelled"])
        self.assertEqual(body["counts"]["cancelled"], 1)
        self.assertEqual(body["counts"]["completed"], 0)

    def test_runpod_status_queue_position_aditivo(self) -> None:
        with mock.patch.object(api.manager, "next_queued", return_value=None):
            api.manager.enqueue(self._values("job-q1"))
            api.manager.enqueue(self._values("job-q2"))
            first = self.client.get("/ep/status/job-q1")
            second = self.client.get("/ep/status/job-q2")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["status"], "IN_QUEUE")
        self.assertEqual(first.json()["queue_position"], 1)
        self.assertEqual(second.json()["queue_position"], 2)

    def test_cancel_via_rota_runpod_aparece_no_painel(self) -> None:
        with mock.patch.object(api.manager, "next_queued", return_value=None):
            api.manager.enqueue(self._values("job-q3"))
            response = self.client.post("/local/cancel/job-q3")
            self.assertEqual(response.status_code, 200)
            body = self.client.get("/dashboard/api/summary").json()
        entry = next(e for e in body["history"] if e["job_id"] == "job-q3")
        self.assertTrue(entry["cancelled"])
        self.assertEqual(entry["video_url"], None)
        self.assertEqual(body["counts"]["cancelled"], 1)

    def test_status_runpod_inclui_progresso_aditivo(self) -> None:
        import threading

        release = threading.Event()

        def slow_job(values, job_id):
            tracker.start(job_id, project_id="p")
            release.wait(timeout=10)
            return {}

        with mock.patch.object(api, "run_job", side_effect=slow_job), mock.patch.object(
            api, "deliver_runpod", return_value=None
        ):
            submit = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            job_id = submit.json()["id"]
            response = self.client.get(f"/ep/status/{job_id}")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertIn("progress", body)
            self.assertIn("percent", body["progress"])
            release.set()
            self._wait_job(api, job_id)

    def test_status_vasto_inclui_progresso_aditivo(self) -> None:
        import threading

        release = threading.Event()

        def slow_job(values, job_id):
            tracker.start(job_id, project_id="p")
            release.wait(timeout=10)
            return {}

        with mock.patch.object(api, "run_job", side_effect=slow_job), mock.patch.object(
            api, "deliver", return_value=None
        ):
            submit = self.client.post("/submit", json={"input": dict(RUNPOD_INPUT, project_id="proj")})
            job_id = submit.json()["job_id"]
            response = self.client.post("/status", json={"job_id": job_id})
            self.assertEqual(response.status_code, 200)
            self.assertIn("progress", response.json())
            release.set()
            self._wait_job(api, job_id)


if __name__ == "__main__":
    unittest.main()
