from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SERVERLESS = ROOT / "ltx-2.3-16gb-serverless"
sys.path.insert(0, str(SERVERLESS / "src"))

import api  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

RUNPOD_INPUT = {
    "image_url": "https://minio.example.com/input/avatar.png?ok=1",
    "audio_url": "https://minio.example.com/input/fala.wav?ok=1",
    "prompt": "glauberavatar speaking naturally",
    "duration_seconds": 12,
    "lora_strength": 0.7,
}

FAKE_RESULT = {
    "comfy_prompt_id": "prompt-123",
    "video_url": "https://minio.example.com/results/video.mp4",
    "last_frame_url": "https://minio.example.com/results/last_frame.png",
    "image_sha": "abc123",
    "execution_seconds": 42.5,
}


def wait_until(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class QueueTests(unittest.TestCase):
    """Fila FIFO das rotas RunPod: vários vídeos seguidos, um por vez."""

    def setUp(self) -> None:
        api.manager = api.JobQueueManager()
        self.client = TestClient(api.app)
        api.STARTUP_DONE.set()
        self._release = threading.Event()

    def tearDown(self) -> None:
        # Nunca deixar um job travado para o teste seguinte.
        self._release.set()
        deadline = time.time() + 10
        while time.time() < deadline:
            pending = [
                r for r in api.manager._jobs.values() if r.status in ("queued", "running")
            ]
            if not pending:
                return
            time.sleep(0.02)

    # -- submissão em série -------------------------------------------------

    def test_tres_videos_seguidos_respondem_in_queue(self) -> None:
        order: list[str] = []

        def job(values, job_id):
            order.append(job_id)
            return dict(FAKE_RESULT)

        with mock.patch.object(api, "run_job", side_effect=job), mock.patch.object(
            api, "deliver_runpod", return_value=None
        ):
            ids = []
            for _ in range(3):
                response = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "IN_QUEUE")
                ids.append(response.json()["id"])

            deadline = time.time() + 10
            while time.time() < deadline and len(order) < 3:
                time.sleep(0.02)

        # Todos aceitos (200) e executados um por vez, na ordem de chegada.
        self.assertEqual(order, ids)
        for job_id in ids:
            self.assertEqual(api.manager.get(job_id).status, "completed")

    def test_fila_na_o_muda_resposta_das_rotas_vast(self) -> None:
        """POST /submit (contrato vast) continua 409 quando o worker está ocupado."""
        release = threading.Event()

        def slow_job(values, job_id):
            release.wait(timeout=10)
            return dict(FAKE_RESULT)

        with mock.patch.object(api, "run_job", side_effect=slow_job), mock.patch.object(
            api, "deliver", return_value=None
        ), mock.patch.object(api, "persist_job_status", return_value=None):
            first = self.client.post(
                "/submit", json={"input": {**RUNPOD_INPUT, "project_id": "proj"}}
            )
            self.assertEqual(first.status_code, 202)
            second = self.client.post(
                "/submit", json={"input": {**RUNPOD_INPUT, "project_id": "proj"}}
            )
            self.assertEqual(second.status_code, 409)

            # O job vast não entra na fila do RunPod: segue na própria thread.
            self.assertEqual(api.manager.queue_size(), 0)
            release.set()

    # -- status -------------------------------------------------------------

    def test_status_mostra_in_queue_ate_assumir(self) -> None:
        release = threading.Event()

        def slow_job(values, job_id):
            release.wait(timeout=10)
            return dict(FAKE_RESULT)

        with mock.patch.object(api, "run_job", side_effect=slow_job), mock.patch.object(
            api, "deliver_runpod", return_value=None
        ):
            first = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            second = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            second_id = second.json()["id"]

            body = self.client.get(f"/ep/status/{second_id}").json()
            self.assertEqual(body["status"], "IN_QUEUE")
            self.assertNotIn("output", body)

            release.set()
            deadline = time.time() + 10
            while time.time() < deadline:
                body = self.client.get(f"/ep/status/{second_id}").json()
                if body["status"] == "COMPLETED":
                    break
                time.sleep(0.02)

        self.assertEqual(body["status"], "COMPLETED")
        self.assertEqual(body["output"]["video_url"], FAKE_RESULT["video_url"])

    def test_runsync_abaixo_da_fila_responde_in_queue_no_timeout(self) -> None:
        release = threading.Event()

        def slow_job(values, job_id):
            release.wait(timeout=10)
            return dict(FAKE_RESULT)

        with mock.patch.object(api, "run_job", side_effect=slow_job), mock.patch.object(
            api, "deliver_runpod", return_value=None
        ), mock.patch.object(api, "RUNSYNC_WAIT_SECONDS", 0.2):
            first = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            # Garante que o 1º está rodando antes do runsync entrar na fila.
            deadline = time.time() + 10
            while time.time() < deadline and not api.manager.busy():
                time.sleep(0.02)
            response = self.client.post(
                "/ep/runsync", json={"input": dict(RUNPOD_INPUT)}
            )
            release.set()

        # Como no RunPod: timeout do runsync devolve o estado atual (na fila).
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["status"], ("IN_QUEUE", "IN_PROGRESS"))
        self.assertEqual(first.status_code, 200)

    # -- limite da fila ------------------------------------------------------

    def test_fila_cheia_responde_409(self) -> None:
        release = threading.Event()

        def slow_job(values, job_id):
            release.wait(timeout=10)
            return dict(FAKE_RESULT)

        with mock.patch.object(api, "run_job", side_effect=slow_job), mock.patch.object(
            api, "deliver_runpod", return_value=None
        ), mock.patch.object(api, "MAX_QUEUE_SIZE", 1):
            running = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            self.assertEqual(running.status_code, 200)
            running_id = running.json()["id"]
            self.assertTrue(
                wait_until(lambda: api.manager.get(running_id).status == "running")
            )
            queued = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            self.assertEqual(queued.status_code, 200)  # 1 vaga na fila
            overflow = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            self.assertEqual(overflow.status_code, 409)
            self.assertIn("fila cheia", overflow.json()["detail"])
            release.set()

    # -- cancelamento --------------------------------------------------------

    def test_cancel_tira_job_da_fila(self) -> None:
        release = threading.Event()
        delivered: list = []

        def slow_job(values, job_id):
            release.wait(timeout=10)
            return dict(FAKE_RESULT)

        def capture(record, **kw):
            # Ignora conclusões assíncronas de jobs de testes anteriores
            # (o dispatcher é global ao processo de testes).
            try:
                api.manager.get(record.job_id)
            except api.JobNotFoundError:
                return
            delivered.append(record)

        with mock.patch.object(api, "run_job", side_effect=slow_job), mock.patch.object(
            api, "deliver_runpod", side_effect=capture
        ):
            running = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            running_id = running.json()["id"]
            queued = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            queued_id = queued.json()["id"]

            # 1º na GPU antes de cancelar o 2º (senão a fila teria 2 pendentes).
            self.assertTrue(
                wait_until(lambda: api.manager.get(running_id).status == "running")
            )
            self.assertEqual(api.manager.get(queued_id).status, "queued")

            response = self.client.post(f"/ep/cancel/{queued_id}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"id": queued_id, "status": "COMPLETED"})

            # Sumiu da fila e do health; status responde COMPLETED com output nulo.
            self.assertEqual(api.manager.queue_size(), 0)
            status = self.client.get(f"/ep/status/{queued_id}").json()
            self.assertEqual(status["status"], "COMPLETED")
            self.assertIsNone(status["output"])
            self.assertEqual(self.client.get("/ep/health").json()["jobs"]["queued"], 0)

            # Webhook de conclusão no formato RunPod, output nulo.
            self.assertEqual(len(delivered), 1)
            self.assertEqual(delivered[0].status, "completed")
            self.assertIsNone(delivered[0].result)

            release.set()
            deadline = time.time() + 10
            while time.time() < deadline and api.manager.get(running.json()["id"]).status != "completed":
                time.sleep(0.02)

    def test_cancel_idempotente_para_job_finalizado(self) -> None:
        with mock.patch.object(api, "run_job", return_value=dict(FAKE_RESULT)), mock.patch.object(
            api, "deliver_runpod", return_value=None
        ):
            submitted = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            job_id = submitted.json()["id"]
            deadline = time.time() + 10
            while time.time() < deadline and api.manager.get(job_id).status != "completed":
                time.sleep(0.02)

        response = self.client.post(f"/ep/cancel/{job_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "COMPLETED")

    def test_cancel_job_em_execucao_responde_400(self) -> None:
        release = threading.Event()

        def slow_job(values, job_id):
            release.wait(timeout=10)
            return dict(FAKE_RESULT)

        with mock.patch.object(api, "run_job", side_effect=slow_job), mock.patch.object(
            api, "deliver_runpod", return_value=None
        ):
            submitted = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            job_id = submitted.json()["id"]
            deadline = time.time() + 10
            while time.time() < deadline and api.manager.get(job_id).status != "running":
                time.sleep(0.02)

            response = self.client.post(f"/ep/cancel/{job_id}")
            self.assertEqual(response.status_code, 400)
            release.set()

    def test_cancel_job_desconhecido_responde_404(self) -> None:
        response = self.client.post("/ep/cancel/nao-existe")
        self.assertEqual(response.status_code, 404)

    # -- job cancelado não executa -------------------------------------------

    def test_job_cancelado_nao_chama_o_comfyui(self) -> None:
        executed: list[str] = []
        gate = threading.Event()  # segura o 1º job na GPU até o cancel decidir

        def slow_first(values, job_id):
            gate.wait(timeout=10)
            executed.append(job_id)
            return dict(FAKE_RESULT)

        with mock.patch.object(api, "run_job", side_effect=slow_first), mock.patch.object(
            api, "deliver_runpod", return_value=None
        ):
            first = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            second = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            second_id = second.json()["id"]

            # 1º bloqueado na GPU; o 2º está garantidamente na fila.
            self.assertTrue(
                wait_until(lambda: api.manager.get(first.json()["id"]).status == "running")
            )
            self.assertEqual(api.manager.get(second_id).status, "queued")

            response = self.client.post(f"/ep/cancel/{second_id}")
            self.assertEqual(response.status_code, 200)
            gate.set()  # libera o 1º; o cancelado não pode executar
            self.assertTrue(wait_until(lambda: len(executed) >= 1))

        self.assertEqual(executed, [first.json()["id"]])
        self.assertNotIn(second_id, executed)


if __name__ == "__main__":
    unittest.main()
