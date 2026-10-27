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
from handler import InputError  # noqa: E402
from job_queue import JobQueueManager  # noqa: E402
from jobs import JobRecord  # noqa: E402
from runpod_compat import (  # noqa: E402
    build_runpod_webhook,
    runpod_output,
    runpod_request_to_vast,
    to_runpod_status,
)

RUNPOD_INPUT = {
    "image_url": "https://minio.example.com/input/avatar.png?ok=1",
    "audio_url": "https://minio.example.com/input/fala.wav?ok=1",
    "prompt": "glauberavatar speaking naturally",
    "duration_seconds": 12,
    "lora_strength": 0.7,
}

FAKE_RESULT = {
    "job_id": "ignorado-no-output",
    "project_id": "também-ignorado-no-output-runpod",
    "comfy_prompt_id": "prompt-123",
    "video_url": "https://minio.example.com/results/video.mp4",
    "last_frame_url": "https://minio.example.com/results/last_frame.png",
    "image_sha": "abc123def456",
    "execution_seconds": 42.5,
}


def make_record(status="completed", result=None, params=None, error=None) -> JobRecord:
    return JobRecord(
        job_id="job-1",
        project_id="proj",
        status=status,
        params=params if params is not None else {},
        result=result,
        error=error,
    )


class TranslationTests(unittest.TestCase):
    def test_envelope_runpod_vira_vast(self) -> None:
        forwarded = runpod_request_to_vast(
            {"input": dict(RUNPOD_INPUT), "webhook": "https://cb.example/hook"},
            default_project_id="proj-default",
        )
        self.assertEqual(forwarded["input"]["webhook"], {"url": "https://cb.example/hook"})
        self.assertEqual(forwarded["input"]["project_id"], "proj-default")

    def test_project_id_do_input_tem_prioridade(self) -> None:
        forwarded = runpod_request_to_vast(
            {"input": {**RUNPOD_INPUT, "project_id": "meu-projeto"}},
            default_project_id="default",
        )
        self.assertEqual(forwarded["input"]["project_id"], "meu-projeto")

    def test_webhook_dict_passa_direto(self) -> None:
        forwarded = runpod_request_to_vast(
            {"input": RUNPOD_INPUT, "webhook": {"url": "https://cb.example/hook", "extra_params": {"a": 1}}},
        )
        self.assertEqual(forwarded["input"]["webhook"]["url"], "https://cb.example/hook")
        self.assertEqual(forwarded["input"]["webhook"]["extra_params"], {"a": 1})

    def test_sem_input_levanta_input_error(self) -> None:
        with self.assertRaises(InputError):
            runpod_request_to_vast({"foo": "bar"})
        with self.assertRaises(InputError):
            runpod_request_to_vast("nao-e-dict")

    def test_mapa_de_status(self) -> None:
        self.assertEqual(to_runpod_status("queued"), "IN_QUEUE")
        self.assertEqual(to_runpod_status("running"), "IN_PROGRESS")
        self.assertEqual(to_runpod_status("completed"), "COMPLETED")
        self.assertEqual(to_runpod_status("failed"), "FAILED")

    def test_output_replica_worker_runpod(self) -> None:
        record = make_record(
            status="completed",
            result=dict(FAKE_RESULT),
            params={
                "prompt": "ola",
                "negative_prompt": "",
                "duration_seconds": 12,
                "fps": 24,
                "audio_start_seconds": 0,
                "lora_strength": 0.7,
                "first_frame_strength": 1.0,
                "cfg": 1.0,
                "decode_tile_size": 512,
                "enable_prompt_enhance": False,
                "base_sigmas": None,
                "refine_sigmas": None,
                # campos que NÃO podem vazar para o output formato Runpod:
                "seed": 99,
                "width": 704,
                "image_strength": 1.0,
            },
        )
        output = runpod_output(record)
        self.assertEqual(
            set(output.keys()),
            {"job_id", "comfy_prompt_id", "video_url", "last_frame_url", "image_sha", "parameters", "execution_seconds"},
        )
        self.assertEqual(
            set(output["parameters"].keys()),
            {
                "prompt", "negative_prompt", "duration_seconds", "fps",
                "audio_start_seconds", "lora_strength", "first_frame_strength",
                "cfg", "decode_tile_size", "enable_prompt_enhance",
                "base_sigmas", "refine_sigmas",
            },
        )
        self.assertNotIn("project_id", output)
        self.assertNotIn("seed", output["parameters"])
        self.assertNotIn("width", output["parameters"])
        self.assertEqual(output["video_url"], FAKE_RESULT["video_url"])
        self.assertEqual(output["parameters"]["lora_strength"], 0.7)

    def test_output_none_antes_de_completar(self) -> None:
        self.assertIsNone(runpod_output(make_record(status="queued", result=None)))

    def test_webhook_body_formato_runpod(self) -> None:
        completed = make_record(status="completed", result=FAKE_RESULT)
        self.assertEqual(
            build_runpod_webhook(completed),
            {"id": "job-1", "status": "COMPLETED", "output": FAKE_RESULT, "error": None},
        )
        failed = make_record(status="failed", error="boom")
        self.assertEqual(
            build_runpod_webhook(failed, error="boom"),
            {"id": "job-1", "status": "FAILED", "output": None, "error": "boom"},
        )


class RunpodRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        api.manager = JobQueueManager()
        self.client = TestClient(api.app)
        STARTUP = api.STARTUP_DONE
        STARTUP.set()

    def test_run_aceita_envelope_runpod(self) -> None:
        captured: dict = {}
        with mock.patch.object(api, "run_job", return_value=dict(FAKE_RESULT)), mock.patch.object(
            api, "deliver_runpod", side_effect=lambda record, **kw: captured.update(record=record)
        ):
            response = self.client.post(
                "/meu-endpoint/run",
                json={"input": dict(RUNPOD_INPUT), "webhook": "https://cb.example/hook"},
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "IN_QUEUE")
            self.assertIn("id", body)

            # o dispatcher entrega o webhook de forma assíncrona
            deadline = time.time() + 10
            while "record" not in captured and time.time() < deadline:
                time.sleep(0.02)
        record = captured["record"]
        self.assertEqual(record.webhook_url, "https://cb.example/hook")
        self.assertEqual(record.params.get("_protocol"), "runpod")

    def test_runsync_retorna_completed_com_output(self) -> None:
        with mock.patch.object(api, "run_job", return_value=dict(FAKE_RESULT)), mock.patch.object(
            api, "deliver_runpod", return_value=None
        ):
            response = self.client.post(
                "/ep/runsync", json={"input": dict(RUNPOD_INPUT), "webhook": "https://cb.example/hook"}
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "COMPLETED")
        self.assertEqual(body["output"]["video_url"], FAKE_RESULT["video_url"])

    def test_status_completo_com_output_runpod(self) -> None:
        with mock.patch.object(api, "run_job", return_value=dict(FAKE_RESULT)), mock.patch.object(
            api, "deliver_runpod", return_value=None
        ):
            submit = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            job_id = submit.json()["id"]
            deadline = time.time() + 10
            payload = None
            while time.time() < deadline:
                response = self.client.get(f"/ep/status/{job_id}")
                if response.json().get("status") == "COMPLETED":
                    payload = response.json()
                    break
                time.sleep(0.02)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["status"], "COMPLETED")
        self.assertEqual(payload["output"]["comfy_prompt_id"], "prompt-123")
        self.assertNotIn("project_id", payload["output"])
        self.assertEqual(payload["output"]["parameters"]["lora_strength"], 0.7)

    def test_status_failed_carrega_erro(self) -> None:
        def boom(values, job_id):
            raise RuntimeError("ComfyUI recusou o workflow")

        with mock.patch.object(api, "run_job", side_effect=boom), mock.patch.object(
            api, "deliver_runpod", return_value=None
        ):
            submit = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            job_id = submit.json()["id"]
            deadline = time.time() + 10
            payload = None
            while time.time() < deadline:
                response = self.client.get(f"/ep/status/{job_id}")
                if response.json().get("status") == "FAILED":
                    payload = response.json()
                    break
                time.sleep(0.02)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["status"], "FAILED")
        self.assertIn("ComfyUI recusou", payload["error"])

    def test_run_sem_input_valido_responde_422(self) -> None:
        response = self.client.post("/ep/run", json={"input": {"project_id": "x"}})
        self.assertEqual(response.status_code, 422)

    def test_run_enfileira_segundo_job_e_executa_em_ordem(self) -> None:
        """Igual ao proxy RunPod: com a GPU ocupada o job entra na fila (IN_QUEUE)."""
        import threading

        release = threading.Event()
        order: list[str] = []

        def slow_first(values, job_id):
            order.append(job_id)
            release.wait(timeout=10)
            return dict(FAKE_RESULT)

        def fast(values, job_id):
            order.append(job_id)
            return dict(FAKE_RESULT)

        with mock.patch.object(api, "run_job", side_effect=slow_first), mock.patch.object(
            api, "deliver_runpod", return_value=None
        ):
            first = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.json()["status"], "IN_QUEUE")
            first_id = first.json()["id"]

            # Aguarda o 1º assumir a GPU antes de enfileirar o resto.
            deadline = time.time() + 10
            while time.time() < deadline and not api.manager.busy():
                time.sleep(0.02)

            second = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            third = self.client.post("/ep/run", json={"input": dict(RUNPOD_INPUT)})
            self.assertEqual(second.status_code, 200)
            self.assertEqual(third.status_code, 200)
            self.assertEqual(second.json()["status"], "IN_QUEUE")
            self.assertEqual(third.json()["status"], "IN_QUEUE")

            # 1º rodando; 2º e 3º na fila.
            self.assertEqual(api.manager.get(first_id).status, "running")
            self.assertEqual(api.manager.get(second.json()["id"]).status, "queued")
            health = self.client.get("/ep/health").json()["jobs"]
            self.assertEqual(health["in_progress"], 1)
            self.assertEqual(health["queued"], 2)

            release.set()  # libera o 1º; o dispatcher roda 2º e 3º em FIFO
            deadline = time.time() + 10
            while time.time() < deadline and len(order) < 3:
                time.sleep(0.02)

        self.assertEqual(len(order), 3)
        self.assertEqual(order[0], first_id)
        self.assertEqual(order[1], second.json()["id"])
        self.assertEqual(order[2], third.json()["id"])
        self.assertEqual(self.client.get("/ep/health").json()["jobs"]["queued"], 0)

    def test_status_desconhecido_responde_404(self) -> None:
        response = self.client.get("/ep/status/nao-existe")
        self.assertEqual(response.status_code, 404)

    def test_health_runpod(self) -> None:
        response = self.client.get("/ep/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()["jobs"].keys()), {"completed", "failed", "in_progress", "queued"}
        )

    def test_auth_bearer_aplica_nas_rotas_runpod(self) -> None:
        with mock.patch.dict(api.os.environ, {"API_TOKEN": "segredo"}):
            denied = self.client.get("/ep/status/qualquer")
            self.assertEqual(denied.status_code, 401)
            allowed = self.client.get(
                "/ep/health", headers={"Authorization": "Bearer segredo"}
            )
            self.assertEqual(allowed.status_code, 200)


if __name__ == "__main__":
    unittest.main()
