#!/usr/bin/env python3
"""Example client for the LTX 2.3 IA2V Vast serverless endpoint.

Pattern: open a worker session, submit the job asynchronously, poll /status,
then close. The job also fires a webhook to the project backend on completion.
"""
import asyncio

from vastai import Serverless

ENDPOINT_NAME = "ltx23-ia2v-personal-lora-vast"


async def submit_and_wait(endpoint, *, project_id: str, image_url: str, audio_url: str, prompt: str, **params):
    # lifetime long enough for the render (minutes); polling renews the lease.
    async with await endpoint.session(cost=100, lifetime=120) as session:
        result = await session.request(
            "/submit",
            {
                "project_id": project_id,
                "image_url": image_url,
                "audio_url": audio_url,
                "prompt": prompt,
                "webhook": {"extra_params": {"origin": "sample-client"}},
                **params,
            },
            cost=100,
        )
        submit = result["response"]
        print("submitted:", submit)

        while True:
            polled = await session.request("/status", {"job_id": submit["job_id"]}, retry=False)
            state = polled["response"]
            print(f"[{submit['job_id']}] {state['status']}")
            if state["status"] in ("completed", "failed"):
                return state
            await asyncio.sleep(2)


async def main() -> None:
    async with Serverless() as client:
        endpoint = await client.get_endpoint(name=ENDPOINT_NAME)
        state = await submit_and_wait(
            endpoint,
            project_id="projeto-abc",
            image_url="https://minio.example.com/ltx-serverless/input/avatar.png?X-Amz-...",
            audio_url="https://minio.example.com/ltx-serverless/input/fala.wav?X-Amz-...",
            prompt="glauberavatar speaking naturally to camera",
            width=704,
            height=1280,
            duration_seconds=18,
            fps=24,
            lora_strength=1.0,
            image_strength=0.7,
        )
        if state["status"] == "completed":
            print("video_url:", state["result"]["video_url"])


if __name__ == "__main__":
    asyncio.run(main())