#!/usr/bin/env python3
"""Example client for the LTX 2.3 IA2V Vast serverless endpoint.

Pattern (session-less): with max_workers=1 the endpoint has a single worker,
so plain routed requests always hit it. /submit returns 202 immediately, the
render runs in background, and the project backend receives the result via
webhook. /status is a fallback for polling.
"""
import asyncio

from vastai import Serverless

ENDPOINT_NAME = "ltx23-ia2v-personal-lora-vast"


async def submit_and_wait(endpoint, *, project_id: str, image_url: str, audio_url: str, prompt: str, **params):
    result = await endpoint.request(
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
        polled = await endpoint.request("/status", {"job_id": submit["job_id"]}, retry=False)
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
            image_url="https://s3.fluxaassist.io/ltx-serverless/input/avatar.png?X-Amz-...",
            audio_url="https://s3.fluxaassist.io/ltx-serverless/input/fala.wav?X-Amz-...",
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