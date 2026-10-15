#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace


COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
REQUIRED_NODES = {
    "CreateVideo",
    "DualCLIPLoaderGGUF",
    "EmptyLTXVLatentVideo",
    "GetImageSize",
    "GetVideoComponents",
    "LTXVAudioVAEDecode",
    "LTXVAudioVAEEncode",
    "LTXVConcatAVLatent",
    "LTXVConditioning",
    "LTXVCropGuides",
    "LTXVEmptyLatentAudio",
    "LTXVImgToVideoInplace",
    "LTXVLatentUpsampler",
    "LTXVPreprocess",
    "LTXVSeparateAVLatent",
    "LatentUpscaleModelLoader",
    "LastFrameFromBatch",
    "ManualSigmas",
    "LoadAudio",
    "LoadImage",
    "ResizeImagesByLongerEdge",
    "SaveVideo",
    "SolidMask",
    "TextGenerateLTX2Prompt",
    "TrimAudioDuration",
    "UnetLoaderGGUF",
    "VAEDecodeTiled",
    "VAELoaderKJ",
}


def main() -> None:
    os.chdir(COMFYUI_HOME)
    sys.path.insert(0, str(COMFYUI_HOME))
    # Docker builders do not expose a GPU. This only affects the import smoke
    # test; the final image starts ComfyUI in normal CUDA mode.
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import server

    if getattr(server.PromptServer, "instance", None) is None:
        prompt_queue = SimpleNamespace(currently_running={}, put=lambda _item: None)
        server.PromptServer.instance = SimpleNamespace(
            routes=server.web.RouteTableDef(),
            prompt_queue=prompt_queue,
            number=0,
            last_node_id=None,
            client_id=None,
            send_sync=lambda *_args, **_kwargs: None,
            # Custom nodes (ex.: KJNodes) tocam app.router.frozen/add_routes
            # durante o import; no smoke test nao existe servidor aiohttp.
            app=SimpleNamespace(
                router=SimpleNamespace(frozen=True),
                add_routes=lambda *_args, **_kwargs: None,
            ),
        )
    import nodes

    asyncio.run(
        nodes.init_extra_nodes(
            init_custom_nodes=True,
            init_api_nodes=True,
        )
    )
    missing = REQUIRED_NODES - set(nodes.NODE_CLASS_MAPPINGS)
    if missing:
        raise SystemExit(
            "Nodes exigidos pela variante GGUF 16 GB nao carregaram: "
            + ", ".join(sorted(missing))
        )
    print("Smoke test dos nodes GGUF do LTX 2.3 concluido.")


if __name__ == "__main__":
    main()
