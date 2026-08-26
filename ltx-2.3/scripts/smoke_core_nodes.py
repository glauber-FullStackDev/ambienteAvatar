#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace


COMFYUI_HOME = Path(os.environ.get("COMFYUI_HOME", "/opt/ComfyUI"))
REQUIRED_NODES = {
    "CheckpointLoaderSimple",
    "CreateVideo",
    "EmptyLTXVLatentVideo",
    "LTXAVTextEncoderLoader",
    "LTXVAudioVAEDecode",
    "LTXVAudioVAEEncode",
    "LTXVAudioVAELoader",
    "LTXVConcatAVLatent",
    "LTXVConditioning",
    "LTXVCropGuides",
    "LTXVDualCFGGuider",
    "LTXVEmptyLatentAudio",
    "LTXVImgToVideoInplace",
    "LTXIdentityOverlapConditioning",
    "LTXVLatentUpsampler",
    "LTXVPreprocess",
    "LTXVReferenceAudio",
    "LTXVSeparateAVLatent",
    "LatentUpscaleModelLoader",
    "ManualSigmas",
    "LoadAudio",
    "LoadImage",
    "SaveVideo",
    "SetLatentNoiseMask",
    "SolidMask",
    "TextGenerateLTX2Prompt",
    "TrimAudioDuration",
    "UNETLoader",
    "VAEDecodeTiled",
    "VAELoader",
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
            "Nodes exigidos pelo LTX 2.3/2.5 nao carregaram: "
            + ", ".join(sorted(missing))
        )
    print("Smoke test dos nodes do LTX 2.3/2.5 concluido.")


if __name__ == "__main__":
    main()
