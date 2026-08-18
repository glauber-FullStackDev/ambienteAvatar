#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import decord


COMFYUI_HOME = Path("/opt/ComfyUI")
CUSTOM_NODES = COMFYUI_HOME / "custom_nodes"


def load_package(module_name: str, directory: Path):
    init_file = directory / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_file,
        submodule_search_locations=[str(directory)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create an import spec for {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def assert_pack(module_name: str, directory_name: str, expected_nodes: tuple[str, ...] = ()) -> None:
    module = load_package(module_name, CUSTOM_NODES / directory_name)
    mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mappings, dict) or not mappings:
        raise AssertionError(f"{directory_name} did not register any nodes")
    missing = [node for node in expected_nodes if node not in mappings]
    if missing:
        raise AssertionError(f"{directory_name} is missing nodes: {', '.join(missing)}")
    print(f"OK {directory_name}: {len(mappings)} nodes")


def assert_expression_controller() -> None:
    directory = CUSTOM_NODES / "ComfyUI-LongCat-Expression-Control"
    module = load_package("longcat_expression_control", directory)
    node = module.NODE_CLASS_MAPPINGS["LongCatAvatarMotionSmoother"]

    embedding = torch.ones((9, 5, 1280), dtype=torch.float32)
    payload = {
        "payload_type": "longcat_avatar_audio_full",
        "full_audio_emb": embedding,
        "audio_features": (embedding,),
        "num_segments": 1,
        "audio_stride": 1,
    }
    result = node.smooth(payload, strength=0.75, smoothing_frames=7)[0]
    assert result is not payload
    assert result["full_audio_emb"] is result["audio_features"][0]
    assert torch.equal(embedding, torch.ones_like(embedding))
    assert torch.allclose(result["full_audio_emb"], torch.full_like(embedding, 0.75))
    print("OK ComfyUI-LongCat-Expression-Control: transform contract")


def main() -> None:
    sys.path.insert(0, str(COMFYUI_HOME))
    print(f"OK decord import: {decord.__version__}")
    # GitHub-hosted Docker builders do not expose a GPU. This flag affects only
    # the import smoke test; the final image still starts ComfyUI in CUDA mode.
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    # Some packs register HTTP routes during import. ComfyUI normally creates
    # this object before loading custom nodes; the build smoke test provides the
    # smallest equivalent route registry.
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

    assert_expression_controller()
    assert_pack(
        "video_helper_suite",
        "ComfyUI-VideoHelperSuite",
    )
    assert_pack(
        "advanced_live_portrait",
        "ComfyUI-AdvancedLivePortrait",
    )
    assert_pack(
        "latent_sync_wrapper",
        "ComfyUI-LatentSyncWrapper",
        expected_nodes=("LatentSyncNode", "VideoLengthAdjuster"),
    )


if __name__ == "__main__":
    main()
