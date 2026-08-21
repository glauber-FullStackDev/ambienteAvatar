#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


COMFYUI_HOME = Path("/opt/ComfyUI")
LATENTSYNC_NODE = COMFYUI_HOME / "custom_nodes/ComfyUI-LatentSyncWrapper"


def main() -> None:
    sys.path.insert(0, str(COMFYUI_HOME))
    sys.argv = [sys.argv[0], "--cpu"]

    import comfy.options

    comfy.options.enable_args_parsing()
    spec = importlib.util.spec_from_file_location(
        "latent_sync_wrapper",
        LATENTSYNC_NODE / "__init__.py",
        submodule_search_locations=[str(LATENTSYNC_NODE)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Nao foi possivel carregar ComfyUI-LatentSyncWrapper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
    required = {"LatentSyncNode", "VideoLengthAdjuster"}
    missing = sorted(required.difference(mappings))
    if missing:
        raise SystemExit(f"Nodes LatentSync ausentes: {', '.join(missing)}")
    print("OK: LatentSyncNode e VideoLengthAdjuster registrados")


if __name__ == "__main__":
    main()
