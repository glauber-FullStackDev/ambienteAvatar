#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import importlib
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
    required = {"LatentSyncNode", "LatentSyncStableNode", "VideoLengthAdjuster"}
    missing = sorted(required.difference(mappings))
    if missing:
        raise SystemExit(f"Nodes LatentSync ausentes: {', '.join(missing)}")

    stable_node = mappings["LatentSyncStableNode"]
    stable_inputs = stable_node.INPUT_TYPES()["required"]
    required_controls = {
        "stabilization_window",
        "stabilization_strength",
        "mask_feather",
        "motion_threshold",
        "motion_min_strength",
        "motion_blur_strength",
        "color_match_strength",
    }
    missing_controls = sorted(required_controls.difference(stable_inputs))
    if missing_controls:
        raise SystemExit(
            "Controles LatentSync Stable ausentes: " + ", ".join(missing_controls)
        )

    stable_runtime = importlib.import_module(
        "latent_sync_wrapper.latentsync_stable_runtime"
    )
    settings = stable_runtime.StableSettings(
        stabilization_strength=0.5,
        motion_protection=True,
    )
    import numpy as np

    matrices = [
        np.array([[1.0, 0.0, float(offset)], [0.0, 1.0, 0.0]])
        for offset in (0, 1, 12, 3, 4)
    ]
    smoothed = stable_runtime.smooth_affine_matrices(matrices, settings)
    if len(smoothed) != len(matrices) or smoothed[2][0, 2] >= 12:
        raise SystemExit("Estabilizacao temporal LatentSync Stable nao aplicada")
    strengths, scores = stable_runtime.motion_strengths(
        smoothed, 420, 560, settings
    )
    if len(strengths) != len(matrices) or np.any(strengths > 1.0):
        raise SystemExit("Protecao de movimento LatentSync Stable invalida")
    if not getattr(
        stable_runtime.LipsyncPipeline,
        "_infinitetalk_stable_installed",
        False,
    ):
        raise SystemExit("Runtime LatentSync Stable nao foi instalado")

    sys.path.insert(0, str(LATENTSYNC_NODE))
    inference_path = LATENTSYNC_NODE / "scripts/inference.py"
    inference_spec = importlib.util.spec_from_file_location(
        "latentsync_inference_smoke",
        inference_path,
    )
    if inference_spec is None or inference_spec.loader is None:
        raise RuntimeError("Nao foi possivel carregar o pipeline LatentSync")
    inference_module = importlib.util.module_from_spec(inference_spec)
    inference_spec.loader.exec_module(inference_module)

    import insightface
    import onnxruntime
    from latentsync.utils.face_detector import INSIGHTFACE_ROOT

    providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise SystemExit(
            "ONNX Runtime sem CUDAExecutionProvider: " + ", ".join(providers)
        )
    onnxruntime.preload_dlls()
    expected_insightface_root = "/opt/ComfyUI/models/latentsync/auxiliary"
    if INSIGHTFACE_ROOT != expected_insightface_root:
        raise SystemExit(
            f"Caminho InsightFace invalido: {INSIGHTFACE_ROOT}; "
            f"esperado: {expected_insightface_root}"
        )
    print(
        "OK: nodes e pipeline LatentSync registrados; "
        f"InsightFace {getattr(insightface, '__version__', 'instalado')}; "
        "ONNX Runtime CUDA"
    )


if __name__ == "__main__":
    main()
