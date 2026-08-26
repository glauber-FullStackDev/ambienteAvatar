#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import importlib
from pathlib import Path
import sys


COMFYUI_HOME = Path("/opt/ComfyUI")
LATENTSYNC_NODE = COMFYUI_HOME / "custom_nodes/ComfyUI-LatentSyncWrapper"
LIPFORCING_NODE = COMFYUI_HOME / "custom_nodes/ComfyUI-LipForcing"


def main() -> None:
    sys.path.insert(0, str(COMFYUI_HOME))
    sys.argv = [sys.argv[0], "--cpu"]

    import comfy.options

    comfy.options.enable_args_parsing()
    lipforcing_spec = importlib.util.spec_from_file_location(
        "comfyui_lipforcing",
        LIPFORCING_NODE / "__init__.py",
        submodule_search_locations=[str(LIPFORCING_NODE)],
    )
    if lipforcing_spec is None or lipforcing_spec.loader is None:
        raise RuntimeError("Nao foi possivel carregar ComfyUI-LipForcing")
    lipforcing_module = importlib.util.module_from_spec(lipforcing_spec)
    sys.modules[lipforcing_spec.name] = lipforcing_module
    lipforcing_spec.loader.exec_module(lipforcing_module)
    lipforcing_mappings = getattr(lipforcing_module, "NODE_CLASS_MAPPINGS", {})
    required_lipforcing = {"LipForcing14B"}
    missing_lipforcing = sorted(required_lipforcing.difference(lipforcing_mappings))
    if missing_lipforcing:
        raise SystemExit(
            "Nodes Lip Forcing ausentes: " + ", ".join(missing_lipforcing)
        )
    obsolete_lipforcing = {
        "LipForcingLoadVideo",
        "LipForcingLoadAudio",
    }.intersection(lipforcing_mappings)
    if obsolete_lipforcing:
        raise SystemExit(
            "Loaders Lip Forcing obsoletos ainda registrados: "
            + ", ".join(sorted(obsolete_lipforcing))
        )
    lipforcing_node = lipforcing_mappings["LipForcing14B"]
    lipforcing_inputs = lipforcing_node.INPUT_TYPES()["required"]
    if "video" not in lipforcing_inputs or "audio" not in lipforcing_inputs:
        raise SystemExit("Seletores diretos do Lip Forcing estao ausentes")
    if "video_path" in lipforcing_inputs or "audio_path" in lipforcing_inputs:
        raise SystemExit("Sockets de caminho obsoletos ainda estao registrados")
    if not callable(getattr(lipforcing_node, "VALIDATE_INPUTS", None)):
        raise SystemExit("Validacao dos arquivos diretos do Lip Forcing ausente")
    if not callable(getattr(lipforcing_node, "IS_CHANGED", None)):
        raise SystemExit("Deteccao de alteracao dos arquivos Lip Forcing ausente")
    if lipforcing_inputs["seed"][1].get("default") != 42:
        raise SystemExit("Seed padrao do Lip Forcing invalida")
    if lipforcing_inputs["decoder"][0][0] != "streaming_taehv":
        raise SystemExit("Decoder padrao do Lip Forcing invalido")
    lipforcing_nodes = importlib.import_module("comfyui_lipforcing.nodes")
    if lipforcing_nodes._exact_latent_frames(1.0) != 9:
        raise SystemExit("Calculo de duracao exata do Lip Forcing invalido")
    edge_latents = lipforcing_nodes._exact_latent_frames(0.88)
    if 1 + (edge_latents - 1) * 4 < 22:
        raise SystemExit("Lip Forcing pode terminar antes do audio")
    if lipforcing_nodes._safe_prefix("../saida teste") != "saida_teste":
        raise SystemExit("Sanitizacao do prefixo Lip Forcing invalida")

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
        "pose_protection",
        "max_head_yaw",
        "resume_head_yaw",
        "pose_guard_frames",
        "motion_blur_strength",
        "color_match_strength",
    }
    missing_controls = sorted(required_controls.difference(stable_inputs))
    if missing_controls:
        raise SystemExit(
            "Controles LatentSync Stable ausentes: " + ", ".join(missing_controls)
        )
    expected_defaults = {
        "lips_expression": 1.6,
        "inference_steps": 25,
        "stabilization_window": 1,
        "stabilization_strength": 0.0,
        "stabilize_translation": False,
        "stabilize_scale": False,
        "mask_expand": 5,
        "mask_feather": 8,
        "motion_protection": False,
        "motion_sensitivity": 1.3,
        "motion_smoothing": 1,
        "mouth_core_radius": 5,
    }
    for field, expected in expected_defaults.items():
        if stable_inputs[field][1].get("default") != expected:
            raise SystemExit(f"Default LatentSync Stable invalido: {field}")

    stable_runtime = importlib.import_module(
        "latent_sync_wrapper.latentsync_stable_runtime"
    )
    stable_video_io = importlib.import_module(
        "latent_sync_wrapper.latentsync_video_io"
    )
    from torchvision import io as torchvision_io

    if not getattr(torchvision_io.read_video, "_infinitetalk_resilient", False):
        raise SystemExit("Fallback FFmpeg do LatentSync nao foi instalado")
    if not stable_video_io._is_scaler_resource_error(
        BlockingIOError(11, "swscaler scaling graph resource temporarily unavailable")
    ):
        raise SystemExit("Falha de recurso do PyAV nao foi reconhecida")
    if not stable_video_io._is_scaler_resource_error(
        RuntimeError(
            "Failed to inject frame into filter network: "
            "Resource temporarily unavailable"
        )
    ):
        raise SystemExit("Falha de recurso do FFmpeg nao foi reconhecida")
    black_yuv420p = bytearray([16, 16, 16, 16, 128, 128])
    black_rgb = stable_video_io._yuv420p_to_rgb(black_yuv420p, 2, 2)
    if black_rgb.shape != (1, 2, 2, 3) or int(black_rgb.max()) != 0:
        raise SystemExit("Fallback YUV420p do LatentSync produziu RGB invalido")
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
    pose_settings = stable_runtime.StableSettings(
        pose_protection=True,
        max_head_yaw=25.0,
        resume_head_yaw=18.0,
        pose_guard_frames=0,
    )
    pose_fallbacks = stable_runtime.pose_fallback_mask(
        [0.0, 20.0, 26.0, 23.0, 17.0, -27.0],
        pose_settings,
    )
    expected_fallbacks = np.asarray(
        [False, False, True, True, False, True],
        dtype=np.bool_,
    )
    if not np.array_equal(pose_fallbacks, expected_fallbacks):
        raise SystemExit("Protecao de pose LatentSync Stable invalida")
    guarded = stable_runtime.pose_fallback_mask(
        [30.0],
        stable_runtime.StableSettings(pose_guard_frames=2),
    )
    if len(guarded) != 1 or not guarded[0]:
        raise SystemExit("Janela de guarda da protecao de pose invalida")
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
        "OK: nodes Lip Forcing e pipeline LatentSync registrados; "
        f"InsightFace {getattr(insightface, '__version__', 'instalado')}; "
        "ONNX Runtime CUDA"
    )


if __name__ == "__main__":
    main()
