"""Optional temporal stabilization and compositing controls for LatentSync 1.6.

This module is copied into the pinned ComfyUI-LatentSyncWrapper package.  It
patches the pipeline once, but keeps the stock node byte-for-byte equivalent
unless ``stable_context`` is active.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
from typing import Iterator

import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as functional
from einops import rearrange

_WRAPPER_HOME = str(Path(__file__).resolve().parent)
if _WRAPPER_HOME not in sys.path:
    sys.path.insert(0, _WRAPPER_HOME)

# scripts/inference.py imports this package as top-level ``latentsync``.  Use
# the same module identity here so the runtime patch reaches that pipeline.
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline


@dataclass(frozen=True)
class StableSettings:
    stabilization_mode: str = "median_gaussian"
    stabilization_window: int = 5
    stabilization_strength: float = 0.35
    stabilize_translation: bool = True
    stabilize_rotation: bool = True
    stabilize_scale: bool = True
    max_translation_correction: float = 18.0
    max_rotation_correction: float = 3.0
    max_scale_correction: float = 0.06
    mask_expand: int = 2
    mask_feather: int = 16
    mask_opacity: float = 1.0
    motion_protection: bool = True
    motion_threshold: float = 0.025
    motion_sensitivity: float = 1.0
    motion_min_strength: float = 0.55
    motion_smoothing: int = 3
    mouth_core_radius: int = 14
    mouth_core_strength: float = 0.85
    motion_blur_strength: float = 0.35
    motion_blur_max: float = 1.2
    color_match_strength: float = 0.15
    color_match_radius: int = 12
    debug_log: bool = False


@dataclass
class StableState:
    settings: StableSettings
    raw_matrices: list[object] = field(default_factory=list)
    motion_strengths: np.ndarray = field(
        default_factory=lambda: np.ones(0, dtype=np.float32)
    )
    motion_scores: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    blend_cursor: int = 0
    logged_blend: bool = False


_ACTIVE_STATE: ContextVar[StableState | None] = ContextVar(
    "latentsync_stable_state", default=None
)


@contextmanager
def stable_context(settings: StableSettings) -> Iterator[StableState]:
    state = StableState(settings=settings)
    token = _ACTIVE_STATE.set(state)
    try:
        yield state
    finally:
        _ACTIVE_STATE.reset(token)


def _odd_window(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _matrix_numpy(matrix: object) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        result = matrix.detach().float().cpu().numpy()
    else:
        result = np.asarray(matrix, dtype=np.float32)
    if result.ndim == 3:
        result = result[0]
    if result.shape != (2, 3):
        raise ValueError(f"Matriz afim inesperada: {result.shape}")
    return result.astype(np.float64, copy=False)


def _decompose_matrices(matrices: list[object]) -> np.ndarray:
    parameters = []
    for matrix in matrices:
        value = _matrix_numpy(matrix)
        linear = value[:, :2]
        determinant = max(float(np.linalg.det(linear)), 1e-12)
        scale = math.sqrt(determinant)
        angle = math.atan2(float(linear[1, 0]), float(linear[0, 0]))
        parameters.append(
            [float(value[0, 2]), float(value[1, 2]), math.log(scale), angle]
        )
    result = np.asarray(parameters, dtype=np.float64)
    if len(result):
        result[:, 3] = np.unwrap(result[:, 3])
    return result


def _compose_matrices(parameters: np.ndarray) -> list[np.ndarray]:
    matrices = []
    for tx, ty, log_scale, angle in parameters:
        scale = math.exp(float(log_scale))
        cosine = math.cos(float(angle)) * scale
        sine = math.sin(float(angle)) * scale
        matrices.append(
            np.asarray(
                [[cosine, -sine, tx], [sine, cosine, ty]], dtype=np.float32
            )
        )
    return matrices


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack(
        [np.median(padded[index : index + window], axis=0) for index in range(len(values))]
    )


def _rolling_gaussian(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values.copy()
    radius = window // 2
    sigma = max(window / 4.0, 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-(offsets**2) / (2.0 * sigma**2))
    weights /= weights.sum()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack(
        [
            np.sum(padded[index : index + window] * weights[:, None], axis=0)
            for index in range(len(values))
        ]
    )


def smooth_affine_matrices(
    matrices: list[object], settings: StableSettings
) -> list[np.ndarray]:
    if len(matrices) < 2 or settings.stabilization_strength <= 0:
        return [_matrix_numpy(matrix).astype(np.float32) for matrix in matrices]

    original = _decompose_matrices(matrices)
    window = _odd_window(settings.stabilization_window)
    if settings.stabilization_mode == "median":
        target = _rolling_median(original, window)
    elif settings.stabilization_mode == "gaussian":
        target = _rolling_gaussian(original, window)
    else:
        target = _rolling_gaussian(_rolling_median(original, window), window)

    correction = (target - original) * settings.stabilization_strength
    if not settings.stabilize_translation:
        correction[:, 0:2] = 0
    else:
        translation_length = np.linalg.norm(correction[:, 0:2], axis=1)
        limit = max(settings.max_translation_correction, 0.0)
        scale = np.minimum(1.0, limit / np.maximum(translation_length, 1e-9))
        correction[:, 0:2] *= scale[:, None]

    if not settings.stabilize_scale:
        correction[:, 2] = 0
    else:
        correction[:, 2] = np.clip(
            correction[:, 2],
            -settings.max_scale_correction,
            settings.max_scale_correction,
        )

    if not settings.stabilize_rotation:
        correction[:, 3] = 0
    else:
        rotation_limit = math.radians(max(settings.max_rotation_correction, 0.0))
        correction[:, 3] = np.clip(
            correction[:, 3], -rotation_limit, rotation_limit
        )
    return _compose_matrices(original + correction)


def _warp_faces(
    pipeline: LipsyncPipeline,
    video_frames: np.ndarray,
    matrices: list[np.ndarray],
) -> torch.Tensor:
    restorer = pipeline.image_processor.restorer
    device = restorer.device
    dtype = restorer.dtype
    height, width = restorer.face_size[1], restorer.face_size[0]
    faces = []
    for frame, matrix in zip(video_frames, matrices):
        image = rearrange(
            torch.from_numpy(frame).to(device=device, dtype=dtype),
            "h w c -> 1 c h w",
        )
        transform = torch.from_numpy(matrix).to(device=device, dtype=dtype).unsqueeze(0)
        face = kornia.geometry.transform.warp_affine(
            image,
            transform,
            (height, width),
            mode="bilinear",
            padding_mode="fill",
            fill_value=restorer.fill_value,
        )
        face = rearrange(face.squeeze(0), "c h w -> h w c")
        face = face.clamp(0, 255).byte().cpu().numpy()
        face = cv2.resize(
            face,
            (pipeline.image_processor.resolution, pipeline.image_processor.resolution),
            interpolation=cv2.INTER_LANCZOS4,
        )
        faces.append(rearrange(torch.from_numpy(face), "h w c -> c h w"))
    return torch.stack(faces)


def _source_face_geometry(
    matrices: list[object], face_width: float, face_height: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = []
    scales = []
    angles = []
    aligned_center = np.asarray([face_width / 2.0, face_height / 2.0, 1.0])
    for matrix in matrices:
        value = _matrix_numpy(matrix)
        homogeneous = np.vstack([value, [0.0, 0.0, 1.0]])
        inverse = np.linalg.inv(homogeneous)
        centers.append((inverse @ aligned_center)[:2])
        source_x = np.linalg.norm(inverse[:2, 0]) * face_width
        source_y = np.linalg.norm(inverse[:2, 1]) * face_height
        scales.append(max((source_x + source_y) / 2.0, 1.0))
        angles.append(math.atan2(float(value[1, 0]), float(value[0, 0])))
    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(scales, dtype=np.float64),
        np.unwrap(np.asarray(angles, dtype=np.float64)),
    )


def motion_strengths(
    matrices: list[object],
    face_width: float,
    face_height: float,
    settings: StableSettings,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(matrices)
    if count == 0:
        return np.ones(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    if not settings.motion_protection:
        return np.ones(count, dtype=np.float32), np.zeros(count, dtype=np.float32)

    centers, scales, angles = _source_face_geometry(
        matrices, face_width, face_height
    )
    center_delta = np.linalg.norm(np.diff(centers, axis=0, prepend=centers[:1]), axis=1)
    reference_scale = np.maximum(
        (scales + np.roll(scales, 1)) / 2.0,
        1.0,
    )
    reference_scale[0] = scales[0]
    translation_score = center_delta / reference_scale
    rotation_score = np.abs(
        np.diff(angles, axis=0, prepend=angles[:1])
    ) / math.radians(45.0)
    scale_score = np.abs(
        np.diff(np.log(scales), axis=0, prepend=np.log(scales[:1]))
    )
    score = (
        translation_score + rotation_score + scale_score
    ) * settings.motion_sensitivity

    threshold = max(settings.motion_threshold, 1e-6)
    normalized = np.clip((score - threshold) / threshold, 0.0, 1.0)
    smooth_step = normalized * normalized * (3.0 - 2.0 * normalized)
    strengths = 1.0 - smooth_step * (1.0 - settings.motion_min_strength)

    smoothing = _odd_window(settings.motion_smoothing)
    if smoothing > 1:
        strengths = _rolling_gaussian(strengths[:, None], smoothing)[:, 0]
        score = _rolling_gaussian(score[:, None], smoothing)[:, 0]
    return strengths.astype(np.float32), score.astype(np.float32)


def _morph(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels == 0:
        return mask
    kernel = _odd_window(abs(pixels) * 2 + 1)
    if pixels > 0:
        return functional.max_pool2d(mask, kernel, stride=1, padding=kernel // 2)
    return -functional.max_pool2d(-mask, kernel, stride=1, padding=kernel // 2)


def _feather(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask
    kernel = _odd_window(pixels * 2 + 1)
    sigma = max(pixels / 2.0, 0.5)
    return kornia.filters.gaussian_blur2d(mask, (kernel, kernel), (sigma, sigma))


def _color_match(
    generated: torch.Tensor,
    source: torch.Tensor,
    alpha: torch.Tensor,
    strength: float,
    radius: int,
) -> torch.Tensor:
    if strength <= 0:
        return generated
    outer = _morph(alpha, max(radius, 1))
    inner = _morph(alpha, -max(radius // 2, 1))
    weights = (outer - inner).clamp(0, 1)
    weights_sum = weights.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    generated_mean = (generated * weights).sum(dim=(2, 3), keepdim=True) / weights_sum
    source_mean = (source * weights).sum(dim=(2, 3), keepdim=True) / weights_sum
    generated_variance = (
        ((generated - generated_mean) ** 2 * weights).sum(dim=(2, 3), keepdim=True)
        / weights_sum
    )
    source_variance = (
        ((source - source_mean) ** 2 * weights).sum(dim=(2, 3), keepdim=True)
        / weights_sum
    )
    ratio = torch.sqrt(
        (source_variance + 1e-6) / (generated_variance + 1e-6)
    ).clamp(0.8, 1.25)
    matched = (generated - generated_mean) * ratio + source_mean
    return torch.lerp(generated, matched.clamp(-1, 1), strength)


def _motion_blur(
    generated: torch.Tensor,
    motion_strength: torch.Tensor,
    strength: float,
    maximum_sigma: float,
) -> torch.Tensor:
    if strength <= 0 or maximum_sigma <= 0:
        return generated
    output = []
    for index in range(len(generated)):
        amount = float((1.0 - motion_strength[index]).item()) * strength
        sigma = amount * maximum_sigma
        if sigma < 0.08:
            output.append(generated[index : index + 1])
            continue
        kernel = _odd_window(max(1, int(math.ceil(sigma * 3.0))))
        output.append(
            kornia.filters.gaussian_blur2d(
                generated[index : index + 1],
                (kernel, kernel),
                (max(sigma, 0.08), max(sigma, 0.08)),
            )
        )
    return torch.cat(output, dim=0)


_ORIGINAL_AFFINE_TRANSFORM_VIDEO = LipsyncPipeline.affine_transform_video
_ORIGINAL_LOOP_VIDEO = LipsyncPipeline.loop_video
_ORIGINAL_PASTE_SURROUNDING_PIXELS_BACK = (
    LipsyncPipeline.paste_surrounding_pixels_back
)


def _stable_affine_transform_video(
    pipeline: LipsyncPipeline, video_frames: np.ndarray
):
    faces, boxes, matrices = _ORIGINAL_AFFINE_TRANSFORM_VIDEO(
        pipeline, video_frames
    )
    state = _ACTIVE_STATE.get()
    if state is None:
        return faces, boxes, matrices
    state.raw_matrices = list(matrices)
    if state.settings.stabilization_strength <= 0:
        return faces, boxes, matrices
    stabilized = smooth_affine_matrices(matrices, state.settings)
    faces = _warp_faces(pipeline, video_frames, stabilized)
    if state.settings.debug_log:
        print(
            "LatentSync Stable: transformacoes estabilizadas "
            f"(quadros={len(stabilized)}, "
            f"janela={_odd_window(state.settings.stabilization_window)}, "
            f"forca={state.settings.stabilization_strength:.2f})"
        )
    return faces, boxes, stabilized


def _stable_loop_video(
    pipeline: LipsyncPipeline, whisper_chunks: list, video_frames: np.ndarray
):
    result = _ORIGINAL_LOOP_VIDEO(pipeline, whisper_chunks, video_frames)
    state = _ACTIVE_STATE.get()
    if state is not None:
        _frames, _faces, _boxes, matrices = result
        restorer = pipeline.image_processor.restorer
        motion_matrices = state.raw_matrices
        if motion_matrices and len(motion_matrices) != len(matrices):
            expanded = []
            cycle = 0
            while len(expanded) < len(matrices):
                expanded.extend(
                    motion_matrices if cycle % 2 == 0 else motion_matrices[::-1]
                )
                cycle += 1
            motion_matrices = expanded[: len(matrices)]
        if not motion_matrices:
            motion_matrices = matrices
        strengths, scores = motion_strengths(
            motion_matrices,
            restorer.face_size[0],
            restorer.face_size[1],
            state.settings,
        )
        state.motion_strengths = strengths
        state.motion_scores = scores
        state.blend_cursor = 0
        if state.settings.debug_log and len(scores):
            print(
                "LatentSync Stable: movimento "
                f"max={scores.max():.4f}, medio={scores.mean():.4f}, "
                f"blend_min={strengths.min():.3f}"
            )
    return result


def _stable_paste_surrounding_pixels_back(
    decoded_latents: torch.Tensor,
    pixel_values: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    weight_dtype: torch.dtype,
):
    state = _ACTIVE_STATE.get()
    if state is None:
        return _ORIGINAL_PASTE_SURROUNDING_PIXELS_BACK(
            decoded_latents, pixel_values, masks, device, weight_dtype
        )

    settings = state.settings
    source = pixel_values.to(device=device, dtype=weight_dtype)
    alpha = masks.to(device=device, dtype=weight_dtype).clamp(0, 1)
    alpha = _morph(alpha, settings.mask_expand)
    alpha = _feather(alpha, settings.mask_feather).clamp(0, 1)
    alpha = (alpha * settings.mask_opacity).clamp(0, 1)

    frame_count = decoded_latents.shape[0]
    start = state.blend_cursor
    stop = start + frame_count
    values = state.motion_strengths[start:stop]
    if len(values) != frame_count:
        values = np.ones(frame_count, dtype=np.float32)
    state.blend_cursor = stop
    motion = torch.as_tensor(
        values, device=device, dtype=weight_dtype
    ).view(-1, 1, 1, 1)

    core = _morph(alpha, -settings.mouth_core_radius)
    spatial_motion = motion + (1.0 - motion) * core * settings.mouth_core_strength
    alpha = (alpha * spatial_motion).clamp(0, 1)

    generated = _motion_blur(
        decoded_latents,
        motion,
        settings.motion_blur_strength,
        settings.motion_blur_max,
    )
    generated = _color_match(
        generated,
        source,
        alpha,
        settings.color_match_strength,
        settings.color_match_radius,
    )
    if settings.debug_log and not state.logged_blend:
        print(
            "LatentSync Stable: composicao "
            f"expand={settings.mask_expand}px, feather={settings.mask_feather}px, "
            f"opacity={settings.mask_opacity:.2f}, color={settings.color_match_strength:.2f}"
        )
        state.logged_blend = True
    return generated * alpha + source * (1.0 - alpha)


def install_stable_runtime() -> None:
    if getattr(LipsyncPipeline, "_infinitetalk_stable_installed", False):
        return
    LipsyncPipeline.affine_transform_video = _stable_affine_transform_video
    LipsyncPipeline.loop_video = _stable_loop_video
    LipsyncPipeline.paste_surrounding_pixels_back = staticmethod(
        _stable_paste_surrounding_pixels_back
    )
    LipsyncPipeline._infinitetalk_stable_installed = True


install_stable_runtime()
