"""ComfyUI node exposing the optional LatentSync stable runtime controls."""

from __future__ import annotations

from .nodes import LatentSyncNode
from .latentsync_stable_runtime import StableSettings, stable_context


class LatentSyncStableNode(LatentSyncNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "seed": ("INT", {"default": 1247}),
                "lips_expression": (
                    "FLOAT",
                    {"default": 1.8, "min": 1.0, "max": 3.0, "step": 0.05},
                ),
                "inference_steps": (
                    "INT",
                    {"default": 20, "min": 10, "max": 100, "step": 1},
                ),
                "stabilization_mode": (
                    ["median_gaussian", "median", "gaussian"],
                    {"default": "median_gaussian"},
                ),
                "stabilization_window": (
                    "INT",
                    {"default": 5, "min": 1, "max": 21, "step": 2},
                ),
                "stabilization_strength": (
                    "FLOAT",
                    {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "stabilize_translation": ("BOOLEAN", {"default": True}),
                "stabilize_rotation": ("BOOLEAN", {"default": False}),
                "stabilize_scale": ("BOOLEAN", {"default": True}),
                "max_translation_correction": (
                    "FLOAT",
                    {"default": 18.0, "min": 0.0, "max": 100.0, "step": 1.0},
                ),
                "max_rotation_correction": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.25},
                ),
                "max_scale_correction": (
                    "FLOAT",
                    {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01},
                ),
                "mask_expand": (
                    "INT",
                    {"default": 3, "min": -32, "max": 32, "step": 1},
                ),
                "mask_feather": (
                    "INT",
                    {"default": 8, "min": 0, "max": 64, "step": 1},
                ),
                "mask_opacity": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "motion_protection": ("BOOLEAN", {"default": True}),
                "motion_threshold": (
                    "FLOAT",
                    {"default": 0.020, "min": 0.002, "max": 0.5, "step": 0.001},
                ),
                "motion_sensitivity": (
                    "FLOAT",
                    {"default": 1.2, "min": 0.0, "max": 4.0, "step": 0.05},
                ),
                "motion_min_strength": (
                    "FLOAT",
                    {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "motion_smoothing": (
                    "INT",
                    {"default": 3, "min": 1, "max": 15, "step": 2},
                ),
                "pose_protection": ("BOOLEAN", {"default": True}),
                "max_head_yaw": (
                    "FLOAT",
                    {"default": 25.0, "min": 5.0, "max": 60.0, "step": 1.0},
                ),
                "resume_head_yaw": (
                    "FLOAT",
                    {"default": 18.0, "min": 0.0, "max": 55.0, "step": 1.0},
                ),
                "pose_guard_frames": (
                    "INT",
                    {"default": 2, "min": 0, "max": 12, "step": 1},
                ),
                "mouth_core_radius": (
                    "INT",
                    {"default": 6, "min": 0, "max": 64, "step": 1},
                ),
                "mouth_core_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "motion_blur_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "motion_blur_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.1},
                ),
                "color_match_strength": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "color_match_radius": (
                    "INT",
                    {"default": 12, "min": 1, "max": 48, "step": 1},
                ),
                "debug_log": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "LatentSyncNode/Stable"
    FUNCTION = "inference_stable"

    def inference_stable(
        self,
        images,
        audio,
        seed,
        lips_expression,
        inference_steps,
        stabilization_mode,
        stabilization_window,
        stabilization_strength,
        stabilize_translation,
        stabilize_rotation,
        stabilize_scale,
        max_translation_correction,
        max_rotation_correction,
        max_scale_correction,
        mask_expand,
        mask_feather,
        mask_opacity,
        motion_protection,
        motion_threshold,
        motion_sensitivity,
        motion_min_strength,
        motion_smoothing,
        pose_protection,
        max_head_yaw,
        resume_head_yaw,
        pose_guard_frames,
        mouth_core_radius,
        mouth_core_strength,
        motion_blur_strength,
        motion_blur_max,
        color_match_strength,
        color_match_radius,
        debug_log,
    ):
        settings = StableSettings(
            stabilization_mode=stabilization_mode,
            stabilization_window=stabilization_window,
            stabilization_strength=stabilization_strength,
            stabilize_translation=stabilize_translation,
            stabilize_rotation=stabilize_rotation,
            stabilize_scale=stabilize_scale,
            max_translation_correction=max_translation_correction,
            max_rotation_correction=max_rotation_correction,
            max_scale_correction=max_scale_correction,
            mask_expand=mask_expand,
            mask_feather=mask_feather,
            mask_opacity=mask_opacity,
            motion_protection=motion_protection,
            motion_threshold=motion_threshold,
            motion_sensitivity=motion_sensitivity,
            motion_min_strength=motion_min_strength,
            motion_smoothing=motion_smoothing,
            pose_protection=pose_protection,
            max_head_yaw=max_head_yaw,
            resume_head_yaw=resume_head_yaw,
            pose_guard_frames=pose_guard_frames,
            mouth_core_radius=mouth_core_radius,
            mouth_core_strength=mouth_core_strength,
            motion_blur_strength=motion_blur_strength,
            motion_blur_max=motion_blur_max,
            color_match_strength=color_match_strength,
            color_match_radius=color_match_radius,
            debug_log=debug_log,
        )
        with stable_context(settings):
            return super().inference(
                images,
                audio,
                seed,
                lips_expression=lips_expression,
                inference_steps=inference_steps,
            )
