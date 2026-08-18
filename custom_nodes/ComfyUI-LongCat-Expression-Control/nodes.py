from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def _smooth_embedding(embedding: torch.Tensor, strength: float, frames: int) -> torch.Tensor:
    """Smooth a LongCat [T, 5, 1280] audio embedding without mutating it."""
    if not torch.is_tensor(embedding):
        raise TypeError("LongCat audio embeddings must be PyTorch tensors.")
    if embedding.ndim != 3:
        raise ValueError(
            "LongCat audio embeddings must have shape [time, layers, width]; "
            f"received {tuple(embedding.shape)}."
        )

    original_dtype = embedding.dtype
    working = embedding
    if working.device.type == "cpu" and working.dtype in (torch.float16, torch.bfloat16):
        working = working.float()

    if frames > 1 and working.shape[0] > 1:
        window = min(int(frames), int(working.shape[0]))
        left = (window - 1) // 2
        right = window // 2
        channels_first = working.reshape(working.shape[0], -1).transpose(0, 1).unsqueeze(0)
        padded = F.pad(channels_first, (left, right), mode="replicate")
        working = F.avg_pool1d(padded, kernel_size=window, stride=1)
        working = working.squeeze(0).transpose(0, 1).reshape_as(embedding)

    return (working * float(strength)).to(dtype=original_dtype)


class LongCatAvatarMotionSmoother:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "au_cond": ("CONDITIONING",),
                "strength": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.5, "step": 0.05},
                ),
                "smoothing_frames": (
                    "INT",
                    {"default": 7, "min": 1, "max": 31, "step": 2},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("au_cond",)
    FUNCTION = "smooth"
    CATEGORY = "LongCat Avatar"
    DESCRIPTION = (
        "Reduz picos temporais do condicionamento de audio do LongCat e permite "
        "diminuir a intensidade dos movimentos faciais."
    )

    @classmethod
    def smooth(cls, au_cond, strength, smoothing_frames):
        if not isinstance(au_cond, Mapping):
            raise TypeError("au_cond must be a LongCat audio-conditioning mapping.")

        output = dict(au_cond)
        transformed_by_identity: dict[int, torch.Tensor] = {}

        def transform(value):
            if value is None:
                return None
            identity = id(value)
            if identity not in transformed_by_identity:
                transformed_by_identity[identity] = _smooth_embedding(
                    value,
                    strength=float(strength),
                    frames=int(smoothing_frames),
                )
            return transformed_by_identity[identity]

        embedding_keys = (
            "full_audio_emb",
            "left_full_audio_emb",
            "back_full_audio_emb",
        )
        found_embedding = False
        for key in embedding_keys:
            if key in output and output[key] is not None:
                output[key] = transform(output[key])
                found_embedding = True

        if "audio_features" in output:
            features = output["audio_features"]
            if not isinstance(features, (tuple, list)):
                raise TypeError("au_cond audio_features must be a tuple or list.")
            output["audio_features"] = tuple(transform(feature) for feature in features)
            found_embedding = found_embedding or bool(features)

        if not found_embedding:
            raise KeyError(
                "au_cond does not contain full_audio_emb or audio_features; "
                "connect LongCat Avatar Audio Encode to this node."
            )

        return (output,)


NODE_CLASS_MAPPINGS = {
    "LongCatAvatarMotionSmoother": LongCatAvatarMotionSmoother,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LongCatAvatarMotionSmoother": "LongCat Avatar Motion Smoother",
}
