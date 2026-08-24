"""Resilient video decoding for the pinned LatentSync ComfyUI wrapper."""

from __future__ import annotations

from fractions import Fraction
import json
import subprocess
from typing import Callable

import torch
from torchvision import io


def _video_metadata(filename: str) -> tuple[int, int, float, str]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,pix_fmt",
        "-of",
        "json",
        filename,
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"FFprobe nao encontrou video em {filename}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    rate = str(stream.get("avg_frame_rate") or "0/1")
    fps = float(Fraction(rate)) if rate != "0/0" else 0.0
    return width, height, fps, str(stream.get("pix_fmt") or "")


def _ffmpeg_error(error: subprocess.CalledProcessError) -> str:
    stderr = error.stderr or b""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    detail = str(stderr).strip()
    return detail or str(error)


def _decode_raw(command: list[str]) -> bytearray:
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(_ffmpeg_error(error)) from error
    return bytearray(result.stdout)


def _yuv420p_to_rgb(raw: bytearray, width: int, height: int) -> torch.Tensor:
    """Convert limited-range BT.601 YUV420p frames without FFmpeg swscale."""
    if width % 2 or height % 2:
        raise ValueError("YUV420p exige largura e altura pares")
    pixels = width * height
    chroma_pixels = pixels // 4
    frame_size = pixels + 2 * chroma_pixels
    if not raw or len(raw) % frame_size:
        raise RuntimeError(
            "FFmpeg retornou video YUV420p incompleto: "
            f"{len(raw)} bytes para frames de {frame_size} bytes"
        )

    frame_count = len(raw) // frame_size
    frames = torch.empty((frame_count, height, width, 3), dtype=torch.uint8)
    for index in range(frame_count):
        packed = torch.frombuffer(
            raw,
            dtype=torch.uint8,
            count=frame_size,
            offset=index * frame_size,
        )
        y = packed[:pixels].reshape(height, width).to(torch.int32)
        u = packed[pixels : pixels + chroma_pixels].reshape(
            height // 2, width // 2
        )
        v = packed[pixels + chroma_pixels :].reshape(
            height // 2, width // 2
        )
        u = u.repeat_interleave(2, 0).repeat_interleave(2, 1).to(torch.int32)
        v = v.repeat_interleave(2, 0).repeat_interleave(2, 1).to(torch.int32)

        c = (y - 16).clamp_min(0)
        d = u - 128
        e = v - 128
        red = (298 * c + 409 * e + 128) >> 8
        green = (298 * c - 100 * d - 208 * e + 128) >> 8
        blue = (298 * c + 516 * d + 128) >> 8
        frames[index] = torch.stack((red, green, blue), dim=-1).clamp(
            0, 255
        ).to(torch.uint8)
    return frames


def read_video_with_ffmpeg(
    filename: str,
    start_pts: float = 0,
    end_pts: float | None = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
):
    """Decode RGB frames with a single-threaded FFmpeg subprocess.

    LatentSync calls ``read_video`` with seconds and only consumes the video
    tensor. The return shape remains compatible with torchvision so the
    wrapper does not need to change.
    """
    if pts_unit not in {"pts", "sec"}:
        raise ValueError(f"pts_unit nao suportado: {pts_unit}")
    if pts_unit == "pts" and (start_pts or end_pts is not None):
        raise ValueError("Recorte por PTS bruto nao e suportado pelo fallback FFmpeg")

    width, height, fps, source_pixel_format = _video_metadata(filename)
    base_command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
    ]
    if pts_unit == "sec" and start_pts:
        base_command.extend(["-ss", str(float(start_pts))])
    base_command.extend(["-i", filename])
    if pts_unit == "sec" and end_pts is not None:
        duration = max(float(end_pts) - float(start_pts), 0.0)
        base_command.extend(["-t", str(duration)])
    raw_output = [
        "-map",
        "0:v:0",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
    ]
    rgb_command = [*base_command, *raw_output, "-pix_fmt", "rgb24", "pipe:1"]
    try:
        raw = _decode_raw(rgb_command)
    except RuntimeError:
        if source_pixel_format != "yuv420p":
            raise
        print(
            "LatentSync: FFmpeg sem recurso no swscale; "
            "decodificando YUV420p sem conversao de pixel no subprocesso"
        )
        yuv_command = [
            *base_command,
            *raw_output,
            "-pix_fmt",
            "yuv420p",
            "pipe:1",
        ]
        frames = _yuv420p_to_rgb(_decode_raw(yuv_command), width, height)
        if output_format == "TCHW":
            frames = frames.permute(0, 3, 1, 2)
        elif output_format != "THWC":
            raise ValueError(f"output_format nao suportado: {output_format}")
        audio = torch.empty((1, 0), dtype=torch.float32)
        return frames, audio, {"video_fps": fps, "audio_fps": 0}

    frame_size = width * height * 3
    if not raw or len(raw) % frame_size:
        raise RuntimeError(
            "FFmpeg retornou video RGB incompleto: "
            f"{len(raw)} bytes para frames de {frame_size} bytes"
        )
    frame_count = len(raw) // frame_size
    frames = torch.frombuffer(raw, dtype=torch.uint8).reshape(
        frame_count, height, width, 3
    )
    if output_format == "TCHW":
        frames = frames.permute(0, 3, 1, 2)
    elif output_format != "THWC":
        raise ValueError(f"output_format nao suportado: {output_format}")
    audio = torch.empty((1, 0), dtype=torch.float32)
    return frames, audio, {"video_fps": fps, "audio_fps": 0}


def _is_scaler_resource_error(error: Exception) -> bool:
    message = str(error).lower()
    return getattr(error, "errno", None) == 11 or (
        "resource temporarily unavailable" in message
        and any(
            marker in message
            for marker in (
                "swscaler",
                "scaling graph",
                "auto_scale",
                "filter network",
                "output pad",
            )
        )
    )


def install_resilient_read_video() -> None:
    """Retry torchvision/PyAV decoder failures through FFmpeg once."""
    current = io.read_video
    if getattr(current, "_infinitetalk_resilient", False):
        return

    original: Callable = current

    def resilient_read_video(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as error:
            if not _is_scaler_resource_error(error):
                raise
            filename = args[0] if args else kwargs.get("filename")
            if not filename:
                raise
            print(
                "LatentSync: PyAV sem recurso para converter o video; "
                "repetindo a leitura com FFmpeg em uma thread"
            )
            return read_video_with_ffmpeg(
                filename,
                start_pts=kwargs.get("start_pts", 0),
                end_pts=kwargs.get("end_pts"),
                pts_unit=kwargs.get("pts_unit", "pts"),
                output_format=kwargs.get("output_format", "THWC"),
            )

    resilient_read_video._infinitetalk_resilient = True
    resilient_read_video._infinitetalk_original = original
    io.read_video = resilient_read_video
