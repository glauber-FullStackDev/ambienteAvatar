#!/usr/bin/env python3
from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
import subprocess


required_modules = (
    "DeepCache",
    "accelerate",
    "decord",
    "diffusers",
    "einops",
    "face_alignment",
    "ffmpeg",
    "gguf",
    "huggingface_hub",
    "insightface",
    "mediapipe",
    "omegaconf",
    "onnx",
    "onnxruntime",
    "peft",
    "pyloudnorm",
    "pytorch_lightning",
    "rotary_embedding_torch",
    "safetensors",
    "sentencepiece",
    "soundfile",
    "torchaudio",
    "transformers",
    "cv2",
)

missing_modules = [name for name in required_modules if find_spec(name) is None]
if missing_modules:
    raise SystemExit(f"Dependencias Python ausentes: {', '.join(missing_modules)}")

ffmpeg_encoders = subprocess.run(
    ["ffmpeg", "-hide_banner", "-encoders"],
    check=True,
    capture_output=True,
    text=True,
).stdout
if "libx264" not in ffmpeg_encoders:
    raise SystemExit("FFmpeg sem encoder libx264 para a saida H.264 MP4")

comfyui_home = Path("/opt/ComfyUI")
required_paths = (
    comfyui_home / "main.py",
    comfyui_home / "custom_nodes/ComfyUI-WanVideoWrapper/__init__.py",
    comfyui_home / "custom_nodes/ComfyUI-KJNodes/__init__.py",
    comfyui_home / "custom_nodes/ComfyUI-MelBandRoFormer/__init__.py",
    comfyui_home / "custom_nodes/ComfyUI-VideoHelperSuite/__init__.py",
    comfyui_home / "custom_nodes/ComfyUI-LatentSyncWrapper/__init__.py",
    comfyui_home
    / "custom_nodes/ComfyUI-LatentSyncWrapper/latentsync_stable_node.py",
    comfyui_home
    / "custom_nodes/ComfyUI-LatentSyncWrapper/latentsync_stable_runtime.py",
    comfyui_home / "checkpoints/auxiliary",
    Path("/opt/defaults/workflows/infinitetalk-i2v.json"),
    Path("/opt/defaults/workflows/infinitetalk-v2v.json"),
    Path("/opt/defaults/workflows/infinitetalk-v2v-docker.json"),
    Path(
        "/opt/defaults/workflows/infinitetalk-v2v-latentsync16-docker.json"
    ),
    Path(
        "/opt/defaults/workflows/infinitetalk-v2v-latentsync16-stable-docker.json"
    ),
)
missing_paths = [str(path) for path in required_paths if not path.exists()]
if missing_paths:
    raise SystemExit(f"Arquivos da imagem ausentes: {', '.join(missing_paths)}")

base_workflow_path = Path("/opt/defaults/workflows/infinitetalk-v2v-docker.json")
latentsync_workflow_path = Path(
    "/opt/defaults/workflows/infinitetalk-v2v-latentsync16-docker.json"
)
stable_workflow_path = Path(
    "/opt/defaults/workflows/infinitetalk-v2v-latentsync16-stable-docker.json"
)
base_workflow = json.loads(base_workflow_path.read_text(encoding="utf-8"))
latentsync_workflow = json.loads(
    latentsync_workflow_path.read_text(encoding="utf-8")
)
stable_workflow = json.loads(stable_workflow_path.read_text(encoding="utf-8"))

if any(node.get("type") == "LatentSyncNode" for node in base_workflow["nodes"]):
    raise SystemExit("O workflow V2V base nao pode conter LatentSyncNode")


def workflow_prompts(workflow: dict) -> tuple[str, str]:
    node = next(
        node
        for node in workflow["nodes"]
        if node.get("type") == "WanVideoTextEncodeCached"
    )
    values = node.get("widgets_values", [])
    named = node.get("widgets_values_named", {})
    positive = named.get("positive_prompt", "")
    negative = named.get("negative_prompt", "")
    if len(values) < 4 or values[2] != positive or values[3] != negative:
        raise SystemExit("Representacoes de prompt do workflow estao divergentes")
    return positive, negative


base_positive, base_negative = workflow_prompts(base_workflow)
latentsync_positive, latentsync_negative = workflow_prompts(latentsync_workflow)
stable_positive, stable_negative = workflow_prompts(stable_workflow)
if base_positive or base_negative:
    raise SystemExit("O workflow InfiniteTalk V2V sem LatentSync deve manter prompts vazios")
if not latentsync_positive or not latentsync_negative:
    raise SystemExit("Prompts InfiniteTalk + LatentSync nao podem ficar vazios")
if (stable_positive, stable_negative) != (
    latentsync_positive,
    latentsync_negative,
):
    raise SystemExit("Os workflows LatentSync devem usar os mesmos prompts")

nodes = {node["id"]: node for node in latentsync_workflow["nodes"]}
base_nodes = {node["id"]: node for node in base_workflow["nodes"]}
links = {link[0]: link for link in latentsync_workflow["links"]}
expected_links = {
    545: [545, 130, 0, 307, 0, "IMAGE"],
    449: [449, 254, 0, 307, 1, "AUDIO"],
    559: [559, 307, 1, 131, 1, "AUDIO"],
    560: [560, 307, 0, 300, 0, "IMAGE"],
}
if nodes.get(307, {}).get("type") != "LatentSyncNode":
    raise SystemExit("LatentSyncNode ausente do novo workflow V2V")
for link_id, expected in expected_links.items():
    if links.get(link_id) != expected:
        raise SystemExit(f"Conexao LatentSync invalida no link {link_id}")

combine = nodes[131]
combine_values = combine.get("widgets_values", {})
if combine_values.get("filename_prefix") != "InfiniteTalk_V2V_LatentSync16":
    raise SystemExit("Prefixo de saida LatentSync invalido")
if combine_values.get("crf") != 16 or combine_values.get("frame_rate") != 25:
    raise SystemExit("O novo workflow deve preservar CRF 16 e 25 FPS")
for node_id, base_node in base_nodes.items():
    if node_id == 131:
        continue
    if nodes[node_id].get("widgets_values") != base_node.get("widgets_values"):
        raise SystemExit(f"Os parametros V2V foram alterados no node {node_id}")

base_combine_values = dict(base_nodes[131].get("widgets_values", {}))
new_combine_values = dict(combine_values)
for values in (base_combine_values, new_combine_values):
    values.pop("filename_prefix", None)
    values.pop("videopreview", None)
if new_combine_values != base_combine_values:
    raise SystemExit("Os parametros de codificacao do V2V foram alterados")

stable_nodes = {node["id"]: node for node in stable_workflow["nodes"]}
stable_links = {link[0]: link for link in stable_workflow["links"]}
if stable_nodes.get(307, {}).get("type") != "LatentSyncStableNode":
    raise SystemExit("LatentSyncStableNode ausente do workflow estabilizado")
for link_id, expected in expected_links.items():
    stable_expected = expected.copy()
    if stable_links.get(link_id) != stable_expected:
        raise SystemExit(f"Conexao LatentSync Stable invalida no link {link_id}")
stable_combine_values = stable_nodes[131].get("widgets_values", {})
if (
    stable_combine_values.get("filename_prefix")
    != "InfiniteTalk_V2V_LatentSync16_Stable"
):
    raise SystemExit("Prefixo de saida LatentSync Stable invalido")
if stable_combine_values.get("format") != "video/h264-mp4":
    raise SystemExit("LatentSync Stable deve usar H.264 MP4")
if (
    stable_combine_values.get("crf") != 16
    or stable_combine_values.get("frame_rate") != 25
):
    raise SystemExit("LatentSync Stable deve preservar CRF 16 e 25 FPS")

print("OK: dependencias e custom nodes do InfiniteTalk + LatentSync 1.6 presentes")
