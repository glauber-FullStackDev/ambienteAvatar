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
ffmpeg_help_result = subprocess.run(
    ["ffmpeg", "-hide_banner", "-h", "full"],
    check=True,
    capture_output=True,
    text=True,
)
ffmpeg_help = ffmpeg_help_result.stdout + ffmpeg_help_result.stderr
if "-vsync" not in ffmpeg_help:
    raise SystemExit("FFmpeg sem a opcao compativel -vsync do fallback LatentSync")
subprocess.run(
    ["ffprobe", "-v", "error", "-version"],
    check=True,
    capture_output=True,
    text=True,
)

comfyui_home = Path("/opt/ComfyUI")
required_paths = (
    comfyui_home / "main.py",
    comfyui_home / "custom_nodes/ComfyUI-WanVideoWrapper/__init__.py",
    comfyui_home / "custom_nodes/ComfyUI-KJNodes/__init__.py",
    comfyui_home / "custom_nodes/ComfyUI-MelBandRoFormer/__init__.py",
    comfyui_home / "custom_nodes/ComfyUI-VideoHelperSuite/__init__.py",
    comfyui_home / "custom_nodes/ComfyUI-LatentSyncWrapper/__init__.py",
    comfyui_home / "custom_nodes/ComfyUI-LipForcing/__init__.py",
    comfyui_home
    / "custom_nodes/ComfyUI-LatentSyncWrapper/latentsync_stable_node.py",
    comfyui_home
    / "custom_nodes/ComfyUI-LatentSyncWrapper/latentsync_stable_runtime.py",
    comfyui_home
    / "custom_nodes/ComfyUI-LatentSyncWrapper/latentsync_video_io.py",
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
    Path(
        "/opt/defaults/workflows/infinitetalk-v2v-stable-no-latentsync-docker.json"
    ),
    Path("/opt/defaults/workflows/lipforcing14b-video-audio-docker.json"),
    Path("/opt/LipForcing/scripts/inference/inference_streaming.py"),
    Path("/opt/LipForcing/scripts/inference/inference_segmentwise.py"),
    Path("/opt/lipforcing-venv/bin/python"),
    Path("/opt/infinitetalk-scripts/download_lipforcing_models.py"),
    Path("/opt/infinitetalk-scripts/precompute_lipforcing_text.py"),
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
stable_no_latentsync_workflow_path = Path(
    "/opt/defaults/workflows/infinitetalk-v2v-stable-no-latentsync-docker.json"
)
lipforcing_workflow_path = Path(
    "/opt/defaults/workflows/lipforcing14b-video-audio-docker.json"
)
base_workflow = json.loads(base_workflow_path.read_text(encoding="utf-8"))
latentsync_workflow = json.loads(
    latentsync_workflow_path.read_text(encoding="utf-8")
)
stable_workflow = json.loads(stable_workflow_path.read_text(encoding="utf-8"))
stable_no_latentsync_workflow = json.loads(
    stable_no_latentsync_workflow_path.read_text(encoding="utf-8")
)
lipforcing_workflow = json.loads(
    lipforcing_workflow_path.read_text(encoding="utf-8")
)

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
if not stable_positive or not stable_negative:
    raise SystemExit("Prompts do preset LatentSync Stable nao podem ficar vazios")

nodes = {node["id"]: node for node in latentsync_workflow["nodes"]}
base_nodes = {node["id"]: node for node in base_workflow["nodes"]}
links = {link[0]: link for link in latentsync_workflow["links"]}
expected_links = {
    545: [545, 130, 0, 307, 0, "IMAGE"],
    449: [449, 254, 0, 307, 1, "AUDIO"],
    559: [559, 254, 0, 131, 1, "AUDIO"],
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
    current_values = nodes[node_id].get("widgets_values")
    base_values = base_node.get("widgets_values")
    if node_id == 241:
        current_values = list(current_values)
        base_values = list(base_values)
        current_values[2:4] = ["", ""]
        base_values[2:4] = ["", ""]
    if current_values != base_values:
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
stable_node = stable_nodes[307]
if stable_node.get("properties", {}).get("infinitetalk_stable_schema") != 4:
    raise SystemExit("Schema do LatentSync Stable nao foi atualizado")
expected_stable_widgets = [
    1247,
    "fixed",
    1.6,
    25,
    "median_gaussian",
    1,
    0,
    False,
    False,
    False,
    18,
    3,
    0.06,
    5,
    8,
    1,
    False,
    0.02,
    1.3,
    1,
    1,
    True,
    25,
    18,
    2,
    5,
    1,
    0,
    1,
    0.15,
    12,
    False,
]
if stable_node.get("widgets_values") != expected_stable_widgets:
    raise SystemExit("Defaults finais do LatentSync Stable estao incorretos")
stable_sampler_values = stable_nodes[128].get("widgets_values_named", {})
if stable_sampler_values.get("denoise_strength") != 1:
    raise SystemExit("Denoise final do workflow Stable deve ser 1")
stable_prompt_values = stable_nodes[241].get("widgets_values_named", {})
if not str(stable_prompt_values.get("positive_prompt", "")).startswith(
    "A calm, relaxed and friendly person"
):
    raise SystemExit("Prompt positivo final do workflow Stable esta incorreto")
if not str(stable_prompt_values.get("negative_prompt", "")).startswith(
    "Exaggerated facial expressions"
):
    raise SystemExit("Prompt negativo final do workflow Stable esta incorreto")
stable_multitalk_values = stable_nodes[194].get("widgets_values", [])
if len(stable_multitalk_values) < 5 or stable_multitalk_values[3:5] != [1, 1]:
    raise SystemExit("Escalas de audio finais do workflow Stable estao incorretas")
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

stable_no_ls_nodes = {
    node["id"]: node for node in stable_no_latentsync_workflow["nodes"]
}
stable_no_ls_links = {
    link[0]: link for link in stable_no_latentsync_workflow["links"]
}
if any(
    node.get("type") in {"LatentSyncNode", "LatentSyncStableNode"}
    for node in stable_no_latentsync_workflow["nodes"]
):
    raise SystemExit("O workflow Stable sem LatentSync ainda contem LatentSync")
if stable_no_ls_links.get(560) != [560, 130, 0, 300, 0, "IMAGE"]:
    raise SystemExit("WanVideoDecode nao esta ligado diretamente a saida Stable")
if 449 in stable_no_ls_links or 545 in stable_no_ls_links:
    raise SystemExit("Links residuais do LatentSync encontrados no novo preset")
if stable_no_ls_links.get(559) != [559, 254, 0, 131, 1, "AUDIO"]:
    raise SystemExit("Audio original nao esta ligado ao VHS do novo preset")
for node_id, stable_source_node in stable_nodes.items():
    if node_id == 307:
        continue
    target_node = stable_no_ls_nodes.get(node_id)
    if target_node is None:
        raise SystemExit(f"Node Stable ausente no preset sem LatentSync: {node_id}")
    source_values = stable_source_node.get("widgets_values")
    target_values = target_node.get("widgets_values")
    if node_id == 131:
        source_values = dict(source_values)
        target_values = dict(target_values)
        for values in (source_values, target_values):
            values.pop("filename_prefix", None)
            values.pop("videopreview", None)
    if target_values != source_values:
        raise SystemExit(
            f"Parametros Stable alterados no preset sem LatentSync: node {node_id}"
        )
stable_no_ls_combine = stable_no_ls_nodes[131].get("widgets_values", {})
if (
    stable_no_ls_combine.get("filename_prefix")
    != "InfiniteTalk_V2V_Stable_NoLatentSync"
):
    raise SystemExit("Prefixo do Stable sem LatentSync invalido")

lipforcing_nodes = {node["id"]: node for node in lipforcing_workflow["nodes"]}
lipforcing_links = {link[0]: link for link in lipforcing_workflow["links"]}
if lipforcing_nodes.get(3, {}).get("type") != "LipForcing14B":
    raise SystemExit("Node LipForcing14B ausente do workflow local")
if any(
    node.get("type") in {"LipForcingLoadVideo", "LipForcingLoadAudio"}
    for node in lipforcing_workflow["nodes"]
):
    raise SystemExit("Loaders de caminho obsoletos ainda existem no Lip Forcing")
if lipforcing_links:
    raise SystemExit("Workflow Lip Forcing direto nao deve conter links internos")
if (
    lipforcing_nodes[3]
    .get("properties", {})
    .get("infinitetalk_lipforcing_schema")
    != 3
):
    raise SystemExit("Schema de qualidade do Lip Forcing deve ser 3")
if lipforcing_nodes[3].get("widgets_values") != [
    None,
    None,
    42,
    "streaming_taehv",
    True,
    "LipForcing14B_Final",
    "segmentwise_max_quality",
    "mouth_only",
    False,
]:
    raise SystemExit("Defaults do Lip Forcing 14B estao incorretos")

print(
    "OK: dependencias, workflows e custom nodes do "
    "InfiniteTalk + LatentSync 1.6 + Lip Forcing presentes"
)
