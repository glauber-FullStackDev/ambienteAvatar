ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_COMMIT=a25c7bf2b8c7408d8724f4245dbe09d95992e3a1
ARG PERSONAL_LORA_SHA256=c9443d3aad3151125c086c10e6a423ccc973bd445aa67cd5a537bdd57f6df78a

ENV COMFYUI_HOME=/opt/ComfyUI \
    COMFYUI_MODELS=/opt/ComfyUI/models \
    COMFYUI_PORT=8188 \
    DOWNLOAD_LTX25_MODELS_ON_START=0 \
    HF_HOME=/opt/ComfyUI/models/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg git libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Comfy-Org/ComfyUI.git "${COMFYUI_HOME}" \
    && git -C "${COMFYUI_HOME}" checkout --detach "${COMFYUI_COMMIT}" \
    && pip install -r "${COMFYUI_HOME}/requirements.txt"

COPY ltx-2.3/scripts/ /opt/ltx23-scripts/
COPY ltx-2.3/custom_nodes/ComfyUI-LastFrameExport/ "${COMFYUI_HOME}/custom_nodes/ComfyUI-LastFrameExport/"
COPY ltx-2.3/assets/glauberavatar.safetensors /opt/ltx23-assets/glauberavatar.safetensors
COPY ltx-2.3/assets/video_ltx2_3_ia2v_personal_source.json /opt/ltx23-assets/video_ltx2_3_ia2v_personal_source.json
COPY ltx-2.3-serverless/requirements.txt /opt/serverless/requirements.txt
COPY ltx-2.3-serverless/src/ /opt/serverless/src/

RUN pip install "huggingface_hub[hf_xet]>=0.34,<2" -r /opt/serverless/requirements.txt \
    && mkdir -p "${COMFYUI_HOME}/input" "${COMFYUI_HOME}/output" "${COMFYUI_HOME}/models/loras" /opt/defaults/workflows /var/log/portal \
    && echo "${PERSONAL_LORA_SHA256}  /opt/ltx23-assets/glauberavatar.safetensors" | sha256sum -c - \
    && python /opt/serverless/src/workflow_api.py \
        --source /opt/ltx23-assets/video_ltx2_3_ia2v_personal_source.json \
        --output /opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_api.json \
    && python -m compileall -q /opt/serverless/src

WORKDIR /opt/serverless
CMD ["python", "-u", "/opt/serverless/src/handler.py"]
