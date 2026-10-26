ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_COMMIT=a25c7bf2b8c7408d8724f4245dbe09d95992e3a1
ARG GGUF_NODE_COMMIT=6ea2651e7df66d7585f6ffee804b20e92fb38b8a
ARG KJNODES_COMMIT=17195fcfa8a05f601b766a0f09c9733284f47416
ARG BFSNODES_COMMIT=0a2553869254eef4f3f735fdd9fea04614c3dd7e
ARG PERSONAL_LORA_SHA256=c9443d3aad3151125c086c10e6a423ccc973bd445aa67cd5a537bdd57f6df78a
ARG PERSONAL_WORKFLOW_SHA256=a410e1018f1ce4f7e906e351c9d483b5fbc2603081e8ac93b4c4a5157fe88a44
ARG LTX_IMAGE_SHA=dev

ENV COMFYUI_HOME=/opt/ComfyUI \
    COMFYUI_MODELS=/opt/ComfyUI/models \
    COMFYUI_PORT=8188 \
    LTX_IMAGE_SHA=${LTX_IMAGE_SHA} \
    LTX23_GGUF_QUANT=Q4_K_M \
    MODELS_ROOT=/data \
    WORKFLOW_PATH=/opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_16gb_api.json \
    DEFAULT_IA2V_PERSONAL_LORA_WORKFLOW=/opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_16gb.json \
    HF_HOME=/opt/ComfyUI/models/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl ffmpeg git libgl1 libglib2.0-0 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Comfy-Org/ComfyUI.git "${COMFYUI_HOME}" \
    && git -C "${COMFYUI_HOME}" checkout --detach "${COMFYUI_COMMIT}" \
    && pip install -r "${COMFYUI_HOME}/requirements.txt"

# Mesmos custom nodes da imagem interativa GGUF 16 GB (../ltx-2.3-16gb/) mais
# o BFSNodes: o workflow GGUF usa ComfySwitchNode (no 348, roteia o prompt
# enhancer) e o pack BFSNodes nao faz parte da imagem do Vast.
RUN git clone https://github.com/city96/ComfyUI-GGUF.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-GGUF" \
    && git -C "${COMFYUI_HOME}/custom_nodes/ComfyUI-GGUF" checkout --detach "${GGUF_NODE_COMMIT}" \
    && pip install -r "${COMFYUI_HOME}/custom_nodes/ComfyUI-GGUF/requirements.txt"

RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-KJNodes" \
    && git -C "${COMFYUI_HOME}/custom_nodes/ComfyUI-KJNodes" checkout --detach "${KJNODES_COMMIT}" \
    && pip install -r "${COMFYUI_HOME}/custom_nodes/ComfyUI-KJNodes/requirements.txt"

RUN git clone https://github.com/alisson-anjos/ComfyUI-BFSNodes.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-BFSNodes" \
    && git -C "${COMFYUI_HOME}/custom_nodes/ComfyUI-BFSNodes" checkout --detach "${BFSNODES_COMMIT}" \
    && pip install -r "${COMFYUI_HOME}/custom_nodes/ComfyUI-BFSNodes/requirements.txt"

COPY ltx-2.3-16gb/scripts/ /opt/ltx23-scripts/
COPY ltx-2.3-16gb-serverless/scripts/smoke_api_workflow.py /opt/serverless/scripts/smoke_api_workflow.py
COPY ltx-2.3-16gb/custom_nodes/ComfyUI-LastFrameExport/ "${COMFYUI_HOME}/custom_nodes/ComfyUI-LastFrameExport/"
COPY ltx-2.3-16gb/assets/glauberavatar.safetensors /opt/ltx23-assets/glauberavatar.safetensors
COPY ltx-2.3-16gb/assets/video_ltx2_3_ia2v_personal_source.json /opt/ltx23-assets/video_ltx2_3_ia2v_personal_source.json
COPY ltx-2.3-16gb-serverless/requirements.txt /opt/serverless/requirements.txt
COPY ltx-2.3-16gb-serverless/src/ /opt/serverless/src/

# Os modelos GGUF (~26 GiB) NAO entram na imagem: o bootstrap baixa no boot do
# worker para $MODELS_ROOT (volume montado), como no worker Runpod.
RUN pip install "huggingface_hub[hf_xet]>=0.34,<2" -r /opt/serverless/requirements.txt \
    && chmod +x /opt/ltx23-scripts/*.py /opt/ltx23-scripts/*.sh \
    && mkdir -p \
        "${COMFYUI_HOME}/input" \
        "${COMFYUI_HOME}/models/diffusion_models" \
        "${COMFYUI_HOME}/models/latent_upscale_models" \
        "${COMFYUI_HOME}/models/loras" \
        "${COMFYUI_HOME}/models/text_encoders" \
        "${COMFYUI_HOME}/models/vae" \
        "${COMFYUI_HOME}/output" \
        "${COMFYUI_HOME}/user/default/workflows" \
        /opt/defaults/workflows \
        /opt/ltx23-assets \
        /data \
        /var/log/portal \
    && echo "${PERSONAL_LORA_SHA256}  /opt/ltx23-assets/glauberavatar.safetensors" | sha256sum -c - \
    && echo "${PERSONAL_WORKFLOW_SHA256}  /opt/ltx23-assets/video_ltx2_3_ia2v_personal_source.json" | sha256sum -c - \
    && python /opt/ltx23-scripts/build_ia2v_personal_lora_16gb.py \
        --source /opt/ltx23-assets/video_ltx2_3_ia2v_personal_source.json \
        --output /opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_16gb.json \
    && python /opt/ltx23-scripts/build_a2v_personal_lora_16gb.py \
        --source /opt/ltx23-assets/video_ltx2_3_ia2v_personal_source.json \
        --output /opt/defaults/workflows/video_ltx2_3_a2v_personal_lora_16gb.json \
    && python /opt/serverless/src/workflow_api.py \
        --source /opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_16gb.json \
        --output /opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_16gb_api.json \
    && python -m compileall -q /opt/serverless/src /opt/ltx23-scripts \
    && python /opt/ltx23-scripts/check_dependencies.py \
    && python /opt/ltx23-scripts/smoke_core_nodes.py \
    && python /opt/serverless/scripts/smoke_api_workflow.py

WORKDIR /opt/serverless/src

EXPOSE 8000

CMD ["python", "-u", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
