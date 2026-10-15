ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_COMMIT=a25c7bf2b8c7408d8724f4245dbe09d95992e3a1
ARG GGUF_NODE_COMMIT=6ea2651e7df66d7585f6ffee804b20e92fb38b8a
ARG KJNODES_COMMIT=17195fcfa8a05f601b766a0f09c9733284f47416
ARG PERSONAL_LORA_SHA256=c9443d3aad3151125c086c10e6a423ccc973bd445aa67cd5a537bdd57f6df78a
ARG PERSONAL_WORKFLOW_SHA256=f2b4a56c5b63ebb7522d5a1b2d191b599aaff4307a7e64c4f6b7dfa652a765a5

ENV COMFYUI_HOME=/opt/ComfyUI \
    COMFYUI_MODELS=/opt/ComfyUI/models \
    COMFYUI_PORT=8188 \
    DOWNLOAD_MODELS_ON_START=1 \
    LTX23_GGUF_QUANT=Q4_K_M \
    HF_HOME=/opt/ComfyUI/models/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        jq \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Comfy-Org/ComfyUI.git "${COMFYUI_HOME}" \
    && git -C "${COMFYUI_HOME}" checkout --detach "${COMFYUI_COMMIT}" \
    && pip install -r "${COMFYUI_HOME}/requirements.txt" \
    && pip install "huggingface_hub[hf_xet]>=0.34,<2" "jupyterlab>=4.2,<5"

# ComfyUI-GGUF: loaders UnetLoaderGGUF/DualCLIPLoaderGGUF (suporte a LTX-2 e
# Gemma GGUF) e ComfyUI-KJNodes (VAELoaderKJ para os VAEs separados).
RUN git clone https://github.com/city96/ComfyUI-GGUF.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-GGUF" \
    && git -C "${COMFYUI_HOME}/custom_nodes/ComfyUI-GGUF" \
        checkout --detach "${GGUF_NODE_COMMIT}" \
    && pip install -r \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-GGUF/requirements.txt"

RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-KJNodes" \
    && git -C "${COMFYUI_HOME}/custom_nodes/ComfyUI-KJNodes" \
        checkout --detach "${KJNODES_COMMIT}" \
    && pip install -r \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-KJNodes/requirements.txt"

COPY scripts/ /opt/ltx23-scripts/
COPY custom_nodes/ComfyUI-LastFrameExport/ "${COMFYUI_HOME}/custom_nodes/ComfyUI-LastFrameExport/"
COPY --chmod=644 assets/glauberavatar.safetensors /opt/ltx23-assets/glauberavatar.safetensors
COPY --chmod=644 assets/video_ltx2_3_ia2v_personal_source.json /opt/ltx23-assets/video_ltx2_3_ia2v_personal_source.json

RUN chmod +x \
        /opt/ltx23-scripts/container-entrypoint.sh \
        /opt/ltx23-scripts/entrypoint.py \
        /opt/ltx23-scripts/download_models.py \
        /opt/ltx23-scripts/workflow_utils.py \
        /opt/ltx23-scripts/personal_workflows.py \
        /opt/ltx23-scripts/gguf_convert.py \
        /opt/ltx23-scripts/build_ia2v_personal_lora_16gb.py \
        /opt/ltx23-scripts/build_a2v_personal_lora_16gb.py \
    && mkdir -p \
        "${COMFYUI_HOME}/input" \
        "${COMFYUI_HOME}/models/diffusion_models" \
        "${COMFYUI_HOME}/models/latent_upscale_models" \
        "${COMFYUI_HOME}/models/loras" \
        "${COMFYUI_HOME}/models/text_encoders" \
        "${COMFYUI_HOME}/models/vae" \
        "${COMFYUI_HOME}/output" \
        "${COMFYUI_HOME}/user/default/workflows" \
        /var/log/portal \
    && echo "${PERSONAL_LORA_SHA256}  /opt/ltx23-assets/glauberavatar.safetensors" | sha256sum -c - \
    && echo "${PERSONAL_WORKFLOW_SHA256}  /opt/ltx23-assets/video_ltx2_3_ia2v_personal_source.json" | sha256sum -c - \
    && install -m 0644 /opt/ltx23-assets/glauberavatar.safetensors "${COMFYUI_HOME}/models/loras/glauberavatar.safetensors" \
    && mkdir -p /opt/defaults/workflows \
    && python /opt/ltx23-scripts/build_ia2v_personal_lora_16gb.py \
        --source /opt/ltx23-assets/video_ltx2_3_ia2v_personal_source.json \
        --output /opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora_16gb.json \
    && python /opt/ltx23-scripts/build_a2v_personal_lora_16gb.py \
        --source /opt/ltx23-assets/video_ltx2_3_ia2v_personal_source.json \
        --output /opt/defaults/workflows/video_ltx2_3_a2v_personal_lora_16gb.json \
    && python -m compileall -q /opt/ltx23-scripts \
    && python /opt/ltx23-scripts/check_dependencies.py \
    && python /opt/ltx23-scripts/smoke_core_nodes.py

WORKDIR ${COMFYUI_HOME}

EXPOSE 8188

HEALTHCHECK --interval=30s --timeout=5s --start-period=6h --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${COMFYUI_PORT}/system_stats" >/dev/null || exit 1

ENTRYPOINT ["/opt/ltx23-scripts/container-entrypoint.sh"]
CMD ["serve"]
