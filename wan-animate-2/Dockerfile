ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG COMFYUI_REF=c67885b14556cf3e4e061862925282d403d09862
ARG COMFYUI_MANAGER_REF=f39cbd56fecae0b27a446c0cd450cd591f3a8bea
ARG WORKFLOW_REF=55818c64caf6d28309ddee204827e51a2c45f4dd
ARG INSTALL_MANAGER=1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/cache/huggingface \
    COMFYUI_HOME=/opt/ComfyUI \
    COMFYUI_MODELS=/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Comfy-Org/ComfyUI.git "${COMFYUI_HOME}" \
    && cd "${COMFYUI_HOME}" \
    && git checkout --detach "${COMFYUI_REF}" \
    && pip install -r requirements.txt \
    && pip install "huggingface_hub[hf_xet]>=0.34,<2"

RUN if [ "${INSTALL_MANAGER}" = "1" ]; then \
        git clone https://github.com/Comfy-Org/ComfyUI-Manager.git \
            "${COMFYUI_HOME}/custom_nodes/ComfyUI-Manager" \
        && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-Manager" \
        && git checkout --detach "${COMFYUI_MANAGER_REF}" \
        && pip install -r requirements.txt; \
    fi

COPY scripts /opt/wan-animate-2/scripts
COPY extra_model_paths.yaml /opt/wan-animate-2/extra_model_paths.yaml

RUN mkdir -p \
        /data/cache/huggingface \
        /models/clip_vision \
        /models/diffusion_models \
        /models/loras \
        /models/text_encoders \
        /models/vae \
        /opt/defaults/workflows \
        "${COMFYUI_HOME}/input" \
        "${COMFYUI_HOME}/output" \
        "${COMFYUI_HOME}/user/default/workflows" \
    && curl --fail --location --silent --show-error \
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/${WORKFLOW_REF}/templates/video_wan_animate2.json" \
        --output /opt/defaults/workflows/wan-animate-2-int8.json \
    && chmod +x /opt/wan-animate-2/scripts/entrypoint.py \
    && python -m compileall -q /opt/wan-animate-2/scripts \
    && python /opt/wan-animate-2/scripts/check_dependencies.py

WORKDIR /opt/ComfyUI
EXPOSE 8188

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=8 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=3)" || exit 1

ENTRYPOINT ["python", "/opt/wan-animate-2/scripts/entrypoint.py"]
CMD ["serve"]
