ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG COMFYUI_REF=8b2d29176f7f1b9b72f9dff11a0848027d136910
ARG WAN_VIDEO_WRAPPER_REF=088128b224242e110d3906c6750e9a3a348a659b
ARG KJ_NODES_REF=3f20054214fec9f9234fd3841ae6f1e4287948f6
ARG WAN_ANIMATE_PREPROCESS_REF=0e0b6a2a555625acf4d4aefb780e27d06937132f
ARG SEGMENT_ANYTHING_2_REF=0c35fff5f382803e2310103357b5e985f5437f32
ARG VIDEO_HELPER_SUITE_REF=4ee72c065db22c9d96c2427954dc69e7b908444b
ARG COMFYUI_MANAGER_REF=d5992a117ee98663b6e93fb040c1e43ee3b25d68
ARG INSTALL_MANAGER=1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/cache/huggingface \
    COMFYUI_HOME=/opt/ComfyUI

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
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Comfy-Org/ComfyUI.git "${COMFYUI_HOME}" \
    && cd "${COMFYUI_HOME}" \
    && git checkout --detach "${COMFYUI_REF}" \
    && pip install -r requirements.txt \
    && pip install "huggingface_hub[hf_xet]>=0.34,<2"

RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-WanVideoWrapper" \
    && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-WanVideoWrapper" \
    && git checkout --detach "${WAN_VIDEO_WRAPPER_REF}" \
    && pip install -r requirements.txt \
    && git clone https://github.com/kijai/ComfyUI-KJNodes.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-KJNodes" \
    && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-KJNodes" \
    && git checkout --detach "${KJ_NODES_REF}" \
    && pip install -r requirements.txt \
    && git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-WanAnimatePreprocess" \
    && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-WanAnimatePreprocess" \
    && git checkout --detach "${WAN_ANIMATE_PREPROCESS_REF}" \
    && pip install -r requirements.txt \
    && git clone https://github.com/kijai/ComfyUI-segment-anything-2.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-segment-anything-2" \
    && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-segment-anything-2" \
    && git checkout --detach "${SEGMENT_ANYTHING_2_REF}" \
    && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-VideoHelperSuite" \
    && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-VideoHelperSuite" \
    && git checkout --detach "${VIDEO_HELPER_SUITE_REF}" \
    && pip install -r requirements.txt

RUN if [ "${INSTALL_MANAGER}" = "1" ]; then \
        git clone https://github.com/Comfy-Org/ComfyUI-Manager.git \
            "${COMFYUI_HOME}/custom_nodes/ComfyUI-Manager" \
        && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-Manager" \
        && git checkout --detach "${COMFYUI_MANAGER_REF}" \
        && pip install -r requirements.txt; \
    fi

COPY scripts /opt/wan-animate-scripts

RUN chmod +x /opt/wan-animate-scripts/entrypoint.py \
    && mkdir -p \
        /data/cache/huggingface \
        /opt/defaults/workflows \
        "${COMFYUI_HOME}/input" \
        "${COMFYUI_HOME}/models/clip_vision" \
        "${COMFYUI_HOME}/models/detection/onnx" \
        "${COMFYUI_HOME}/models/diffusion_models/WanVideo/2_2" \
        "${COMFYUI_HOME}/models/loras/WanVideo/Lightx2v" \
        "${COMFYUI_HOME}/models/sam2" \
        "${COMFYUI_HOME}/models/text_encoders" \
        "${COMFYUI_HOME}/models/vae/wanvideo" \
        "${COMFYUI_HOME}/output" \
        "${COMFYUI_HOME}/user/default/workflows" \
    && cp "${COMFYUI_HOME}/custom_nodes/ComfyUI-WanVideoWrapper/example_workflows/wanvideo_WanAnimate_preprocess_example_02.json" \
        /opt/defaults/workflows/wan-animate-preprocess.json \
    && python -m compileall -q \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-WanVideoWrapper" \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-KJNodes" \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-WanAnimatePreprocess" \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-segment-anything-2" \
        /opt/wan-animate-scripts \
    && python /opt/wan-animate-scripts/check_dependencies.py

WORKDIR /opt/ComfyUI
EXPOSE 8188

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=3)" || exit 1

ENTRYPOINT ["python", "/opt/wan-animate-scripts/entrypoint.py"]
CMD ["serve"]
