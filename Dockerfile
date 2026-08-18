ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG COMFYUI_REF=c1739380c6fab78e7e263cb665d04aafbfe24593
ARG LONGCAT_NODE_REF=08b4daedfaed69abaf467097f8665615b2137331
ARG COMFYUI_MANAGER_REF=4f56cf3dfa7de5d8a8614dfe202ff8d613ba2244
ARG VIDEO_HELPER_SUITE_REF=4ee72c065db22c9d96c2427954dc69e7b908444b
ARG ADVANCED_LIVE_PORTRAIT_REF=3bba732915e22f18af0d221b9c5c282990181f1b
ARG LATENT_SYNC_WRAPPER_REF=360d5283d7276aee68b4237b1387e594e4ce640e
ARG INSTALL_MANAGER=1
ARG INSTALL_XFORMERS=0
ARG INSTALL_FLASH_ATTN=0

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

RUN git clone https://github.com/comfyanonymous/ComfyUI.git "${COMFYUI_HOME}" \
    && cd "${COMFYUI_HOME}" \
    && git checkout --detach "${COMFYUI_REF}" \
    && pip install -r requirements.txt \
    && pip install "huggingface_hub[hf_xet]>=0.34,<2"

RUN git clone https://github.com/rookiestar28/ComfyUI-LongCat-Avatar.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-LongCat-Avatar" \
    && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-LongCat-Avatar" \
    && git checkout --detach "${LONGCAT_NODE_REF}" \
    && pip install -r requirements.txt

RUN if [ "${INSTALL_MANAGER}" = "1" ]; then \
        git clone https://github.com/ltdrdata/ComfyUI-Manager.git \
            "${COMFYUI_HOME}/custom_nodes/ComfyUI-Manager" \
        && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-Manager" \
        && git checkout --detach "${COMFYUI_MANAGER_REF}" \
        && pip install -r requirements.txt; \
    fi

# Packs used by the avatar refinement and lip-sync workflows. Revisions are
# pinned so rebuilding the image does not silently change node contracts.
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-VideoHelperSuite" \
    && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-VideoHelperSuite" \
    && git checkout --detach "${VIDEO_HELPER_SUITE_REF}" \
    && pip install -r requirements.txt \
    && git clone https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-AdvancedLivePortrait" \
    && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-AdvancedLivePortrait" \
    && git checkout --detach "${ADVANCED_LIVE_PORTRAIT_REF}" \
    && pip install -r requirements.txt \
    && git clone https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper" \
    && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper" \
    && git checkout --detach "${LATENT_SYNC_WRAPPER_REF}" \
    && pip install -r requirements.txt \
    && pip install pytorch-lightning accelerate "numpy==2.2.6" \
    && rm -rf "${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints" \
    && mkdir -p "${COMFYUI_HOME}/models/latentsync" \
    && ln -s "${COMFYUI_HOME}/models/latentsync" \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints" \
    && ln -s "${COMFYUI_HOME}/models/latentsync" /root/.latentsync16_models \
    && touch /root/.latentsync16_dependencies_installed

# Optional acceleration is deliberately opt-in: the correct wheels/toolchain depend
# on the target GPU architecture. The seeded workflow uses portable PyTorch SDPA.
RUN if [ "${INSTALL_XFORMERS}" = "1" ]; then \
        pip install --no-deps xformers==0.0.32.post2; \
    fi \
    && if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
        pip install packaging psutil \
        && MAX_JOBS=4 pip install --no-build-isolation flash-attn==2.7.4.post1; \
    fi

COPY custom_nodes/ComfyUI-LongCat-Expression-Control \
    ${COMFYUI_HOME}/custom_nodes/ComfyUI-LongCat-Expression-Control
COPY scripts /opt/avatar-scripts

RUN chmod +x /opt/avatar-scripts/entrypoint.py \
    && mkdir -p \
        /data/cache/huggingface \
        /opt/defaults/workflows \
        "${COMFYUI_HOME}/input" \
        "${COMFYUI_HOME}/models/audio_encoders" \
        "${COMFYUI_HOME}/models/clip" \
        "${COMFYUI_HOME}/models/diffusion_models" \
        "${COMFYUI_HOME}/models/latentsync" \
        "${COMFYUI_HOME}/models/liveportrait" \
        "${COMFYUI_HOME}/models/longcat" \
        "${COMFYUI_HOME}/models/loras" \
        "${COMFYUI_HOME}/models/ultralytics" \
        "${COMFYUI_HOME}/models/vae" \
        "${COMFYUI_HOME}/output" \
        "${COMFYUI_HOME}/user/default/workflows" \
    && cp "${COMFYUI_HOME}/custom_nodes/ComfyUI-LongCat-Avatar/example_workflows/longcat-avatar1.5.json" \
        /opt/defaults/workflows/longcat-avatar1.5.json \
    && python -m compileall -q \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-LongCat-Avatar" \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-LongCat-Expression-Control" \
        /opt/avatar-scripts \
    && python /opt/avatar-scripts/check_dependencies.py \
    && python /opt/avatar-scripts/smoke_custom_nodes.py

WORKDIR /opt/ComfyUI
EXPOSE 8188

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=3)" || exit 1

ENTRYPOINT ["python", "/opt/avatar-scripts/entrypoint.py"]
CMD ["serve"]
