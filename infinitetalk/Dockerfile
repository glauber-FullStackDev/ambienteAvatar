ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG COMFYUI_REF=8b2d29176f7f1b9b72f9dff11a0848027d136910
ARG WAN_VIDEO_WRAPPER_REF=088128b224242e110d3906c6750e9a3a348a659b
ARG COMFYUI_MANAGER_REF=d5992a117ee98663b6e93fb040c1e43ee3b25d68
ARG VIDEO_HELPER_SUITE_REF=4ee72c065db22c9d96c2427954dc69e7b908444b
ARG KJ_NODES_REF=3f20054214fec9f9234fd3841ae6f1e4287948f6
ARG MELBAND_ROFORMER_REF=92c86854e6654f4aacc97484471af95c98ea16d4
ARG LATENT_SYNC_WRAPPER_REF=360d5283d7276aee68b4237b1387e594e4ce640e
ARG LIPFORCING_REF=fc864771eb347ca3ccaaef9c0b583ff6ccc9f184
ARG INSTALL_MANAGER=1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/cache/huggingface \
    LATENTSYNC_INSIGHTFACE_ROOT=/opt/ComfyUI/models/latentsync/auxiliary \
    XDG_CACHE_HOME=/opt/ComfyUI/models/latentsync \
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
    && git clone https://github.com/kijai/ComfyUI-MelBandRoFormer.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-MelBandRoFormer" \
    && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-MelBandRoFormer" \
    && git checkout --detach "${MELBAND_ROFORMER_REF}" \
    && pip install -r requirements.txt \
    && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-VideoHelperSuite" \
    && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-VideoHelperSuite" \
    && git checkout --detach "${VIDEO_HELPER_SUITE_REF}" \
    && pip install -r requirements.txt \
    && git clone https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper" \
    && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper" \
    && git checkout --detach "${LATENT_SYNC_WRAPPER_REF}" \
    && pip install -r requirements.txt \
    && pip install pytorch-lightning accelerate "numpy==2.2.6" \
    && pip install "insightface==1.0.1" \
    && pip uninstall -y onnxruntime onnxruntime-gpu \
    && pip install \
        "onnxruntime-gpu[cuda,cudnn]==1.26.0" \
        "nvidia-cublas-cu12==12.8.4.1" \
    && rm -rf "${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints" \
    && mkdir -p \
        "${COMFYUI_HOME}/models/latentsync/auxiliary/models/buffalo_l" \
        "${COMFYUI_HOME}/checkpoints" \
    && ln -s "${COMFYUI_HOME}/models/latentsync" \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper/checkpoints" \
    && ln -s "${COMFYUI_HOME}/models/latentsync/auxiliary" \
        "${COMFYUI_HOME}/checkpoints/auxiliary" \
    && ln -s "${COMFYUI_HOME}/models/latentsync" /root/.latentsync16_models \
    && touch /root/.latentsync16_dependencies_installed

RUN if [ "${INSTALL_MANAGER}" = "1" ]; then \
        git clone https://github.com/Comfy-Org/ComfyUI-Manager.git \
            "${COMFYUI_HOME}/custom_nodes/ComfyUI-Manager" \
        && cd "${COMFYUI_HOME}/custom_nodes/ComfyUI-Manager" \
        && git checkout --detach "${COMFYUI_MANAGER_REF}" \
        && pip install -r requirements.txt; \
    fi

RUN git clone https://github.com/cvlab-kaist/LipForcing.git /opt/LipForcing \
    && cd /opt/LipForcing \
    && git checkout --detach "${LIPFORCING_REF}" \
    && conda create -y -p /opt/lipforcing-venv python=3.12 pip \
    && /opt/lipforcing-venv/bin/pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 \
    && /opt/lipforcing-venv/bin/pip install -e /opt/LipForcing \
    && /opt/lipforcing-venv/bin/pip uninstall -y onnxruntime onnxruntime-gpu \
    && /opt/lipforcing-venv/bin/pip install onnxruntime-gpu==1.26.0 \
    && conda clean -afy

COPY scripts /opt/infinitetalk-scripts
COPY defaults /opt/defaults
COPY custom_nodes/ComfyUI-LipForcing \
    ${COMFYUI_HOME}/custom_nodes/ComfyUI-LipForcing
COPY patches/latentsync_face_detector.py \
    ${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper/latentsync/utils/face_detector.py
COPY patches/latentsync_stable_runtime.py \
    ${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper/latentsync_stable_runtime.py
COPY patches/latentsync_stable_node.py \
    ${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper/latentsync_stable_node.py
COPY patches/latentsync_video_io.py \
    ${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper/latentsync_video_io.py
COPY patches/latentsync_wrapper_init.py \
    ${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper/__init__.py

RUN chmod +x \
        /opt/infinitetalk-scripts/container-entrypoint.sh \
        /opt/infinitetalk-scripts/entrypoint.py \
    && mkdir -p \
        /data/cache/huggingface \
        /opt/defaults/workflows \
        "${COMFYUI_HOME}/input" \
        "${COMFYUI_HOME}/models/clip_vision" \
        "${COMFYUI_HOME}/models/diffusion_models/MelBandRoFormer" \
        "${COMFYUI_HOME}/models/diffusion_models/WanVideo/InfiniteTalk" \
        "${COMFYUI_HOME}/models/loras/WanVideo/Lightx2v" \
        "${COMFYUI_HOME}/models/latentsync/auxiliary/models/buffalo_l" \
        "${COMFYUI_HOME}/models/latentsync/vae" \
        "${COMFYUI_HOME}/models/latentsync/whisper" \
        "${COMFYUI_HOME}/models/lipforcing" \
        "${COMFYUI_HOME}/models/text_encoders" \
        "${COMFYUI_HOME}/models/vae/wanvideo" \
        "${COMFYUI_HOME}/models/wav2vec2" \
        "${COMFYUI_HOME}/output" \
        "${COMFYUI_HOME}/user/default/workflows" \
        /opt/LipForcing/checkpoints \
    && ln -s "${COMFYUI_HOME}/models/latentsync/auxiliary" \
        /opt/LipForcing/checkpoints/auxiliary \
    && cp "${COMFYUI_HOME}/custom_nodes/ComfyUI-WanVideoWrapper/example_workflows/wanvideo_2_1_14B_I2V_InfiniteTalk_example_03.json" \
        /opt/defaults/workflows/infinitetalk-i2v.json \
    && cp "${COMFYUI_HOME}/custom_nodes/ComfyUI-WanVideoWrapper/example_workflows/wanvideo_2_1_14B_V2V_InfiniteTalk_example_02.json" \
        /opt/defaults/workflows/infinitetalk-v2v.json \
    && python -m compileall -q \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-WanVideoWrapper" \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-KJNodes" \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-MelBandRoFormer" \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-LatentSyncWrapper" \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-LipForcing" \
        /opt/infinitetalk-scripts \
    && /opt/lipforcing-venv/bin/python -m compileall -q /opt/LipForcing \
    && /opt/lipforcing-venv/bin/python \
        /opt/LipForcing/scripts/inference/inference_streaming.py --help \
        >/dev/null \
    && python /opt/infinitetalk-scripts/check_dependencies.py \
    && python /opt/infinitetalk-scripts/smoke_custom_nodes.py

WORKDIR /opt/ComfyUI
EXPOSE 8188

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=3)" || exit 1

ENTRYPOINT ["/opt/infinitetalk-scripts/container-entrypoint.sh"]
CMD ["serve"]
