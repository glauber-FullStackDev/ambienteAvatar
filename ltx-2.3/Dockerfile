ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_COMMIT=2e47082c8ed1d1a0fe54add57f98b63433cfacbb
ARG WORKFLOW_COMMIT=d11b69157009227ad2a7d3a927a1eb68a3d5f281
ARG WORKFLOW_SHA256=7823a703f472d9c5e6f82c462235ff89a0fa14752ec1fd947c4422cf53e47685
ARG ID_LORA_WORKFLOW_COMMIT=04f33569dad7a1d277429bda9f35209dfa4d91cf
ARG ID_LORA_WORKFLOW_SHA256=fcffe421129bac16b4f0655e54130d633280cdaf6949e145221e7090be42151f

ENV COMFYUI_HOME=/opt/ComfyUI \
    COMFYUI_MODELS=/opt/ComfyUI/models \
    COMFYUI_PORT=8188 \
    DOWNLOAD_MODELS_ON_START=1 \
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

RUN mkdir -p /opt/defaults/workflows \
    && curl -fsSL \
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/${WORKFLOW_COMMIT}/templates/video_ltx2_3_ia2v.json" \
        -o /opt/defaults/workflows/video_ltx2_3_ia2v.json \
    && echo "${WORKFLOW_SHA256}  /opt/defaults/workflows/video_ltx2_3_ia2v.json" | sha256sum -c - \
    && curl -fsSL \
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/${ID_LORA_WORKFLOW_COMMIT}/templates/video_ltx2_3_id_lora.json" \
        -o /opt/defaults/workflows/video_ltx2_3_id_lora.json \
    && echo "${ID_LORA_WORKFLOW_SHA256}  /opt/defaults/workflows/video_ltx2_3_id_lora.json" | sha256sum -c -

COPY scripts/ /opt/ltx23-scripts/

RUN chmod +x \
        /opt/ltx23-scripts/container-entrypoint.sh \
        /opt/ltx23-scripts/entrypoint.py \
        /opt/ltx23-scripts/download_models.py \
        /opt/ltx23-scripts/build_ia2v_talkvid_workflow.py \
    && mkdir -p \
        "${COMFYUI_HOME}/input" \
        "${COMFYUI_HOME}/models/checkpoints" \
        "${COMFYUI_HOME}/models/latent_upscale_models" \
        "${COMFYUI_HOME}/models/loras" \
        "${COMFYUI_HOME}/models/text_encoders" \
        "${COMFYUI_HOME}/output" \
        "${COMFYUI_HOME}/user/default/workflows" \
        /var/log/portal \
    && python /opt/ltx23-scripts/build_ia2v_talkvid_workflow.py \
        --ia2v /opt/defaults/workflows/video_ltx2_3_ia2v.json \
        --id-lora /opt/defaults/workflows/video_ltx2_3_id_lora.json \
        --output /opt/defaults/workflows/video_ltx2_3_ia2v_talkvid.json \
    && python -m compileall -q /opt/ltx23-scripts \
    && python /opt/ltx23-scripts/check_dependencies.py \
    && python /opt/ltx23-scripts/smoke_core_nodes.py

WORKDIR ${COMFYUI_HOME}

EXPOSE 8188

HEALTHCHECK --interval=30s --timeout=5s --start-period=6h --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${COMFYUI_PORT}/system_stats" >/dev/null || exit 1

ENTRYPOINT ["/opt/ltx23-scripts/container-entrypoint.sh"]
CMD ["serve"]
