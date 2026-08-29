ARG PYTORCH_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_COMMIT=a25c7bf2b8c7408d8724f4245dbe09d95992e3a1
ARG WORKFLOW_COMMIT=d11b69157009227ad2a7d3a927a1eb68a3d5f281
ARG WORKFLOW_SHA256=7823a703f472d9c5e6f82c462235ff89a0fa14752ec1fd947c4422cf53e47685
ARG ID_LORA_WORKFLOW_COMMIT=04f33569dad7a1d277429bda9f35209dfa4d91cf
ARG ID_LORA_WORKFLOW_SHA256=fcffe421129bac16b4f0655e54130d633280cdaf6949e145221e7090be42151f
ARG LTX25_WORKFLOW_COMMIT=1121504798345b1bb4e6350991f90512c4ba1ed9
ARG LTX25_WORKFLOW_SHA256=0a88024394467250013ce611ee46d01cc7e73078a0899e4c80709080c5101f71
ARG BFSNODES_COMMIT=0a2553869254eef4f3f735fdd9fea04614c3dd7e
ARG LTXVIDEO_COMMIT=15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d
ARG INGREDIENTS_WORKFLOW_SHA256=a9d75c1c8546d5e0125d16ac071abd473b45bbfdcfbec87b1cf7abc17dd90d32
ARG PERSONAL_LORA_SHA256=c9443d3aad3151125c086c10e6a423ccc973bd445aa67cd5a537bdd57f6df78a

ENV COMFYUI_HOME=/opt/ComfyUI \
    COMFYUI_MODELS=/opt/ComfyUI/models \
    COMFYUI_PORT=8188 \
    DOWNLOAD_MODELS_ON_START=1 \
    DOWNLOAD_LTX25_MODELS_ON_START=1 \
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

RUN git clone https://github.com/alisson-anjos/ComfyUI-BFSNodes.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-BFSNodes" \
    && git -C "${COMFYUI_HOME}/custom_nodes/ComfyUI-BFSNodes" \
        checkout --detach "${BFSNODES_COMMIT}" \
    && pip install -r \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-BFSNodes/requirements.txt"

RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-LTXVideo" \
    && git -C "${COMFYUI_HOME}/custom_nodes/ComfyUI-LTXVideo" \
        checkout --detach "${LTXVIDEO_COMMIT}" \
    && pip install -r \
        "${COMFYUI_HOME}/custom_nodes/ComfyUI-LTXVideo/requirements.txt" \
    && echo "${INGREDIENTS_WORKFLOW_SHA256}  ${COMFYUI_HOME}/custom_nodes/ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json" \
        | sha256sum -c -

RUN mkdir -p /opt/defaults/workflows \
    && curl -fsSL \
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/${WORKFLOW_COMMIT}/templates/video_ltx2_3_ia2v.json" \
        -o /opt/defaults/workflows/video_ltx2_3_ia2v.json \
    && echo "${WORKFLOW_SHA256}  /opt/defaults/workflows/video_ltx2_3_ia2v.json" | sha256sum -c - \
    && curl -fsSL \
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/${ID_LORA_WORKFLOW_COMMIT}/templates/video_ltx2_3_id_lora.json" \
        -o /opt/defaults/workflows/video_ltx2_3_id_lora.json \
    && echo "${ID_LORA_WORKFLOW_SHA256}  /opt/defaults/workflows/video_ltx2_3_id_lora.json" | sha256sum -c - \
    && curl -fsSL \
        "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/${LTX25_WORKFLOW_COMMIT}/templates/video_ltx2_5_i2v.json" \
        -o /opt/defaults/workflows/video_ltx2_5_i2v.json \
    && echo "${LTX25_WORKFLOW_SHA256}  /opt/defaults/workflows/video_ltx2_5_i2v.json" | sha256sum -c -

COPY scripts/ /opt/ltx23-scripts/
COPY custom_nodes/ComfyUI-LastFrameExport/ "${COMFYUI_HOME}/custom_nodes/ComfyUI-LastFrameExport/"
COPY --chmod=644 assets/glauberavatar.safetensors /opt/ltx23-assets/glauberavatar.safetensors

RUN chmod +x \
        /opt/ltx23-scripts/container-entrypoint.sh \
        /opt/ltx23-scripts/entrypoint.py \
        /opt/ltx23-scripts/download_models.py \
        /opt/ltx23-scripts/patch_ltxvideo_kornia.py \
        /opt/ltx23-scripts/build_ia2v_talkvid_workflow.py \
        /opt/ltx23-scripts/build_ia2v_best_face_workflow.py \
        /opt/ltx23-scripts/build_ia2v_ingredients_workflow.py \
        /opt/ltx23-scripts/build_ia2v_personal_lora_workflow.py \
        /opt/ltx23-scripts/build_ingredients_reference_workflows.py \
        /opt/ltx23-scripts/build_ltx25_ia2v_workflow.py \
    && mkdir -p \
        "${COMFYUI_HOME}/input" \
        "${COMFYUI_HOME}/models/checkpoints" \
        "${COMFYUI_HOME}/models/diffusion_models" \
        "${COMFYUI_HOME}/models/latent_upscale_models" \
        "${COMFYUI_HOME}/models/loras" \
        "${COMFYUI_HOME}/models/text_encoders" \
        "${COMFYUI_HOME}/models/vae" \
        "${COMFYUI_HOME}/output" \
        "${COMFYUI_HOME}/user/default/workflows" \
        /var/log/portal \
    && echo "${PERSONAL_LORA_SHA256}  /opt/ltx23-assets/glauberavatar.safetensors" | sha256sum -c - \
    && install -m 0644 /opt/ltx23-assets/glauberavatar.safetensors "${COMFYUI_HOME}/models/loras/glauberavatar.safetensors" \
    && python /opt/ltx23-scripts/patch_ltxvideo_kornia.py \
    && python /opt/ltx23-scripts/build_ia2v_talkvid_workflow.py \
        --ia2v /opt/defaults/workflows/video_ltx2_3_ia2v.json \
        --id-lora /opt/defaults/workflows/video_ltx2_3_id_lora.json \
        --output /opt/defaults/workflows/video_ltx2_3_ia2v_talkvid.json \
    && python /opt/ltx23-scripts/build_ia2v_best_face_workflow.py \
        --ia2v /opt/defaults/workflows/video_ltx2_3_ia2v.json \
        --output /opt/defaults/workflows/video_ltx2_3_ia2v_best_face.json \
    && python /opt/ltx23-scripts/build_ia2v_personal_lora_workflow.py \
        --ia2v /opt/defaults/workflows/video_ltx2_3_ia2v.json \
        --output /opt/defaults/workflows/video_ltx2_3_ia2v_personal_lora.json \
    && python /opt/ltx23-scripts/build_ia2v_ingredients_workflow.py \
        --ia2v /opt/defaults/workflows/video_ltx2_3_ia2v.json \
        --profile corrected \
        --output /opt/defaults/workflows/video_ltx2_3_ia2v_ingredients.json \
    && python /opt/ltx23-scripts/build_ia2v_ingredients_workflow.py \
        --ia2v /opt/defaults/workflows/video_ltx2_3_ia2v.json \
        --profile legacy-v2 \
        --output /opt/defaults/workflows/video_ltx2_3_ia2v_ingredients_legacy_v2.json \
    && python /opt/ltx23-scripts/build_ingredients_reference_workflows.py \
        --official "${COMFYUI_HOME}/custom_nodes/ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json" \
        --profile official \
        --output /opt/defaults/workflows/video_ltx2_3_ingredients_official_single_stage.json \
    && python /opt/ltx23-scripts/build_ingredients_reference_workflows.py \
        --official "${COMFYUI_HOME}/custom_nodes/ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_ICLoRA_Ingredients_Single_Stage_Distilled.json" \
        --profile wangp-i2v-15s \
        --output /opt/defaults/workflows/video_ltx2_3_ingredients_wangp_i2v_15s.json \
    && python /opt/ltx23-scripts/build_ltx25_ia2v_workflow.py \
        --i2v /opt/defaults/workflows/video_ltx2_5_i2v.json \
        --output /opt/defaults/workflows/video_ltx2_5_ia2v_distilled_8steps.json \
    && python -m compileall -q /opt/ltx23-scripts \
    && python /opt/ltx23-scripts/check_dependencies.py \
    && python /opt/ltx23-scripts/smoke_core_nodes.py

WORKDIR ${COMFYUI_HOME}

EXPOSE 8188

HEALTHCHECK --interval=30s --timeout=5s --start-period=6h --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${COMFYUI_PORT}/system_stats" >/dev/null || exit 1

ENTRYPOINT ["/opt/ltx23-scripts/container-entrypoint.sh"]
CMD ["serve"]
