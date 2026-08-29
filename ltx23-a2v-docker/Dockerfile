# syntax=docker/dockerfile:1.7

# Verified on 2026-08-28 through the Docker Hub API for linux/amd64.
# This image intentionally omits system cuDNN: the pinned PyTorch cu132 wheels
# provide the exact cuDNN version selected by the LTX dependency lock.
ARG CUDA_IMAGE=nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1f7b8e96fa9ec5e36d4709a38c62df7b5665977446081811c12b8234d874bf
FROM ${CUDA_IMAGE}

ARG TARGETARCH
ARG LTX_COMMIT=a95ab856bf29407b6b066ede0abe1846050db56c
ARG UV_VERSION=0.11.7

LABEL org.opencontainers.image.title="LTX-2.3 personal A2V LoRA trainer" \
      org.opencontainers.image.source="https://github.com/Lightricks/LTX-2" \
      org.opencontainers.image.revision="${LTX_COMMIT}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1 \
    UV_PROJECT_ENVIRONMENT=/opt/LTX-2/.venv \
    PATH=/opt/LTX-2/.venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    LTX_HOME=/opt/LTX-2 \
    LTX_COMMIT=${LTX_COMMIT} \
    LTX_MODELS_DIR=/workspace/models \
    WANDB_PROJECT=ltx23-personal-a2v \
    HF_HOME=/workspace/.cache/huggingface \
    WANDB_DIR=/workspace/logs/wandb \
    XDG_CACHE_HOME=/workspace/.cache

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN test "${TARGETARCH}" = "amd64" || { echo "This image targets linux/amd64; got TARGETARCH=${TARGETARCH}" >&2; exit 1; }

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        htop \
        jq \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        nano \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
        rsync \
        tmux \
        unzip \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Official Astral installer, version-pinned.
RUN export UV_INSTALL_DIR=/usr/local/bin \
    && curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh \
    && uv --version

RUN export GIT_LFS_SKIP_SMUDGE=1 \
    && git clone --filter=blob:none https://github.com/Lightricks/LTX-2.git /opt/LTX-2 \
    && git -C /opt/LTX-2 checkout --detach "${LTX_COMMIT}" \
    && test "$(git -C /opt/LTX-2 rev-parse HEAD)" = "${LTX_COMMIT}"

# Upstream does not publish uv.lock at the pinned commit. This lock was resolved
# from that exact commit's pyproject files with uv 0.11.7 / Python 3.12.
COPY locks/uv.lock /opt/LTX-2/uv.lock

WORKDIR /opt/LTX-2

RUN UV_NO_SYNC=0 uv sync --frozen --extra natten --no-dev \
    && uv pip install --python /opt/LTX-2/.venv \
        jupyterlab==4.6.3 \
        ipywidgets==8.1.9 \
        psutil==7.1.3 \
        nvitop==1.7.1 \
        gpustat==1.1.1 \
    && /opt/LTX-2/.venv/bin/python -c "import ltx_core, ltx_trainer, torch; print(torch.__version__)"

COPY scripts/ /usr/local/bin/
COPY configs/a2v_personal.yaml.example /opt/ltx23-a2v/configs/a2v_personal.yaml.example

RUN chmod 0755 /usr/local/bin/start.sh \
        /usr/local/bin/bootstrap_a2v.sh \
        /usr/local/bin/init_workspace.sh \
        /usr/local/bin/check_ltx_environment.sh \
        /usr/local/bin/download_ltx_models.sh \
        /usr/local/bin/preprocess_a2v.sh \
        /usr/local/bin/train_a2v.sh \
        /usr/local/bin/resume_a2v.sh \
        /usr/local/bin/smoke_test.sh \
    && mkdir -p \
        /workspace/models/ltx-2.3 \
        /workspace/models/gemma-3-12b \
        /workspace/dataset/raw/videos \
        /workspace/dataset/preprocessed/latents \
        /workspace/dataset/preprocessed/audio_latents \
        /workspace/dataset/preprocessed/conditions \
        /workspace/validation/audio \
        /workspace/validation/reference \
        /workspace/configs \
        /workspace/outputs \
        /workspace/notebooks \
        /workspace/logs \
    && install -m 0644 /opt/ltx23-a2v/configs/a2v_personal.yaml.example \
        /workspace/configs/a2v_personal.yaml.example \
    && chmod -R a-w /opt/LTX-2

WORKDIR /workspace

VOLUME ["/workspace"]
EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python3", "-c", "import socket; s=socket.create_connection(('127.0.0.1', 8888), 3); s.close()"]

ENTRYPOINT ["/usr/local/bin/start.sh"]
