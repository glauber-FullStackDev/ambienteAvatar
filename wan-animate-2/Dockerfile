ARG PYTORCH_DEVEL_IMAGE=pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel
ARG PYTORCH_RUNTIME_IMAGE=pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

FROM ${PYTORCH_DEVEL_IMAGE} AS flash-attn-builder

ARG FLASH_ATTN_VERSION=2.8.3

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && pip install packaging psutil wheel \
    && MAX_JOBS=4 pip wheel --no-build-isolation --no-deps \
        "flash-attn==${FLASH_ATTN_VERSION}" --wheel-dir /wheels

FROM ${PYTORCH_RUNTIME_IMAGE}

ARG DIFFUSERS_REF=360bef807475899c2e4d7d99c2f371148a78b1a7

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/cache/huggingface \
    MODEL_ROOT=/models \
    OUTPUT_DIR=/outputs

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=flash-attn-builder /wheels /tmp/wheels

RUN pip install /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels \
    && git clone https://github.com/huggingface/diffusers.git /opt/diffusers \
    && cd /opt/diffusers \
    && git checkout --detach "${DIFFUSERS_REF}" \
    && pip install . \
    && pip install \
        "accelerate==1.13.0" \
        "av==12.0.0" \
        "decord==0.6.0" \
        "gradio==6.4.0" \
        "huggingface_hub[hf_xet]==0.36.0" \
        "imageio==2.37.2" \
        "imageio-ffmpeg==0.6.0" \
        "opencv-python-headless==4.13.0.90" \
        "peft==0.19.1" \
        "protobuf==6.33.6" \
        "sentencepiece" \
        "transformers==4.57.6"

COPY app.py /opt/wan-animate-2/app.py
COPY scripts /opt/wan-animate-2/scripts

RUN chmod +x /opt/wan-animate-2/scripts/entrypoint.py \
    && mkdir -p /data/cache/huggingface /models /outputs \
    && python -m compileall -q /opt/wan-animate-2 \
    && python /opt/wan-animate-2/scripts/check_dependencies.py

WORKDIR /opt/wan-animate-2
EXPOSE 8188

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8188/', timeout=3)" || exit 1

ENTRYPOINT ["python", "/opt/wan-animate-2/scripts/entrypoint.py"]
CMD ["serve"]
