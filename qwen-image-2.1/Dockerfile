FROM --platform=linux/amd64 ubuntu:24.04 AS llama-build
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libcurl4t64 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/llama.tar.gz https://github.com/ggml-org/llama.cpp/releases/download/b11160/llama-b11160-bin-ubuntu-cuda-12.8-x64.tar.gz \
    && echo '8ed8d659383b6624ef2bd14ebd6984c8c1b1b2f13e4ba0c5c43f6038ca22591d  /tmp/llama.tar.gz' | sha256sum -c - \
    && mkdir -p /opt/llama/bin \
    && tar -xzf /tmp/llama.tar.gz -C /opt/llama/bin --strip-components=1 \
    && rm /tmp/llama.tar.gz

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ARG DEBIAN_FRONTEND=noninteractive
# Override with a tested diffusers commit when rebuilding.
ARG DIFFUSERS_REV=main
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir "git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_REV}"
RUN python -c "from diffusers import QwenImage21Pipeline"
COPY --from=llama-build /opt/llama/bin /opt/llama/bin
# O binário oficial é compilado para glibc Ubuntu 24.04; isole seu loader e
# bibliotecas sem alterar a glibc do PyTorch (Ubuntu 22.04).
COPY --from=llama-build /lib/x86_64-linux-gnu /opt/llama/runtime
RUN printf '%s\n' '#!/bin/sh' 'cd /opt/llama/bin || exit 1' 'exec /opt/llama/runtime/ld-linux-x86-64.so.2 --library-path /opt/llama/bin:/opt/llama/runtime:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu /opt/llama/bin/llama-server "$@"' > /opt/llama/llama-server \
    && chmod +x /opt/llama/llama-server \
    && /opt/llama/llama-server --version \
    && (/opt/llama/llama-server -m /tmp/modelo-inexistente.gguf -ngl 0 --port 0 -lv 4 > /tmp/llama-smoke.log 2>&1 || true) \
    && cat /tmp/llama-smoke.log \
    && grep -q 'failed to open GGUF file' /tmp/llama-smoke.log \
    && ! grep -q 'no backends are loaded' /tmp/llama-smoke.log
COPY src/ ./
ENV HF_HOME=/data/huggingface PYTHONUNBUFFERED=1 LLAMA_SERVER_PATH=/opt/llama/llama-server
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
