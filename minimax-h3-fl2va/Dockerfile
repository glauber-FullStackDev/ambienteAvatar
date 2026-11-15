FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG COMFY_REV=4a9636acd031525d37b1023af36085313b1a1543
ARG GGUF_REV=6ea2651e7df66d7585f6ffee804b20e92fb38b8a
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates ffmpeg && rm -rf /var/lib/apt/lists/*
RUN git init /opt/ComfyUI && git -C /opt/ComfyUI remote add origin https://github.com/Comfy-Org/ComfyUI.git && git -C /opt/ComfyUI fetch --depth 1 origin "$COMFY_REV" && git -C /opt/ComfyUI checkout --detach FETCH_HEAD
RUN git init /opt/ComfyUI/custom_nodes/ComfyUI-GGUF && git -C /opt/ComfyUI/custom_nodes/ComfyUI-GGUF remote add origin https://github.com/city96/ComfyUI-GGUF.git && git -C /opt/ComfyUI/custom_nodes/ComfyUI-GGUF fetch --depth 1 origin "$GGUF_REV" && git -C /opt/ComfyUI/custom_nodes/ComfyUI-GGUF checkout --detach FETCH_HEAD
COPY gguf-minimax-h3.patch /tmp/gguf-minimax-h3.patch
RUN git -C /opt/ComfyUI/custom_nodes/ComfyUI-GGUF apply --check /tmp/gguf-minimax-h3.patch && git -C /opt/ComfyUI/custom_nodes/ComfyUI-GGUF apply /tmp/gguf-minimax-h3.patch
RUN pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt -r /opt/ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt 'huggingface_hub[cli]>=0.30,<2'
COPY h3_first_last_audio*_api.json /opt/pipeline/
COPY workflows-video/ /opt/pipeline/workflows-video/
COPY executar.py baixar_modelos.py /opt/pipeline/
WORKDIR /opt/ComfyUI
ENV PYTHONUNBUFFERED=1 HF_HOME=/data/cache/huggingface
EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--lowvram", "--disable-smart-memory", "--disable-pinned-memory", "--disable-dynamic-vram"]
