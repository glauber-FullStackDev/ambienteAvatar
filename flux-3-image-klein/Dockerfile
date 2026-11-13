FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime
ARG DEBIAN_FRONTEND=noninteractive
ARG COMFY_REV=v0.41.3
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${COMFY_REV}" https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r /opt/ComfyUI/requirements.txt -r requirements.txt
COPY src/ ./
ENV HF_HOME=/data/huggingface PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
