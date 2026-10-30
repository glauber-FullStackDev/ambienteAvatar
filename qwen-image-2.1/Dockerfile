FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ARG DEBIAN_FRONTEND=noninteractive
# Override with a tested diffusers commit when rebuilding.
ARG DIFFUSERS_REV=main
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir "git+https://github.com/huggingface/diffusers.git@${DIFFUSERS_REV}"
RUN python -c "from diffusers import QwenImage21Pipeline"
COPY src/ ./
ENV HF_HOME=/data/huggingface PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
