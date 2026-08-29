# LTX-2.3 Personal A2V LoRA — Docker/Vast.ai

Single-image, Linux x86_64 environment for training an **LTX-2.3 Audio-to-Video LoRA** with the official [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) trainer. It does not use AI Toolkit or AI Toolkit-derived code.

The safety-critical training mode is:

```yaml
model:
  training_mode: lora
training_strategy:
  name: flexible
  video:
    is_generated: true
  audio:
    is_generated: false
```

The current official schema calls this section `training_strategy` (not `strategy`). Audio stays clean/frozen conditioning; only video receives diffusion noise and loss. `train_a2v.sh` refuses to start if these semantics change.

## Reproducibility record

Inspected and pinned on 2026-08-28:

| Component | Pinned value |
| --- | --- |
| LTX repository | `Lightricks/LTX-2` |
| LTX commit | `a95ab856bf29407b6b066ede0abe1846050db56c` |
| Upstream release at commit | `1.3.0` |
| Python | 3.12 from Ubuntu 24.04; upstream supports `>=3.10` |
| CUDA base | `nvidia/cuda:13.2.1-devel-ubuntu24.04` |
| linux/amd64 base digest | `sha256:0e1f7b8e96fa9ec5e36d4709a38c62df7b5665977446081811c12b8234d874bf` |
| uv | `0.11.7` via the official Astral installer |
| PyTorch selected by LTX NATTEN extra | `2.13.0+cu132` |
| NATTEN | `0.21.7+torch2130cu132` |
| cuDNN Python wheel override | `9.24.0.43` |
| Generated lock SHA-256 | `50119955fb24eec232ab0a3ca06ac115f7f16ef765d325171bd123663474c34f` |

The pinned upstream commit does **not** contain an official `uv.lock`; this was confirmed through the GitHub contents API. To prevent dependency drift, [`locks/uv.lock`](locks/uv.lock) was resolved from that exact commit's unmodified workspace `pyproject.toml` files with uv 0.11.7 and Python 3.12. The Docker build installs it with:

```bash
uv sync --frozen --extra natten --no-dev
```

The base intentionally uses CUDA `devel` without system cuDNN. PyTorch's locked cu132 wheels provide the upstream-selected cuDNN and avoid mixing a different system cuDNN into the process. The tag and linux/amd64 digest were verified through Docker Hub's official `nvidia/cuda` catalog.

Because CUDA 13.2 is current upstream policy, the Vast.ai host needs an NVIDIA driver new enough to run CUDA 13.2 containers.

## Build

From this directory:

```bash
docker build --platform linux/amd64 -t <IMAGE_NAME>:<TAG> .
```

The source tree is installed at `/opt/LTX-2`, detached at the pinned SHA, and marked read-only. All mutable data belongs under `/workspace`.

No model weights, datasets, tokens, `.env` files, SSH keys, checkpoints, or outputs are copied into the image.

## Local GPU test

```bash
mkdir -p workspace

docker run --rm --gpus all \
  -p 8888:8888 \
  -e JUPYTER_TOKEN=test-token \
  -v "$(pwd)/workspace:/workspace" \
  <IMAGE_NAME>:<TAG>
```

For a shell instead of Jupyter:

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)/workspace:/workspace" \
  <IMAGE_NAME>:<TAG> bash
```

Then run:

```bash
smoke_test.sh
check_ltx_environment.sh
```

`smoke_test.sh` never downloads models. The environment check returns nonzero for blockers such as no GPU or missing model weights, while an absent raw dataset is a warning because it may be uploaded later.

## Runtime variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `HF_TOKEN` | Hugging Face read token; required only for model download | unset |
| `WANDB_API_KEY` | W&B credential; required when W&B is enabled | unset |
| `WANDB_PROJECT` | Effective W&B project | `ltx23-personal-a2v` |
| `JUPYTER_TOKEN` | Jupyter authentication token | randomly generated at startup |
| `LTX_MODELS_DIR` | Model root | `/workspace/models` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | NVIDIA runtime default |
| `LTX_RESOLUTION_BUCKET` | Preprocessing bucket in task-friendly `FxHxW` order | required |
| `LTX_FRAME_COUNT` | Required frames in each bucket | `89` |
| `LTX_LORA_TRIGGER` | Optional unique token prepended to every training caption during preprocessing | unset |
| `LTX_VALIDATION_INTERVAL` | Optional runtime override of YAML validation interval | YAML value |
| `LTX_CHECKPOINT_INTERVAL` | Optional runtime override of YAML checkpoint interval | YAML value |
| `LTX_RESUME_CHECKPOINT` | Explicit LoRA `.safetensors` to resume | unset |
| `AUTO_DOWNLOAD_MODELS` | Download/verify model weights before Jupyter starts | `0` |

Secrets are read only from the runtime environment. The scripts never print `HF_TOKEN` or `WANDB_API_KEY`, never persist them, and never use them as build arguments.

## Persistent workspace

The entrypoint idempotently creates this layout on every start and seeds the example config when missing:

```text
/workspace/
├── models/
│   ├── ltx-2.3/
│   └── gemma-3-12b/
├── dataset/
│   ├── raw/
│   │   ├── videos/
│   │   └── dataset.json
│   └── preprocessed/
│       ├── latents/
│       ├── audio_latents/
│       └── conditions/
├── validation/
│   ├── audio/
│   └── reference/
├── configs/
├── outputs/
├── notebooks/
└── logs/
```

Mount persistent Vast.ai storage at `/workspace`. The Dockerfile declares it as a volume.

## JupyterLab

JupyterLab listens on `0.0.0.0:8888`, uses `/workspace` as its root, and allows root inside the container. Authentication is never disabled.

If `JUPYTER_TOKEN` is present, its value is used but not printed. Otherwise the entrypoint generates a strong random token and prints it once in the startup log.

## Download models

Inside the running container:

```bash
export HF_TOKEN=hf_your_read_token
download_ltx_models.sh
unset HF_TOKEN
```

The explicit, idempotent download fetches:

- `Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors` into `/workspace/models/ltx-2.3/`
- all files from `google/gemma-3-12b-it-qat-q4_0-unquantized` into `/workspace/models/gemma-3-12b/`

By default, models are never downloaded at container startup. On HTTP 401/403, accept the LTX/Gemma model terms on Hugging Face and ensure the token has read access to gated repositories.

For a managed Vast.ai instance, the included [Vast template](vast/ltx23-a2v-trainer-template.json) deliberately sets `AUTO_DOWNLOAD_MODELS=1`. This is opt-in and requires `HF_TOKEN`; without a token, startup stops with a clear error instead of opening an unusable Jupyter session.

## Dataset

Upload without recompressing or modifying the MP4 files:

```text
/workspace/dataset/raw/
├── dataset.json
└── videos/
    ├── clip_0001.mp4
    └── ...
```

The official trainer accepts `media_path` as the legacy alias for the video column:

```json
[
  {
    "media_path": "videos/clip_0001.mp4",
    "caption": "..."
  }
]
```

Embedded audio is automatically extracted by the official preprocessing code. The wrapper does not pass `--skip-audio` and does not alter the source video.

## Preprocess

Choose the spatial bucket explicitly. The wrapper accepts the requested `frames x height x width` order and translates it to the current official CLI's `width x height x frames` order:

```bash
export LTX_RESOLUTION_BUCKET=89x448x768
preprocess_a2v.sh
```

Other examples:

```bash
export LTX_RESOLUTION_BUCKET=89x512x768
export LTX_RESOLUTION_BUCKET=89x544x960
```

Multiple buckets use semicolons:

```bash
export LTX_RESOLUTION_BUCKET='89x448x768;89x544x960'
```

Height and width must be divisible by 32. The default frame count remains 89 and must satisfy the LTX VAE temporal rule `frames % 8 == 1`. Optional controls:

```bash
export LTX_PREPROCESS_BATCH_SIZE=1
export LTX_VAE_TILING=1          # optional for memory pressure
export LTX_PREPROCESS_OVERWRITE=1 # recompute existing artifacts
preprocess_a2v.sh
```

### Trigger token (recommended for a personal LoRA)

Choose one unique, simple token before preprocessing, for example `glauberavatar`. Do not use a common word or your full name. Apply it consistently:

```bash
export LTX_LORA_TRIGGER=glauberavatar
export LTX_RESOLUTION_BUCKET=89x448x768
preprocess_a2v.sh
```

The official preprocess script prepends that token to every caption before computing text conditions. Put the exact same token at the beginning of every validation or inference prompt:

```yaml
prompt: >-
  glauberavatar, speaking directly to camera, medium shot, soft studio lighting,
  natural expression and accurate lip synchronization.
```

If preprocessing already ran without the trigger, rerun it with `LTX_PREPROCESS_OVERWRITE=1`; otherwise the cached text conditions still lack the token.

The exact official command is printed before execution. Successful preprocessing is verified to contain non-empty `.pt` files in `latents/`, `audio_latents/`, and `conditions/`.

## Configure training and validation

```bash
cp /workspace/configs/a2v_personal.yaml.example \
  /workspace/configs/a2v_personal.yaml
nano /workspace/configs/a2v_personal.yaml
```

Review every `TUNE ME` marker. Learning rate, steps, LoRA rank, quantization, and spatial resolution are deliberately examples, not finalized recommendations.

Place validation audio at the path used in `validation.samples[].conditions[].audio`, for example:

```text
/workspace/validation/audio/example.wav
```

The template generates video only at 25 FPS from frozen validation audio every 300 steps, writes samples under `/workspace/outputs/personal_a2v`, and logs validation videos to W&B. Edit the YAML or override the interval:

```bash
export LTX_VALIDATION_INTERVAL=300
export LTX_CHECKPOINT_INTERVAL=300
```

Checkpoints keep the latest five weights/state pairs and save full optimizer/scheduler/RNG/step/W&B state. Increase `keep_last_n` only after considering persistent disk use.

## W&B

The example config enables W&B. Set credentials only at runtime:

```bash
export WANDB_API_KEY=your_key
export WANDB_PROJECT=ltx23-personal-a2v
```

`train_a2v.sh` applies `WANDB_PROJECT` to an effective config in `/workspace/logs` and fails clearly if W&B is enabled without `WANDB_API_KEY`.

## Training

Run the environment check first if desired:

```bash
check_ltx_environment.sh
```

Launch inside `tmux` so a browser or SSH disconnect does not stop training:

```bash
tmux new -s ltx
train_a2v.sh
```

Detach with `Ctrl-b d` and reattach with:

```bash
tmux attach -t ltx
```

The script immediately launches when valid; there is no interactive confirmation. It checks the GPU, models, preprocessed artifacts and A2V semantics, prints GPU VRAM and the effective config, then runs:

```bash
cd /opt/LTX-2/packages/ltx-trainer
uv run --frozen --no-sync python scripts/train.py /workspace/logs/effective_a2v_config.yaml
```

The official trainer creates:

```text
/workspace/outputs/personal_a2v/
├── checkpoints/
├── samples/
└── training_config.yaml
```

## Resume

Inspect the latest checkpoint/state pair without launching anything:

```bash
resume_a2v.sh
```

For a different explicit run directory:

```bash
resume_a2v.sh /workspace/outputs/personal_a2v
```

The script prints the exact command to resume. Review it, then run the explicit file path it reports, for example:

```bash
LTX_RESUME_CHECKPOINT=/workspace/outputs/personal_a2v/checkpoints/lora_weights_step_00300.safetensors \
  train_a2v.sh
```

Directories are rejected by `train_a2v.sh` so it cannot silently select a checkpoint from the wrong run. A matching `training_state_step_00300.pt` is required for true optimizer/scheduler/step restoration.

## Vast.ai lifecycle

1. Push to `main` or run the `Build and publish LTX-2.3 A2V trainer image` workflow manually. It publishes `ghcr.io/glauber-fullstackdev/ambienteavatar-ltx23-a2v:vast`.
2. Create a Vast.ai GPU instance with persistent storage mounted at `/workspace` and port 8888 exposed.
3. Open JupyterLab using the runtime or generated token.
4. Upload `dataset.json` and MP4 files below `/workspace/dataset/raw/`.
5. With the included Vast template, set `HF_TOKEN` before startup and model download runs automatically. For a manual container, set `HF_TOKEN`, run `download_ltx_models.sh`, then unset it.
6. Run `check_ltx_environment.sh`.
7. Set `LTX_RESOLUTION_BUCKET` and run `preprocess_a2v.sh`.
8. Copy/edit `a2v_personal.yaml.example` to `a2v_personal.yaml`.
9. Set `WANDB_API_KEY` and optionally `WANDB_PROJECT`.
10. Start `train_a2v.sh` inside `tmux`.
11. Monitor checkpoints/samples in persistent storage and validation media/metrics in W&B.

No Python package installation is required after startup.
