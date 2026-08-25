.PHONY: setup build models verify lipforcing-models verify-lipforcing up logs down

setup:
	@test -f .env || cp .env.example .env
	@mkdir -p data/cache data/input data/models data/output data/user

build: setup
	docker compose build comfyui

models: setup
	docker compose --profile tools run --rm download-models

verify: setup
	docker compose --profile tools run --rm verify

lipforcing-models: setup
	docker compose --profile tools run --rm download-lipforcing-models

verify-lipforcing: setup
	docker compose --profile tools run --rm verify-lipforcing

up: setup
	docker compose up -d comfyui

logs:
	docker compose logs -f comfyui

down:
	docker compose down
