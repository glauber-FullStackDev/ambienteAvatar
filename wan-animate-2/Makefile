.PHONY: setup build models verify up logs down

setup:
	@test -f .env || cp .env.example .env
	@mkdir -p data/cache data/input data/models data/output data/user

build: setup
	docker compose build wan-animate-2

models: setup
	docker compose --profile tools run --rm download-models

verify: setup
	docker compose --profile tools run --rm verify

up: setup
	docker compose up -d wan-animate-2

logs:
	docker compose logs -f wan-animate-2

down:
	docker compose down
