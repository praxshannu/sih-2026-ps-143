PYTHON ?= python
SERVICES := api ingest detect drift attribute intel
RUN_PY   := /Users/praxsmac/.workbuddy-ai/binaries/python/envs/default/bin/python

# Each service ships its own top-level `app` package. Running mypy over
# `services/` in one invocation makes those collide in sys.modules, so the
# type check is per-service on purpose.
MYPY_TARGETS := $(foreach s,$(SERVICES),mypy-$(s))

.PHONY: dev dev-lite build build-lite up down logs test lint format typecheck \
        clean db-init test-all $(MYPY_TARGETS) mypy-each

dev:
	docker compose -f docker-compose.txt -f docker-compose.override.yml up --build

# 16 GB Apple Silicon profile: no CUDA, no Prometheus/Grafana/Flower.
dev-lite:
	docker compose -f docker-compose.txt -f docker-compose.lite.yml up --build

lite:
	$(MAKE) dev-lite

build:
	docker compose -f docker-compose.txt build

build-lite:
	docker compose -f docker-compose.txt -f docker-compose.lite.yml build

up:
	docker compose -f docker-compose.txt up -d

down:
	docker compose -f docker-compose.txt down

logs:
	docker compose -f docker-compose.txt logs -f

test:
	$(RUN_PY) -m pytest services/ ml/ --tb=short -q

lint:
	$(RUN_PY) -m ruff check services/ ml/ scripts/
	$(RUN_PY) -m ruff format --check services/ ml/ scripts/

format:
	$(RUN_PY) -m ruff format services/ ml/ scripts/
	$(RUN_PY) -m ruff check --fix services/ ml/ scripts/

mypy-each: $(MYPY_TARGETS)

$(MYPY_TARGETS): mypy-%:
	$(RUN_PY) -m mypy services/$*

typecheck: mypy-each
	$(RUN_PY) -m mypy ml/ --ignore-missing-imports

seed:
	$(PYTHON) scripts/seed_demo_case.py

seed-ais:
	$(PYTHON) scripts/seed_ais_demo.py

clean:
	docker compose -f docker-compose.txt down -v --rmi local
	rm -rf data/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

db-init:
	docker compose exec db psql -U sentinel -d sentinel -f /docker-entrypoint-initdb.d/00_extensions.sql

test-all: lint typecheck test
