.PHONY: dev build test lint typecheck clean seed demo down logs

dev:
	docker compose -f docker-compose.txt -f docker-compose.override.yml up --build

build:
	docker compose -f docker-compose.txt build

up:
	docker compose -f docker-compose.txt up -d

down:
	docker compose -f docker-compose.txt down

logs:
	docker compose -f docker-compose.txt logs -f

test:
	pytest services/ --tb=short -q

lint:
	ruff check services/
	ruff format --check services/

format:
	ruff format services/
	ruff check --fix services/

typecheck:
	mypy services/ --ignore-missing-imports

seed:
	python scripts/seed_demo_case.py

seed-ais:
	python scripts/seed_ais_demo.py

clean:
	docker compose -f docker-compose.txt down -v --rmi local
	rm -rf data/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

db-init:
	docker compose exec db psql -U sentinel -d sentinel -f /docker-entrypoint-initdb.d/00_extensions.sql

test-all: lint typecheck test
