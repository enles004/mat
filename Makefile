.PHONY: sync test lint typecheck verify

sync:
	cd ai && uv sync --locked --extra transformer

test:
	cd ai && uv run --locked pytest -q

lint:
	cd ai && uv run --locked ruff check src scripts tests

typecheck:
	cd ai && uv run --locked mypy

verify: lint typecheck test
