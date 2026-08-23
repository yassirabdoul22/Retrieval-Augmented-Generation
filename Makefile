.PHONY: install run debug clean lint lint-strict

MAX_CHUNK_SIZE ?= 2000

install:
	uv sync

run:
	uv run python3 -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

debug:
	uv run python3 -m pdb -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
