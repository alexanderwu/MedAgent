# justfile
default:
    just --list

install:
    uv sync

# test:
#     uv run pytest

lint:
    uv run ruff check .
    uv run mypy src

fmt:
    uv run ruff format .

notebook:
    uv run jupyter lab

run:
    uv run python -m medagent.main
