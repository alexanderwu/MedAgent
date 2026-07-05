# justfile
default:
    just --list

# Generate git commit message idea
diff:
    git diff --staged | ollama run llama3.1:8b "You are an expert developer. Generate a short, meaningful Git commit message based on this staged diff. Output ONLY the message text without quotes, headers, or explanations."

install:
    uv sync

# test:
#     uv run pytest

# Ruff lint
lint:
    uv run ruff check .
    uv run mypy src

# Ruff format
fmt:
    uv run ruff format .

# Run Jupyter Lab
notebook:
    uv run jupyter lab

# Run medagent main.py
run:
    uv run python -m medagent.main
