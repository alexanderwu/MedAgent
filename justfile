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

# Build DuckDB from the MIMIC-IV demo subset (~19 MB, seconds)
ingest-demo:
    uv run python -m medagent.ingest --demo

# Build DuckDB from full MIMIC-IV 3.1 (long run; close the app first — DuckDB allows one writer OR many readers)
ingest:
    uv run python -m medagent.ingest --full

# Ask a question via the CLI agent harness
ask q:
    uv run python -m medagent.cli "{{q}}"

# Launch the Streamlit chat app
app:
    uv run streamlit run src/medagent/app.py
