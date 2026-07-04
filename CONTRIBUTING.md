# Contributing to MedAgent

## Prerequisites

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

With uv, install [`just`](https://github.com/casey/just) with `uv tool install rust-just`.

## Setup

```bash
git clone git@github.com:alexanderwu/MedAgent.git
cd medagent
uv sync --dev
```

`uv sync` creates a `.venv` and installs exact locked dependencies from `uv.lock`. No manual venv activation needed — prefix commands with `uv run`, or activate with `source .venv/bin/activate` if you prefer.

## Common tasks

This project uses [`just`](https://github.com/casey/just) as a task runner. Run `just --list` to see all commands.

| Command | Purpose |
|---|---|
| `just install` | Sync dependencies from lockfile |
| `just lint` | Run ruff + mypy checks |
| `just fmt` | Auto-format code |
| `just notebook` | Launch Jupyter Lab |
| `just run` | Run the main pipeline |

## Project structure

```
src/medagent/
├── config.py       # paths, constants
├── data/           # load MIMIC data
└── main.py         # entry point

notebooks/          # exploration only — import logic from medagent, don't duplicate it
data/raw/           # MIMIC-IV source data
data/processed/     # pipeline outputs
```

## Code style

- Run `just fmt` before committing; `just lint` must pass before opening a PR.
- Type-hint function signatures, especially DataFrame-in/DataFrame-out functions.
- No hardcoded paths or column-name strings scattered through the codebase — add them to `config.py`.

## Notebooks

- Number notebooks by execution order: `0_EDA.ipynb`, `1_feature_check.ipynb`, etc.
- If you copy a cell into a second notebook, move that logic into `src/medagent/` instead and import it.

## Commits and PRs

- Keep commits scoped to one logical change.
- PR description should state what changed and why, and call out any data or result changes (e.g. "updated cleaning logic — row count in processed dataset changes from X to Y").
- Run `just lint && just test` locally before pushing.

## Adding dependencies

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # dev-only (linting, testing, etc.)
```

Always commit the updated `uv.lock` alongside `pyproject.toml`.
