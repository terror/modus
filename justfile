set dotenv-load

export EDITOR := 'nvim'

default:
  just --list

fmt:
  ruff check --select I --fix
  ruff format

lint:
  ruff check

run:
  uv run logic.py

test:
  uv run pytest
