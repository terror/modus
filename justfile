set dotenv-load

export EDITOR := 'nvim'

default:
  just --list

fmt:
  ruff check --select I --fix
  ruff format

run:
  uv run main.py
