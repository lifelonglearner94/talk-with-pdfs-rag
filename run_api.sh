#!/usr/bin/env bash
# Lightweight runner for the FastAPI app used during development.
# Usage: ./run_api.sh

export PYTHONPATH="${PYTHONPATH:-.}"
uvicorn interfaces.api:app --host 127.0.0.1 --port 8000 --reload
