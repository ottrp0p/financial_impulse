#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
mkdir -p web/data
exec .venv/bin/uvicorn web.backend.app:app --port "${PORT:-8765}" --reload
