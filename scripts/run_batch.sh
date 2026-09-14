#!/usr/bin/env bash
set -e

PROJECT_ROOT="\((cd "\)(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment not found at .venv"
    exit 1
fi

export PYTHONPATH="\(PROJECT_ROOT/src/wizard:\)PYTHONPATH"

echo "[*] Launching Autolab Checkpoint 2 batch execution..."
python autolab_loop.py
