#!/usr/bin/env bash
set -e

echo "[*] Cleaning dataset..."
grep '"status": "COMPLETED"' autolab_finetune_dataset.jsonl > autolab_dataset_clean.jsonl

echo "[*] Generating updated failure boundary plot..."
python plot_boundaries.py

echo "[*] Formatting SFT training and validation sets..."
python format_sft.py

echo "[*] Syncing assets to repository..."
cp autolab_dataset_clean.jsonl plot_boundaries.py format_sft.py failure_boundary.png ~/autolab-harness/

echo "[*] Post-processing complete."