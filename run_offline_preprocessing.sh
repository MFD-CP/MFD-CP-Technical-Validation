#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
CLASS_CONFIG="class/configs/dataset/dataset.yaml"
INTERACTION_CONFIG="interaction/configs/dataset/interaction_dataset.yaml"

usage() {
  echo "Usage: $0 DATASET_ROOT CLASSIFICATION_ROOT [--dry-run] [--overwrite]"
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

DATASET_ROOT="$1"
CLASSIFICATION_ROOT="$2"
shift 2
cd "$ROOT"

read_output_root() {
  "${PYTHON_BIN}" -c '
from pathlib import Path
import sys
from src.utils.config import load_yaml

path = Path(load_yaml(sys.argv[1])["preprocessed_root"])
print(path if path.is_absolute() else Path.cwd() / path)
' "$1"
}

CLASS_OUTPUT="$(read_output_root "$CLASS_CONFIG")"
INTERACTION_OUTPUT="$(read_output_root "$INTERACTION_CONFIG")"

echo "[1/3] Class temporal and GelSight"
"${PYTHON_BIN}" preprocess_offline/preprocess_class_temporal_gelsight.py \
  --config "$CLASS_CONFIG" --dataset-root "$DATASET_ROOT" \
  --output-root "$CLASS_OUTPUT" "$@"

echo "[2/3] Class visual"
"${PYTHON_BIN}" preprocess_offline/preprocess_visual.py \
  --config "$CLASS_CONFIG" --classification-root "$CLASSIFICATION_ROOT" \
  --output-root "$CLASS_OUTPUT" "$@"

echo "[3/3] Interaction temporal and GelSight"
"${PYTHON_BIN}" preprocess_offline/preprocess_interaction_temporal_gelsight.py \
  --config "$INTERACTION_CONFIG" --dataset-root "$DATASET_ROOT" \
  --output-root "$INTERACTION_OUTPUT" "$@"

echo "Offline data preparation completed."
