#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
PYTHON_BIN="${PYTHON:-python3}"
DATASET_CONFIG="class/configs/dataset/dataset.yaml"

cd "$PROJECT_ROOT"
PREPROCESSED_ROOT="$(
  "${PYTHON_BIN}" -c '
from pathlib import Path
import sys
import yaml

project_root = Path(sys.argv[1])
config_path = project_root / sys.argv[2]
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
path = Path(config["preprocessed_root"])
print(path if path.is_absolute() else project_root / path)
' "$PROJECT_ROOT" "$DATASET_CONFIG"
)"

echo "============================================================"
echo "Step 1: Create classification splits"
echo "============================================================"
if [[ -f class/results/tables/split_summary.csv \
  && -f "$PREPROCESSED_ROOT/trial_split_manifest.csv" ]]; then
  echo "Skip Step 1: existing fixed trial split found."
else
  "${PYTHON_BIN}" class/scripts/create_splits.py \
    --config "$DATASET_CONFIG"
fi

echo "============================================================"
echo "Step 2: Extract handcrafted features"
echo "============================================================"
if [[ -f class/results/features/audio_handcrafted_features.csv \
  && -f class/results/features/audio_handcrafted_features.h5 \
  && -f class/results/features/accel_handcrafted_features.csv \
  && -f class/results/features/accel_handcrafted_features.h5 \
  && -f class/results/features/gelsight_handcrafted_features.csv \
  && -f class/results/features/gelsight_handcrafted_features.h5 \
  && -f class/results/features/visual_handcrafted_features.csv \
  && -f class/results/features/visual_handcrafted_features.h5 \
  && -f cache/class/audio_features.csv \
  && -f cache/class/audio_features.h5 \
  && -f cache/class/accel_features.csv \
  && -f cache/class/accel_features.h5 \
  && -f cache/class/gelsight_features.csv \
  && -f cache/class/gelsight_features.h5 \
  && -f cache/class/visual_features.csv \
  && -f cache/class/visual_features.h5 ]]; then
  echo "Skip Step 2: existing handcrafted feature exports found."
else
  "${PYTHON_BIN}" class/scripts/extract_handcrafted_features.py \
    --config "$DATASET_CONFIG"
fi

echo "============================================================"
echo "Step 3: Train ML single-modal classification models"
echo "============================================================"
"${PYTHON_BIN}" class/scripts/train_ml_single_modal.py \
  --config class/configs/ml/single_modal.yaml

echo "============================================================"
echo "Step 4: Train ML multimodal XGBoost classification model"
echo "============================================================"
"${PYTHON_BIN}" class/scripts/train_ml_multimodal.py \
  --config class/configs/ml/multimodal_xgboost.yaml

echo "============================================================"
echo "Step 5: Train deep single-modal classification models"
echo "============================================================"
"${PYTHON_BIN}" class/scripts/train_deep_single_modal.py \
  --config class/configs/deep/single_modal_all_encoders.yaml

echo "============================================================"
echo "Step 6: Train deep multimodal classification model"
echo "============================================================"
"${PYTHON_BIN}" class/scripts/train_deep_multimodal.py \
  --config class/configs/deep/multimodal_late_concat.yaml

echo "============================================================"
echo "Step 7: Create five-seed reports and per-experiment confusion matrices"
echo "============================================================"
"${PYTHON_BIN}" class/scripts/create_experiment_reports.py \
  --num-workers 8

echo "============================================================"
echo "All classification experiments finished."
echo "============================================================"
