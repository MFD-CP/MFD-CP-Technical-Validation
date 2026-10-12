#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
PYTHON_BIN="${PYTHON:-python3}"
DATASET_CONFIG="interaction/configs/dataset/interaction_dataset.yaml"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/fabric_interaction_numba_cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fabric_interaction_matplotlib_cache}"

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
mkdir -p \
  interaction/results/logs \
  interaction/results/tables \
  "$NUMBA_CACHE_DIR" \
  "$MPLCONFIGDIR"
export NUMBA_CACHE_DIR MPLCONFIGDIR

echo "============================================"
echo "Step 0: Check Interaction offline preprocessed files"
echo "============================================"
missing_files=()
for split in train val test; do
  for path in \
    "$PREPROCESSED_ROOT/temporal_${split}.h5" \
    "$PREPROCESSED_ROOT/gelsight_${split}.pt"; do
    [[ -f "$path" ]] || missing_files+=("$path")
  done
done
manifest="$PREPROCESSED_ROOT/trial_split_manifest.csv"
[[ -f "$manifest" ]] || missing_files+=("$manifest")
if (( ${#missing_files[@]} > 0 )); then
  echo "Interaction offline preprocessing output is incomplete; missing:" >&2
  printf '  %s\n' "${missing_files[@]}" >&2
  exit 1
fi

echo "============================================"
echo "Step 1: Export fixed Interaction task splits"
echo "============================================"
"${PYTHON_BIN}" interaction/scripts/create_interaction_splits.py \
  --config "$DATASET_CONFIG" \
  --overwrite

echo ""
echo "============================================"
echo "Step 2: Extract handcrafted features"
echo "============================================"
"${PYTHON_BIN}" interaction/scripts/extract_interaction_features.py \
  --config "$DATASET_CONFIG"

echo ""
echo "============================================"
echo "Step 3: Train ML models (XGBoost)"
echo "============================================"

echo "--- Task 1: Interaction primitive recognition (fixed trial split) ---"
"${PYTHON_BIN}" interaction/scripts/train_interaction_ml.py \
  --config interaction/configs/ml/primitive_xgboost.yaml

echo "--- Task 2: Press-angle recognition (fixed trial split) ---"
"${PYTHON_BIN}" interaction/scripts/train_interaction_ml.py \
  --config interaction/configs/ml/press_angle_within_fabric_xgboost.yaml

echo "--- Task 3: Slide-trajectory recognition (fixed trial split) ---"
"${PYTHON_BIN}" interaction/scripts/train_interaction_ml.py \
  --config interaction/configs/ml/slide_trajectory_within_fabric_xgboost.yaml

echo "--- Task 4: Twist-position recognition (fixed trial split) ---"
"${PYTHON_BIN}" interaction/scripts/train_interaction_ml.py \
  --config interaction/configs/ml/twist_position_within_fabric_xgboost.yaml

echo ""
echo "============================================"
echo "Step 4: Train DL models (LSTM + EfficientNet-B0 Late Concat)"
echo "============================================"

echo "--- Task 1: Interaction primitive recognition (fixed trial split) ---"
"${PYTHON_BIN}" interaction/scripts/train_interaction_deep.py \
  --config interaction/configs/deep/primitive_late_concat_lstm_efficientnet.yaml

echo "--- Task 2: Press-angle recognition (fixed trial split) ---"
"${PYTHON_BIN}" interaction/scripts/train_interaction_deep.py \
  --config interaction/configs/deep/press_angle_within_fabric_late_concat_lstm_efficientnet.yaml

echo "--- Task 3: Slide-trajectory recognition (fixed trial split) ---"
"${PYTHON_BIN}" interaction/scripts/train_interaction_deep.py \
  --config interaction/configs/deep/slide_trajectory_within_fabric_late_concat_lstm_efficientnet.yaml

echo "--- Task 4: Twist-position recognition (fixed trial split) ---"
"${PYTHON_BIN}" interaction/scripts/train_interaction_deep.py \
  --config interaction/configs/deep/twist_position_within_fabric_late_concat_lstm_efficientnet.yaml

echo ""
echo "============================================"
echo "Step 5: Generate official task-stratified reports and confusion matrices"
echo "============================================"
"${PYTHON_BIN}" \
  interaction/scripts/create_interaction_experiment_reports.py

echo ""
echo "============================================"
echo "All interaction recognition experiments completed!"
echo "============================================"
