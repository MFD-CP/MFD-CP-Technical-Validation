# MFD-CP: Technical Validation

This repository provides offline preprocessing and reproducible experimental pipelines for the technical validation of the Multimodal Fabric Dataset under Controlled Perception (MFD-CP), covering multimodal fabric classification and interaction recognition.

## Installation

```bash
conda create -n fabric-mm python=3.10 -y
conda activate fabric-mm
pip install -r requirements.txt
```

Run all commands from the repository root.

## Data preparation

Expected raw-data directories:

```text
DATA_ROOT/
├── robotic_exploration/
└── multi_view_imaging/
```

Preprocessed output directories are set by `preprocessed_root` in:

- `class/configs/dataset/dataset.yaml` (default: `preprocessed_data/class`)
- `interaction/configs/dataset/interaction_dataset.yaml` (default: `preprocessed_data/interaction`)

Validate the raw data without writing outputs:

```bash
bash run_offline_preprocessing.sh \
  /path/to/MFD-CP/robotic_exploration \
  /path/to/MFD-CP/multi_view_imaging \
  --dry-run
```

Rerun without `--dry-run` to generate the offline datasets. Use `--overwrite` only to replace existing outputs.

## Experiments

Experiment settings are defined in `class/configs/` and `interaction/configs/`.

```bash
bash run_classification_experiments.sh
bash run_interaction_experiments.sh
```

The scripts use `python3` from the active environment. Set `PYTHON=/path/to/python` to use another interpreter.
