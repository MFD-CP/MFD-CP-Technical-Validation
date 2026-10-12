from __future__ import annotations
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.data.offline_dataset import load_offline_classification_tables
from src.utils.config import load_yaml
from src.utils.io import write_csv
from src.utils.logger import setup_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    logger = setup_logger("create_splits", PROJECT_ROOT / "class/results/logs/create_splits.log")
    dataset_root, _, split_df = load_offline_classification_tables(
        cfg, PROJECT_ROOT
    )
    summary = (
        split_df.groupby(["split", "label"])
        .size()
        .reset_index(name="num_trials")
    )
    write_csv(
        summary,
        PROJECT_ROOT / "class/results/tables/split_summary.csv",
    )
    logger.info(
        "use the fixed split from offline preprocessing trial split：%s",
        dataset_root / cfg.get("offline", {}).get(
            "trial_manifest", "trial_split_manifest.csv"
        ),
    )

if __name__ == "__main__":
    main()
