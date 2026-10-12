from __future__ import annotations
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
from interaction.offline_adapter import (
    load_interaction_manifest,
    task_split_from_manifest,
)
from src.utils.config import load_yaml
from src.utils.io import write_csv
from src.utils.logger import setup_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="export Interaction fixed task split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    _, manifest = load_interaction_manifest(config, PROJECT_ROOT)
    output_dir = resolve(
        config.get("output", {}).get(
            "split_dir", "interaction/results/splits"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        "create_interaction_splits",
        PROJECT_ROOT / "interaction/results/logs/create_splits.log",
    )

    for task in config.get("tasks", []):
        task_name = str(task["name"])
        target_label = str(task["target_label"])
        split_df = task_split_from_manifest(
            manifest, task_name, target_label
        )
        expected_fabric = int(task["fabric_id"])
        actual_fabrics = sorted(
            split_df["fabric_id"].dropna().astype(int).unique()
        )
        if actual_fabrics != [expected_fabric]:
            raise ValueError(
                f"task {task_name} should use only fabric_{expected_fabric}，"
                f"actual {actual_fabrics}"
            )

        task_output_dir = output_dir / task_name
        output_path = task_output_dir / "splits.csv"
        summary_path = task_output_dir / "split_summary.csv"
        if output_path.exists() and not args.overwrite:
            existing = pd.read_csv(output_path)
            comparable = ["trial_id", "split", "fabric_id", target_label]
            if existing[comparable].equals(split_df[comparable]):
                logger.info("fixed split is already consistent，skip：%s", output_path)
                continue
            raise FileExistsError(
                f"{output_path} differs from offline manifest inconsistent；use --overwrite update"
            )

        write_csv(split_df, output_path)
        summary = (
            split_df.groupby(
                ["split", target_label, "task_label_name"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "num_trials"})
        )
        write_csv(summary, summary_path)
        logger.info(
            "fixed split saved：task=%s fabric=%s trials=%d path=%s",
            task_name,
            expected_fabric,
            len(split_df),
            output_path,
        )

if __name__ == "__main__":
    main()
