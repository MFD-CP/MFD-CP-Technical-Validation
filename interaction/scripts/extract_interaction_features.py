from __future__ import annotations
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from interaction.offline_adapter import (
    build_label_mapping,
    extract_offline_interaction_features,
    load_interaction_manifest,
    task_split_from_manifest,
)
from src.features.feature_cache import export_cached_feature_files
from src.utils.config import load_yaml
from src.utils.logger import setup_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="extract Interaction offline handcrafted features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--modalities", nargs="*", default=None)
    return parser.parse_args()

def resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def main() -> None:
    args = parse_args()
    dataset_config = load_yaml(args.config)
    input_root, manifest = load_interaction_manifest(
        dataset_config, PROJECT_ROOT
    )
    modalities = args.modalities or dataset_config.get(
        "modalities", ["audio", "accel", "gelsight"]
    )
    if "visual" in {str(value).lower() for value in modalities}:
        raise ValueError("Interaction does not use Visual modality")

    cache_root = resolve(
        dataset_config.get("features", {}).get(
            "cache_dir", "cache/interaction"
        )
    )
    export_config = dataset_config.get("feature_exports", {})
    export_root = resolve(
        export_config.get(
            "output_dir", "interaction/results/features"
        )
    )
    logger = setup_logger(
        "extract_interaction_features",
        PROJECT_ROOT / "interaction/results/logs/extract_features.log",
    )

    for task in dataset_config.get("tasks", []):
        task_name = str(task["name"])
        target_label = str(task["target_label"])
        split_df = task_split_from_manifest(
            manifest, task_name, target_label
        )
        label_to_index, _ = build_label_mapping(split_df)
        task_cache = cache_root / task_name
        cached = extract_offline_interaction_features(
            input_root=input_root,
            split_df=split_df,
            label_to_index=label_to_index,
            config=dataset_config,
            cache_dir=task_cache,
            modalities=list(modalities),
        )
        for modality, path in cached.items():
            logger.info("[%s] %s feature cache：%s", task_name, modality, path)

        if export_config.get("enabled", True):
            exported = export_cached_feature_files(
                cached_paths=cached,
                output_dir=export_root / task_name,
                config=dataset_config,
                save_csv=bool(export_config.get("save_csv", True)),
                save_hdf5=bool(export_config.get("save_hdf5", True)),
                save_parquet=bool(export_config.get("save_parquet", False)),
                save_feature_manifest=bool(
                    export_config.get("save_feature_manifest", True)
                ),
                save_config_snapshot=bool(
                    export_config.get("save_config_snapshot", True)
                ),
            )
            for name, path in exported.items():
                logger.info("[%s] official export [%s]：%s", task_name, name, path)

if __name__ == "__main__":
    main()
