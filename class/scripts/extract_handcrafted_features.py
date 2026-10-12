from __future__ import annotations
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.features.feature_cache import (
    export_cached_feature_files,
    extract_offline_and_cache_features,
)
from src.data.offline_dataset import resolve_offline_root
from src.utils.config import load_yaml
from src.utils.logger import setup_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--modalities", nargs="*", default=None)
    return parser.parse_args()

def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    logger = setup_logger("extract_features", PROJECT_ROOT / "class/results/logs/extract_features.log")

    cache_dir = _resolve_project_path(cfg.get("features", {}).get("cache_dir", "cache/class"))
    dataset_root = resolve_offline_root(cfg, PROJECT_ROOT)
    cached_outputs = extract_offline_and_cache_features(
        input_root=dataset_root,
        config=cfg,
        cache_dir=cache_dir,
        modalities=args.modalities,
    )

    for modality, path in cached_outputs.items():
        logger.info("%s feature training cache saved: %s", modality, path)
        hdf5_path = path.with_suffix(".h5")
        if hdf5_path.exists():
            logger.info(
                "%s feature HDF5 cache saved: %s",
                modality,
                hdf5_path,
            )

    export_cfg = cfg.get("feature_exports", {})
    if export_cfg.get("enabled", True):
        export_dir = _resolve_project_path(export_cfg.get("output_dir", "class/results/features"))

        exported_outputs = export_cached_feature_files(
            cached_paths=cached_outputs,
            output_dir=export_dir,
            config=cfg,
            save_csv=bool(export_cfg.get("save_csv", True)),
            save_hdf5=bool(export_cfg.get("save_hdf5", True)),
            save_parquet=bool(export_cfg.get("save_parquet", False)),
            save_feature_manifest=bool(export_cfg.get("save_feature_manifest", True)),
            save_config_snapshot=bool(export_cfg.get("save_config_snapshot", True)),
        )

        for name, path in exported_outputs.items():
            logger.info("official handcrafted-feature output saved [%s]: %s", name, path)

if __name__ == "__main__":
    main()
