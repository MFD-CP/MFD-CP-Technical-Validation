from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from src.features.accel_features import extract_accel_features
from src.features.audio_features import extract_audio_features
from src.features.image_features import extract_gelsight_features, extract_visual_features
from src.data.offline_dataset import (
    OfflineClassificationDataset,
    load_offline_trial_tables,
)

BASE_COLUMNS = [
    "trial_id",
    "window_id",
    "window_index",
    "window_start",
    "window_end",
    "start_sec",
    "end_sec",
    "fabric_id",
    "split",
    "label",
    "session_id",
    "operator_id",
]

def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in BASE_COLUMNS]

def load_feature_table(cache_dir: str | Path, modality: str) -> pd.DataFrame:
    cache_path = Path(cache_dir)
    csv_path = cache_path / f"{modality}_features.csv"
    hdf5_path = cache_path / f"{modality}_features.h5"
    parquet_path = cache_path / f"{modality}_features.parquet"

    if csv_path.exists():
        return pd.read_csv(csv_path)

    if hdf5_path.exists():
        return load_feature_hdf5(hdf5_path)

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(
        f"feature cache not found：{csv_path}、{hdf5_path} or {parquet_path}"
    )

def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "saving handcrafted features HDF5 requires h5py，run first pip install h5py。"
        ) from error
    return h5py

def save_feature_hdf5(df: pd.DataFrame, path: str | Path) -> Path:
    h5py = _import_h5py()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    base_cols = [
        column for column in BASE_COLUMNS if column in df.columns
    ]
    feat_cols = feature_columns(df)

    try:
        with h5py.File(temporary, "w") as file:
            file.attrs["format"] = "fabric_handcrafted_features"
            file.attrs["schema_version"] = 1
            file.attrs["num_rows"] = len(df)
            file.attrs["column_order"] = json.dumps(
                list(df.columns), ensure_ascii=False
            )
            file.attrs["metadata_columns"] = json.dumps(
                base_cols, ensure_ascii=False
            )
            file.attrs["feature_columns"] = json.dumps(
                feat_cols, ensure_ascii=False
            )
            metadata_group = file.create_group("metadata")
            features_group = file.create_group("features")

            for column in df.columns:
                group = (
                    metadata_group
                    if column in BASE_COLUMNS
                    else features_group
                )
                series = df[column]
                if pd.api.types.is_numeric_dtype(series.dtype):
                    group.create_dataset(
                        column,
                        data=series.to_numpy(),
                        compression="gzip",
                        compression_opts=4,
                    )
                else:
                    values = (
                        series.fillna("").astype(str).to_numpy(dtype=object)
                    )
                    group.create_dataset(
                        column,
                        data=values,
                        dtype=h5py.string_dtype(encoding="utf-8"),
                        compression="gzip",
                        compression_opts=4,
                    )
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output_path

def load_feature_hdf5(path: str | Path) -> pd.DataFrame:
    h5py = _import_h5py()
    input_path = Path(path)
    with h5py.File(input_path, "r") as file:
        format_name = file.attrs.get("format", "")
        if isinstance(format_name, bytes):
            format_name = format_name.decode("utf-8")
        if format_name != "fabric_handcrafted_features":
            raise ValueError(f"is not a project handcrafted feature HDF5：{input_path}")
        column_order = file.attrs["column_order"]
        if isinstance(column_order, bytes):
            column_order = column_order.decode("utf-8")
        columns = json.loads(str(column_order))
        data: dict[str, np.ndarray] = {}
        for column in columns:
            group_name = (
                "metadata" if column in file["metadata"] else "features"
            )
            dataset = file[group_name][column]
            if h5py.check_string_dtype(dataset.dtype) is not None:
                data[column] = dataset.asstr()[:]
            else:
                data[column] = dataset[:]
    return pd.DataFrame(data, columns=columns)

def _save(
    rows: list[dict[str, Any]],
    path: Path,
    save_hdf5: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(path, index=False)
    if save_hdf5:
        save_feature_hdf5(dataframe, path.with_suffix(".h5"))

def extract_offline_and_cache_features(
    input_root: str | Path,
    config: dict[str, Any],
    cache_dir: str | Path,
    modalities: list[str] | None = None,
) -> dict[str, Path]:
    selected = [
        str(value).lower()
        for value in (
            modalities or ["audio", "accel", "gelsight", "visual"]
        )
    ]
    unknown = set(selected).difference(
        {"audio", "accel", "gelsight", "visual"}
    )
    if unknown:
        raise ValueError(f"unknown modality：{sorted(unknown)}")

    root = Path(input_root)
    offline_cfg = config.get("offline", {})
    metadata, _ = load_offline_trial_tables(
        root,
        str(
            offline_cfg.get(
                "trial_manifest", "trial_split_manifest.csv"
            )
        ),
    )
    class_names = sorted(
        metadata["label"].astype(str).unique(),
        key=lambda value: int(value),
    )
    label_to_index = {
        label: index for index, label in enumerate(class_names)
    }
    rows_by_modality: dict[str, list[dict[str, Any]]] = {
        modality: [] for modality in selected
    }
    image_feature_cache: dict[
        str, dict[tuple[str, int], dict[str, float]]
    ] = {
        modality: {}
        for modality in selected
        if modality in {"gelsight", "visual"}
    }
    audio_rate = int(
        config.get("audio", {}).get("sample_rate", 44_100)
    )
    accel_rate = int(
        config.get("accel", {}).get("sample_rate", 3_000)
    )

    for split in ("train", "val", "test"):
        dataset = OfflineClassificationDataset(
            input_root=root,
            split=split,
            label_to_index=label_to_index,
            modalities=selected,
            config=config,
        )
        if "audio" in selected:
            audio_rate = dataset.audio_sample_rate
        try:
            for index in tqdm(
                range(len(dataset)),
                desc=f"offline {split} features",
            ):
                base = dataset.sample_metadata(index)
                for modality in selected:
                    if modality == "audio":
                        values = dataset.load_raw_modality(
                            index, modality
                        )
                        features = extract_audio_features(
                            values, audio_rate
                        )
                    elif modality == "accel":
                        values = dataset.load_raw_modality(
                            index, modality
                        )
                        features = extract_accel_features(
                            values, accel_rate
                        )
                    elif modality == "gelsight":
                        source_indices = (
                            dataset.paired_source_indices(
                                index, modality
                            )
                        )
                        rgb_frames = dataset.load_raw_modality(
                            index, modality
                        )
                        cache = image_feature_cache[modality]
                        frame_features: list[dict[str, float]] = []
                        for source_index, rgb in zip(
                            source_indices,
                            rgb_frames,
                        ):
                            source_key = (
                                split,
                                int(source_index),
                            )
                            if source_key not in cache:
                                cache[source_key] = (
                                    extract_gelsight_features(rgb)
                                )
                            frame_features.append(cache[source_key])
                        features = {
                            name: float(
                                np.mean(
                                    [
                                        values[name]
                                        for values in frame_features
                                    ]
                                )
                            )
                            for name in frame_features[0]
                        }
                    else:
                        source_index = dataset.paired_source_index(
                            index, modality
                        )
                        cache = image_feature_cache[modality]
                        source_key = (split, source_index)
                        if source_key not in cache:
                            rgb = dataset.load_raw_modality(
                                index, modality
                            )
                            cache[source_key] = (
                                extract_visual_features(rgb)
                            )
                        features = cache[source_key]
                    rows_by_modality[modality].append(
                        {**base, **features}
                    )
        finally:
            dataset.close()

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for modality, rows in rows_by_modality.items():
        path = out_dir / f"{modality}_features.csv"
        _save(
            rows,
            path,
            save_hdf5=bool(
                config.get("features", {}).get("save_hdf5", True)
            ),
        )
        outputs[modality] = path
    return outputs

def export_handcrafted_features(
    feature_tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
    config: dict[str, Any],
    save_csv: bool = True,
    save_hdf5: bool = True,
    save_parquet: bool = False,
    save_feature_manifest: bool = True,
    save_config_snapshot: bool = True,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exported_paths: dict[str, Path] = {}
    manifest_rows: list[dict[str, Any]] = []

    for modality, df in feature_tables.items():
        if df.empty:
            continue

        if save_csv:
            csv_path = output_path / f"{modality}_handcrafted_features.csv"
            df.to_csv(csv_path, index=False)
            exported_paths[f"{modality}_csv"] = csv_path

        if save_hdf5:
            hdf5_path = (
                output_path / f"{modality}_handcrafted_features.h5"
            )
            save_feature_hdf5(df, hdf5_path)
            exported_paths[f"{modality}_hdf5"] = hdf5_path

        if save_parquet:
            parquet_path = output_path / f"{modality}_handcrafted_features.parquet"
            df.to_parquet(parquet_path, index=False)
            exported_paths[f"{modality}_parquet"] = parquet_path

        modality_feature_cols = feature_columns(df)
        for col in modality_feature_cols:
            manifest_rows.append(
                {
                    "modality": modality,
                    "feature_name": col,
                }
            )

    if save_feature_manifest:
        manifest_path = output_path / "handcrafted_feature_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        exported_paths["feature_manifest"] = manifest_path

    if save_config_snapshot:
        config_path = output_path / "feature_export_config_snapshot.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        exported_paths["config_snapshot"] = config_path

    return exported_paths

def export_cached_feature_files(
    cached_paths: dict[str, Path],
    output_dir: str | Path,
    config: dict[str, Any],
    save_csv: bool = True,
    save_hdf5: bool = True,
    save_parquet: bool = False,
    save_feature_manifest: bool = True,
    save_config_snapshot: bool = True,
) -> dict[str, Path]:
    feature_tables: dict[str, pd.DataFrame] = {}

    for modality, path in cached_paths.items():
        if path.suffix.lower() == ".parquet":
            feature_tables[modality] = pd.read_parquet(path)
        else:
            feature_tables[modality] = pd.read_csv(path)

    return export_handcrafted_features(
        feature_tables=feature_tables,
        output_dir=output_dir,
        config=config,
        save_csv=save_csv,
        save_hdf5=save_hdf5,
        save_parquet=save_parquet,
        save_feature_manifest=save_feature_manifest,
        save_config_snapshot=save_config_snapshot,
    )
