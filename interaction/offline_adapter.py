from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from src.features.accel_features import extract_accel_features
from src.features.audio_features import extract_audio_features
from src.features.feature_cache import _save
from src.features.image_features import (
    extract_gelsight_features,
)
from src.data.offline_dataset import (
    OfflineClassificationDataset,
    resolve_offline_root,
)

SPLITS = ("train", "val", "test")
SUPPORTED_MODALITIES = {"audio", "accel", "gelsight"}

def normalize_label(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)

def load_interaction_manifest(
    config: dict[str, Any],
    project_root: str | Path,
) -> tuple[Path, pd.DataFrame]:
    root = resolve_offline_root(config, project_root)
    manifest_name = str(
        config.get("offline", {}).get(
            "trial_manifest", "trial_split_manifest.csv"
        )
    )
    path = root / manifest_name
    if not path.is_file():
        raise FileNotFoundError(f"cannot find Interaction offline manifest：{path}")
    manifest = pd.read_csv(path)
    required = {
        "trial_uid",
        "split",
        "fabric_id",
        "interaction_task",
        "task_label",
        "task_label_name",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing fields：{sorted(missing)}")
    unknown_splits = set(manifest["split"].astype(str)).difference(SPLITS)
    if unknown_splits:
        raise ValueError(f"{path} contains unknown split：{sorted(unknown_splits)}")
    if manifest["trial_uid"].duplicated().any():
        raise ValueError(f"{path} contains duplicate trial_uid")
    return root, manifest

def task_split_from_manifest(
    manifest: pd.DataFrame,
    task_name: str,
    target_label: str,
) -> pd.DataFrame:
    if target_label not in manifest.columns:
        raise ValueError(f"manifest is missing the task-label column：{target_label}")
    rows = manifest[
        manifest["interaction_task"].astype(str) == str(task_name)
    ].copy()
    if rows.empty:
        raise ValueError(f"manifest contains no task：{task_name}")
    rows = rows[rows[target_label].notna()].copy()
    if pd.api.types.is_numeric_dtype(rows[target_label]):
        rows = rows[rows[target_label] >= 0].copy()
    rows = rows.rename(columns={"trial_uid": "trial_id"})
    rows["task_name"] = task_name
    rows["target_label"] = rows[target_label]
    rows["label"] = rows[target_label].map(normalize_label)

    for split in SPLITS:
        part = rows[rows["split"] == split]
        if part.empty:
            raise ValueError(f"task {task_name} of {split} set is empty")
    train_labels = set(rows.loc[rows["split"] == "train", "label"])
    for split in ("val", "test"):
        labels = set(rows.loc[rows["split"] == split, "label"])
        if labels != train_labels:
            raise ValueError(
                f"task {task_name} of {split} label set {sorted(labels)} "
                f"and train {sorted(train_labels)} inconsistent"
            )
    return rows.reset_index(drop=True)

def build_label_mapping(
    split_df: pd.DataFrame,
) -> tuple[dict[str, int], list[str]]:
    values = list(dict.fromkeys(split_df["label"].map(normalize_label)))
    try:
        class_names = sorted(values, key=lambda value: int(value))
    except ValueError:
        class_names = sorted(values)
    return (
        {label: index for index, label in enumerate(class_names)},
        class_names,
    )

class OfflineInteractionDataset(Dataset):

    def __init__(
        self,
        input_root: str | Path,
        split: str,
        split_df: pd.DataFrame,
        label_to_index: dict[str, int],
        modalities: list[str],
        config: dict[str, Any],
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split：{split}")
        required = {"trial_id", "split", "label", "fabric_id"}
        missing = required.difference(split_df.columns)
        if missing:
            raise ValueError(f"task split is missing fields：{sorted(missing)}")

        fabric_labels = {
            str(int(value)): index
            for index, value in enumerate(
                sorted(split_df["fabric_id"].dropna().astype(int).unique())
            )
        }
        self.base = OfflineClassificationDataset(
            input_root=input_root,
            split=split,
            label_to_index=fabric_labels,
            modalities=modalities,
            config=config,
        )
        current = split_df[split_df["split"] == split].copy()
        self.label_by_trial = {
            str(row.trial_id): normalize_label(row.label)
            for row in current.itertuples()
        }
        self.label_to_index = label_to_index
        self.modalities = [str(value).lower() for value in modalities]
        self.indices = np.asarray(
            [
                index
                for index, trial_id in enumerate(self.base.trial_ids)
                if trial_id in self.label_by_trial
            ],
            dtype=np.int64,
        )
        found = {self.base.trial_ids[int(index)] for index in self.indices}
        missing_trials = set(self.label_by_trial).difference(found)
        if missing_trials:
            preview = sorted(missing_trials)[:5]
            raise ValueError(
                f"{split} offline file is missing {len(missing_trials)} task trial："
                f"{preview}"
            )
        if len(self.indices) == 0:
            raise ValueError(f"{split} has no windows for the current task")

    @property
    def audio_sample_rate(self) -> int:
        return self.base.audio_sample_rate

    def __len__(self) -> int:
        return len(self.indices)

    def _base_index(self, index: int) -> int:
        return int(self.indices[index])

    def sample_metadata(self, index: int) -> dict[str, Any]:
        raw_index = self._base_index(index)
        metadata = self.base.sample_metadata(raw_index)
        metadata["label"] = self.label_by_trial[metadata["trial_id"]]
        return metadata

    def paired_source_indices(
        self, index: int, modality: str
    ) -> np.ndarray:
        return self.base.paired_source_indices(
            self._base_index(index), modality
        )

    def paired_source_index(self, index: int, modality: str) -> int:
        return int(self.paired_source_indices(index, modality)[0])

    def load_raw_modality(self, index: int, modality: str) -> np.ndarray:
        return self.base.load_raw_modality(
            self._base_index(index), modality
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw_index = self._base_index(index)
        item = self.base[raw_index]
        trial_id = str(item["trial_id"])
        label = self.label_by_trial[trial_id]
        item["label"] = torch.tensor(
            self.label_to_index[label], dtype=torch.long
        )
        return item

    def close(self) -> None:
        self.base.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["base"] = self.base.__getstate__()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        base_state = state["base"]
        base = OfflineClassificationDataset.__new__(
            OfflineClassificationDataset
        )
        base.__dict__.update(base_state)
        state["base"] = base
        self.__dict__.update(state)

def extract_offline_interaction_features(
    input_root: str | Path,
    split_df: pd.DataFrame,
    label_to_index: dict[str, int],
    config: dict[str, Any],
    cache_dir: str | Path,
    modalities: list[str],
) -> dict[str, Path]:
    selected = [str(value).lower() for value in modalities]
    unknown = set(selected).difference(SUPPORTED_MODALITIES)
    if unknown:
        raise ValueError(f"unknown modality：{sorted(unknown)}")

    rows_by_modality: dict[str, list[dict[str, Any]]] = {
        modality: [] for modality in selected
    }
    image_feature_cache: dict[
        str, dict[tuple[str, int], dict[str, float]]
    ] = {
        modality: {}
        for modality in selected
        if modality == "gelsight"
    }
    audio_rate = int(config.get("audio", {}).get("sample_rate", 44_100))
    accel_rate = int(config.get("accel", {}).get("sample_rate", 3_000))

    for split in SPLITS:
        dataset = OfflineInteractionDataset(
            input_root=input_root,
            split=split,
            split_df=split_df,
            label_to_index=label_to_index,
            modalities=selected,
            config=config,
        )
        if "audio" in selected:
            audio_rate = dataset.audio_sample_rate
        try:
            for index in tqdm(
                range(len(dataset)),
                desc=f"interaction offline {split} features",
            ):
                metadata = dataset.sample_metadata(index)
                for modality in selected:
                    if modality == "audio":
                        features = extract_audio_features(
                            dataset.load_raw_modality(index, modality),
                            audio_rate,
                        )
                    elif modality == "accel":
                        features = extract_accel_features(
                            dataset.load_raw_modality(index, modality),
                            accel_rate,
                        )
                    elif modality == "gelsight":
                        source_indices = dataset.paired_source_indices(
                            index, modality
                        )
                        frames = dataset.load_raw_modality(index, modality)
                        cache = image_feature_cache[modality]
                        frame_features: list[dict[str, float]] = []
                        for source_index, rgb in zip(source_indices, frames):
                            key = (split, int(source_index))
                            if key not in cache:
                                cache[key] = extract_gelsight_features(rgb)
                            frame_features.append(cache[key])
                        features = {
                            name: float(
                                np.mean([row[name] for row in frame_features])
                            )
                            for name in frame_features[0]
                        }
                    else:
                        raise ValueError(f"unknown Interaction modality：{modality}")
                    rows_by_modality[modality].append(
                        {**metadata, **features}
                    )
        finally:
            dataset.close()

    output_root = Path(cache_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    save_hdf5 = bool(config.get("features", {}).get("save_hdf5", True))
    for modality, rows in rows_by_modality.items():
        path = output_root / f"{modality}_features.csv"
        _save(rows, path, save_hdf5=save_hdf5)
        outputs[modality] = path
    return outputs
