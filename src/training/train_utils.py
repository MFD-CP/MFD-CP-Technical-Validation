from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.offline_dataset import (
    OfflineClassificationDataset,
    load_offline_classification_tables,
    offline_collate_fn,
)

def build_label_mapping(metadata: pd.DataFrame) -> tuple[dict[str, int], list[str]]:
    values = [str(x) for x in metadata["label"].dropna().unique()]
    try:
        class_names = sorted(values, key=lambda value: int(value))
    except ValueError:
        class_names = sorted(values)
    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    return label_to_index, class_names

def make_dataloaders(
    metadata: pd.DataFrame,
    split_df: pd.DataFrame,
    dataset_root: str | Path,
    modalities: list[str],
    config: dict[str, Any],
    label_to_index: dict[str, int],
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = OfflineClassificationDataset(
            input_root=dataset_root,
            split=split,
            label_to_index=label_to_index,
            modalities=modalities,
            config=config,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=offline_collate_fn,
        )
    return loaders

def load_classification_data(
    config: dict[str, Any],
    project_root: str | Path,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    return load_offline_classification_tables(config, project_root)

def compute_class_weights(metadata: pd.DataFrame, label_to_index: dict[str, int]) -> torch.Tensor:
    labels = metadata["label"].astype(str).map(label_to_index).to_numpy(dtype=int)
    counts = np.bincount(labels, minlength=len(label_to_index)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / max(weights.mean(), 1e-8)
    return torch.tensor(weights, dtype=torch.float32)

