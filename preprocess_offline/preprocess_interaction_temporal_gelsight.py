from __future__ import annotations
import argparse
import csv
import gc
import json
import os
import random
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
import preprocess_class_temporal_gelsight as base
from preprocess_config import load_interaction_fabric_tasks

RANDOM_SEED = base.RANDOM_SEED
SPLITS = base.SPLITS
INTERACTION_GELSIGHT_WORKERS = 8
TASK_TO_ID = {
    "primitive": 0,
    "press_angle": 1,
    "slide_trajectory": 2,
    "twist_position": 3,
}
FABRIC_TASKS: dict[int, str] = {}
TASK_PRIMITIVES = {
    "primitive": {"press", "slide", "twist"},
    "press_angle": {"press"},
    "slide_trajectory": {"slide"},
    "twist_position": {"twist"},
}
PRIMITIVE_TRIAL_GLOBS = {
    "press": "continuous_pressing/position_*/orientation_*/trial_*",
    "slide": "stable_sliding/track_*/trial_*",
    "twist": "stable_twisting/position_*/direction_*/trial_*",
}
EXPECTED_COUNTS_BY_TASK = {
    "primitive": {"press": 36, "slide": 24, "twist": 24},
    "press_angle": {"press": 36},
    "slide_trajectory": {"slide": 24},
    "twist_position": {"twist": 24},
}
TASK_LABEL_MAPPINGS = {
    "primitive": base.PRIMITIVE_TO_ID,
    "press_angle": {"0": 0, "45": 1, "90": 2},
    "slide_trajectory": base.TRACK_TO_ID,
    "twist_position": base.POSITION_TO_ID,
}

def interaction_task(record: base.TrialRecord) -> str:
    return FABRIC_TASKS[record.fabric_id]

def semantic_target(record: base.TrialRecord) -> str:
    task = interaction_task(record)
    if task == "primitive":
        return record.primitive
    if task == "press_angle":
        return str(record.press_orientation_deg)
    if task == "slide_trajectory":
        return record.slide_track
    if task == "twist_position":
        return record.position
    raise AssertionError(task)

def numeric_target(record: base.TrialRecord) -> int:
    task = interaction_task(record)
    return int(TASK_LABEL_MAPPINGS[task][semantic_target(record)])

def discover_interaction_trials(dataset_root: Path) -> list[base.TrialRecord]:
    records: list[base.TrialRecord] = []
    for fabric_id, task in FABRIC_TASKS.items():
        fabric_dir = dataset_root / f"fabric_{fabric_id:03d}"
        if not fabric_dir.is_dir():
            raise FileNotFoundError(f"is missing Interaction fabric directory：{fabric_dir}")
        trial_dirs: list[Path] = []
        for primitive in sorted(TASK_PRIMITIVES[task]):
            trial_dirs.extend(
                path
                for path in fabric_dir.glob(PRIMITIVE_TRIAL_GLOBS[primitive])
                if path.is_dir()
            )
        selected = [
            base.parse_trial_directory(path, dataset_root)
            for path in sorted(
                trial_dirs,
                key=lambda path: base.natural_key(
                    path.relative_to(fabric_dir).as_posix()
                ),
            )
        ]
        actual_counts = {
            primitive: sum(row.primitive == primitive for row in selected)
            for primitive in sorted(TASK_PRIMITIVES[task])
        }
        expected_counts = EXPECTED_COUNTS_BY_TASK[task]
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"fabric_{fabric_id:03d} ({task}) trial count is incorrect："
                f"expected {expected_counts}，actual {actual_counts}"
            )
        for record in selected:
            missing = [
                path for path in base.required_trial_paths(record)
                if not path.exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{record.trial_uid} is missing required paths："
                    + ", ".join(str(path) for path in missing)
                )
            numeric_target(record)
        records.extend(selected)
    return sorted(records, key=lambda row: (row.fabric_id, row.trial_uid))

def _split_sizes(group_size: int) -> tuple[int, int, int]:
    if group_size < 3:
        raise ValueError(f"each task class requires at least 3  trial，actual {group_size}")
    if group_size == 3:
        return 1, 1, 1
    val_size = max(1, int(round(group_size / 6)))
    test_size = val_size
    train_size = group_size - val_size - test_size
    if train_size < 1:
        raise ValueError(f"cannot split {group_size}  trial")
    return train_size, val_size, test_size

def assign_stratified_splits(
    records: list[base.TrialRecord],
) -> list[base.TrialRecord]:
    groups: dict[tuple[str, str], list[base.TrialRecord]] = {}
    for record in records:
        key = (interaction_task(record), semantic_target(record))
        groups.setdefault(key, []).append(record)

    result: list[base.TrialRecord] = []
    for (task, label), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: row.trial_uid)
        stable_offset = zlib.crc32(f"{task}:{label}".encode("utf-8"))
        rng = random.Random(RANDOM_SEED + stable_offset)
        rng.shuffle(rows)
        train_size, val_size, test_size = _split_sizes(len(rows))
        split_rows = {
            "test": rows[:test_size],
            "val": rows[test_size:test_size + val_size],
            "train": rows[test_size + val_size:],
        }
        if len(split_rows["train"]) != train_size:
            raise AssertionError("stratified split counts are inconsistent")
        for split, part in split_rows.items():
            result.extend(replace(row, split=split) for row in part)
    return sorted(
        result,
        key=lambda row: (SPLITS.index(row.split), row.fabric_id, row.trial_uid),
    )

def manifest_row(record: base.TrialRecord) -> dict[str, Any]:
    task = interaction_task(record)
    return {
        "trial_uid": record.trial_uid,
        "split": record.split,
        "fabric_id": record.fabric_id,
        "interaction_task": task,
        "task_id": TASK_TO_ID[task],
        "task_label": numeric_target(record),
        "task_label_name": semantic_target(record),
        "source_primitive": record.primitive,
        "primitive": (
            base.PRIMITIVE_TO_ID[record.primitive]
            if task == "primitive" else -1
        ),
        "press_angle": (
            TASK_LABEL_MAPPINGS[task][semantic_target(record)]
            if task == "press_angle" else -1
        ),
        "press_angle_deg": (
            record.press_orientation_deg if task == "press_angle" else -1
        ),
        "slide_trajectory": (
            base.TRACK_TO_ID[record.slide_track]
            if task == "slide_trajectory" else -1
        ),
        "slide_trajectory_name": (
            record.slide_track if task == "slide_trajectory" else ""
        ),
        "twist_position": (
            base.POSITION_TO_ID[record.position]
            if task == "twist_position" else -1
        ),
        "twist_position_name": (
            record.position if task == "twist_position" else ""
        ),
        "position": record.position,
        "press_orientation_deg": record.press_orientation_deg,
        "slide_track": record.slide_track,
        "twist_direction": record.twist_direction,
        "repeat": record.repeat,
    }

def write_interaction_manifest(
    records: list[base.TrialRecord], output_path: Path
) -> None:
    rows = [manifest_row(record) for record in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, output_path)

def _derived_numpy_labels(labels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    fabric_ids = np.asarray(labels["fabric_id"])
    size = len(fabric_ids)
    task_ids = np.full(size, -1, dtype=np.int32)
    task_labels = np.full(size, -1, dtype=np.int32)
    primitive = np.full(size, -1, dtype=np.int32)
    press_angle = np.full(size, -1, dtype=np.int32)
    slide_trajectory = np.full(size, -1, dtype=np.int32)
    twist_position = np.full(size, -1, dtype=np.int32)

    for fabric_id, task in FABRIC_TASKS.items():
        mask = fabric_ids == fabric_id
        task_ids[mask] = TASK_TO_ID[task]
        if task == "primitive":
            primitive[mask] = np.asarray(labels["primitive"])[mask]
            task_labels[mask] = primitive[mask]
        elif task == "press_angle":
            degrees = np.asarray(labels["press_orientation_deg"])[mask]
            encoded = np.asarray(
                [TASK_LABEL_MAPPINGS[task][str(int(value))] for value in degrees],
                dtype=np.int32,
            )
            press_angle[mask] = encoded
            task_labels[mask] = encoded
        elif task == "slide_trajectory":
            slide_trajectory[mask] = np.asarray(labels["slide_track"])[mask]
            task_labels[mask] = slide_trajectory[mask]
        elif task == "twist_position":
            twist_position[mask] = np.asarray(labels["position"])[mask]
            task_labels[mask] = twist_position[mask]

    if np.any(task_ids < 0) or np.any(task_labels < 0):
        raise RuntimeError("output contains an unconfigured fabric_id or invalid task_label")
    return {
        "task_id": task_ids,
        "task_label": task_labels,
        "interaction_primitive": primitive,
        "press_angle": press_angle,
        "slide_trajectory": slide_trajectory,
        "twist_position": twist_position,
    }

def add_h5_interaction_labels(path: Path) -> None:
    import h5py

    with h5py.File(path, "r+") as file:
        labels_group = file["labels"]
        source = {name: dataset[:] for name, dataset in labels_group.items()}
        derived = _derived_numpy_labels(source)
        for name, values in derived.items():
            if name in labels_group:
                del labels_group[name]
            labels_group.create_dataset(
                name,
                data=values,
                chunks=True,
                compression=base.HDF5_COMPRESSION,
                compression_opts=base.HDF5_COMPRESSION_LEVEL,
            )
        file.attrs["dataset_profile"] = "interaction"
        file.attrs["task_mapping"] = json.dumps(TASK_TO_ID)
        file.attrs["fabric_task_mapping"] = json.dumps(FABRIC_TASKS)
        file.attrs["task_label_mappings"] = json.dumps(TASK_LABEL_MAPPINGS)

def patch_gelsight_save() -> None:
    original_save = base.atomic_torch_save

    def interaction_save(
        payload: dict[str, Any], path: Path, overwrite: bool
    ) -> None:
        import torch

        labels = payload["labels"]
        numpy_labels = {
            name: tensor.detach().cpu().numpy() for name, tensor in labels.items()
        }
        for name, values in _derived_numpy_labels(numpy_labels).items():
            labels[name] = torch.from_numpy(values)
        payload["config"].update(
            {
                "dataset_profile": "interaction",
                "task_mapping": TASK_TO_ID,
                "fabric_task_mapping": FABRIC_TASKS,
                "task_label_mappings": TASK_LABEL_MAPPINGS,
                "conditional_task_label_missing_value": -1,
            }
        )
        original_save(payload, path, overwrite)

    base.atomic_torch_save = interaction_save

def _resize_interaction_gelsight_image(path: Path) -> np.ndarray:
    from PIL import Image

    resampling_api = getattr(Image, "Resampling", Image)
    resampling = {
        "nearest": resampling_api.NEAREST,
        "bilinear": resampling_api.BILINEAR,
        "bicubic": resampling_api.BICUBIC,
        "lanczos": resampling_api.LANCZOS,
    }[base.GELSIGHT_RESAMPLE]
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(
            base.IMAGE_SIZE, resample=resampling
        )
        array = np.asarray(resized, dtype=np.uint8)
    return np.transpose(array, (2, 0, 1))

def process_interaction_gelsight(
    records: list[base.TrialRecord],
    output_root: Path,
    overwrite: bool,
    sync_by_trial: dict[str, base.SyncInfo],
) -> dict[str, int]:
    import torch

    counts: dict[str, int] = {}
    channels, height, width = 3, base.IMAGE_SIZE[1], base.IMAGE_SIZE[0]
    for split in SPLITS:
        split_records = [row for row in records if row.split == split]
        frame_rows = {
            record.trial_uid: base.read_gelsight_rows(record)
            for record in split_records
        }
        total = sum(len(rows) for rows in frame_rows.values())
        counts[split] = total
        output_root.mkdir(parents=True, exist_ok=True)
        descriptor, mmap_name = tempfile.mkstemp(
            dir=output_root,
            prefix=f".interaction_gelsight_{split}.",
            suffix=".uint8.mmap",
        )
        os.close(descriptor)
        mmap_path = Path(mmap_name)
        images = np.memmap(
            mmap_path,
            mode="w+",
            dtype=np.uint8,
            shape=(total, channels, height, width),
        )
        label_names = (
            "fabric_id",
            "primitive",
            "position",
            "press_orientation_deg",
            "slide_track",
            "twist_direction",
            "repeat",
        )
        labels = {
            name: np.empty(total, dtype=np.int32) for name in label_names
        }
        trial_indices = np.empty(total, dtype=np.int32)
        source_frame_indices = np.empty(total, dtype=np.int32)
        host_timestamps = np.empty(total, dtype=np.float64)
        relative_timestamps = np.empty(total, dtype=np.float32)
        window_indices = np.empty(total, dtype=np.int32)
        trial_uids = [record.trial_uid for record in split_records]

        cursor = 0
        try:
            with ThreadPoolExecutor(
                max_workers=INTERACTION_GELSIGHT_WORKERS,
                thread_name_prefix="interaction_gelsight",
            ) as executor:
                for trial_index, record in enumerate(split_records):
                    sync = sync_by_trial.get(record.trial_uid)
                    if sync is None:
                        sync = base.compute_sync_only(record)
                        sync_by_trial[record.trial_uid] = sync
                    label_values = record.label_values()
                    trial_rows = frame_rows[record.trial_uid]
                    resized_images = executor.map(
                        _resize_interaction_gelsight_image,
                        (image_path for _, image_path, _ in trial_rows),
                    )
                    for (
                        source_frame_index,
                        _image_path,
                        host_ts,
                    ), resized_image in zip(trial_rows, resized_images):
                        images[cursor] = resized_image
                        for name in label_names:
                            labels[name][cursor] = label_values[name]
                        trial_indices[cursor] = trial_index
                        source_frame_indices[cursor] = source_frame_index
                        host_timestamps[cursor] = host_ts
                        relative_timestamps[cursor] = (
                            host_ts - sync.sync_start_host_sec
                        )
                        window_indices[cursor] = base.nearest_window_index(
                            host_ts, sync
                        )
                        cursor += 1
                    if (
                        (trial_index + 1) % 10 == 0
                        or trial_index + 1 == len(split_records)
                    ):
                        print(
                            f"[GelSight:{split}] processed {trial_index + 1}/"
                            f"{len(split_records)}  trial，"
                            f"{cursor}/{total} frames",
                            flush=True,
                        )
            if cursor != total:
                raise RuntimeError(
                    f"during processing GelSight count changed：{cursor} and {total}"
                )
            images.flush()
            payload = {
                "images": torch.from_numpy(images),
                "labels": {
                    name: torch.from_numpy(values)
                    for name, values in labels.items()
                },
                "metadata": {
                    "trial_index": torch.from_numpy(trial_indices),
                    "source_frame_index": torch.from_numpy(
                        source_frame_indices
                    ),
                    "host_timestamp_sec": torch.from_numpy(host_timestamps),
                    "time_from_sync_start_sec": torch.from_numpy(
                        relative_timestamps
                    ),
                    "nearest_window_index": torch.from_numpy(window_indices),
                    "trial_uids": trial_uids,
                },
                "config": {
                    "random_seed": RANDOM_SEED,
                    "image_size": list(base.IMAGE_SIZE),
                    "image_dtype": "uint8",
                    "image_layout": "NCHW_RGB",
                    "primitive_mapping": base.PRIMITIVE_TO_ID,
                    "position_mapping": base.POSITION_TO_ID,
                    "slide_track_mapping": base.TRACK_TO_ID,
                    "twist_direction_mapping": base.TWIST_DIRECTION_TO_ID,
                    "window_seconds": base.WINDOW_SECONDS,
                    "window_stride_seconds": base.WINDOW_STRIDE_SECONDS,
                },
            }
            base.atomic_torch_save(
                payload,
                output_root / f"gelsight_{split}.pt",
                overwrite=overwrite,
            )
            del payload
        finally:
            del images
            gc.collect()
            mmap_path.unlink(missing_ok=True)
    return counts

def interaction_statistics(records: list[base.TrialRecord]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "dataset_profile": "interaction",
        "random_seed": RANDOM_SEED,
        "total_trials": len(records),
        "fabric_task_mapping": {
            str(fabric_id): task for fabric_id, task in FABRIC_TASKS.items()
        },
        "task_label_mappings": TASK_LABEL_MAPPINGS,
        "splits": {},
    }
    for split in SPLITS:
        split_rows = [row for row in records if row.split == split]
        split_stats: dict[str, Any] = {"trials": len(split_rows), "tasks": {}}
        for task in TASK_TO_ID:
            task_rows = [row for row in split_rows if interaction_task(row) == task]
            split_stats["tasks"][task] = {
                "trials": len(task_rows),
                "label_counts": {
                    label: sum(semantic_target(row) == label for row in task_rows)
                    for label in TASK_LABEL_MAPPINGS[task]
                },
            }
        stats["splits"][split] = split_stats
    return stats

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="offline preprocessing Interaction synchronized temporal modalities and GelSight。"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("all", "temporal", "gelsight", "scan"), default="all"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main() -> None:
    global FABRIC_TASKS
    args = parse_args()
    FABRIC_TASKS = load_interaction_fabric_tasks(
        args.config, TASK_TO_ID
    )
    base.set_reproducible_seed(RANDOM_SEED)
    print(f"scanning Interaction dataset：{args.dataset_root}", flush=True)
    records = assign_stratified_splits(
        discover_interaction_trials(args.dataset_root)
    )
    base.assert_no_trial_leakage(records)
    stats = interaction_statistics(records)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run or args.mode == "scan":
        print("read-only validation completed：no files were written。", flush=True)
        return

    modes = {"temporal", "gelsight"} if args.mode == "all" else {args.mode}
    base.check_output_targets(args.output_root, modes, args.overwrite)
    write_interaction_manifest(
        records, args.output_root / base.SPLIT_MANIFEST_NAME
    )

    sync_by_trial: dict[str, base.SyncInfo] = {}
    if "temporal" in modes:
        sync_by_trial, counts = base.process_temporal(
            records, args.output_root, args.overwrite
        )
        for split in SPLITS:
            add_h5_interaction_labels(
                args.output_root / f"temporal_{split}.h5"
            )
        stats["temporal_windows"] = counts
    if "gelsight" in modes:
        patch_gelsight_save()
        stats["gelsight_frames"] = process_interaction_gelsight(
            records, args.output_root, args.overwrite, sync_by_trial
        )

    summary_path = args.output_root / base.SUMMARY_NAME
    summary_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Interaction processing completed，output directory：{args.output_root}", flush=True)

if __name__ == "__main__":
    main()
