from __future__ import annotations
from pathlib import Path
import argparse
import csv
import gc
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass, replace
from typing import Any
import numpy as np
from preprocess_config import load_classification_fabric_ids

RANDOM_SEED = 20260730
TRAIN_TRIALS_PER_FABRIC = 60
VAL_TRIALS_PER_FABRIC = 12
TEST_TRIALS_PER_FABRIC = 12
EXPECTED_TRIALS_PER_FABRIC = 84

WINDOW_SECONDS = 0.5
WINDOW_STRIDE_SECONDS = 0.25
EXPECTED_AUDIO_SAMPLE_RATE_HZ = 44_100
AUDIO_PEAK_NORMALIZE_PER_WINDOW = True
ACCEL_ZSCORE_PER_WINDOW = False
NORMALIZATION_EPSILON = 1e-8

AUDIO_MAIN_RELATIVE_PATH = Path("audio_main.csv")
AUDIO_SECONDARY_RELATIVE_PATH = Path("audio_secondary.csv")
AUDIO_SECONDARY_GAIN = 1.0
ACCEL_RELATIVE_PATH = Path("acceleration.csv")
GELSIGHT_IMAGE_DIR_RELATIVE_PATH = Path("gelsight_images")
GELSIGHT_SAMPLE_RATE_HZ = 25.0

IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
GELSIGHT_RESAMPLE = "bilinear"

HDF5_COMPRESSION = "gzip"
HDF5_COMPRESSION_LEVEL = 4
PT_USE_NEW_ZIPFILE_SERIALIZATION = True
STRICT_VALIDATION = True
OVERWRITE_OUTPUTS = False

SPLIT_MANIFEST_NAME = "trial_split_manifest.csv"
SUMMARY_NAME = "preprocessing_summary.json"

PRIMITIVE_TO_ID = {"press": 0, "slide": 1, "twist": 2}
POSITION_TO_ID = {"A": 0, "B": 1, "C": 2, "D": 3}
TWIST_DIRECTION_TO_ID = {"CW": 0, "CCW": 1}
TRACK_TO_ID = {
    "A_to_B": 0,
    "B_to_A": 1,
    "B_to_D": 2,
    "D_to_B": 3,
    "D_to_C": 4,
    "C_to_D": 5,
    "C_to_A": 6,
    "A_to_C": 7,
}
SPLITS = ("train", "val", "test")

@dataclass(frozen=True)
class TrialRecord:
    fabric_id: int
    primitive: str
    position: str
    press_orientation_deg: int
    slide_track: str
    twist_direction: str
    repeat: int
    trial_uid: str
    trial_dir: Path
    split: str = ""

    def label_values(self) -> dict[str, int]:
        return {
            "fabric_id": self.fabric_id,
            "primitive": PRIMITIVE_TO_ID[self.primitive],
            "position": (
                POSITION_TO_ID.get(self.position, -1)
                if self.primitive in {"press", "twist"}
                else -1
            ),
            "press_orientation_deg": self.press_orientation_deg,
            "slide_track": TRACK_TO_ID.get(self.slide_track, -1),
            "twist_direction": TWIST_DIRECTION_TO_ID.get(
                self.twist_direction, -1
            ),
            "repeat": self.repeat,
        }

@dataclass(frozen=True)
class SyncInfo:
    audio_start_host_sec: float
    audio_end_host_sec: float
    accel_start_host_sec: float
    accel_end_host_sec: float
    sync_start_host_sec: float
    sync_end_host_sec: float
    num_windows: int

def natural_key(text: str) -> tuple[Any, ...]:
    import re

    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    )

def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

def normalize_primitive(raw: str) -> str:
    mapping = {
        "continuous_pressing": "press",
        "stable_sliding": "slide",
        "stable_twisting": "twist",
    }
    if raw not in mapping:
        raise ValueError(f"unsupported interaction primitive：{raw!r}")
    return mapping[raw]

def parse_fabric_id(name: str) -> int:
    suffix = name.removeprefix("fabric_")
    if not name.startswith("fabric_") or len(suffix) != 3 or not suffix.isdigit():
        raise ValueError(f"invalid fabric directory name：{name}")
    fabric_id = int(suffix)
    if fabric_id < 1:
        raise ValueError(f"invalid fabric directory name：{name}")
    return fabric_id

def parse_trial_directory(trial_dir: Path, dataset_root: Path) -> TrialRecord:
    relative = trial_dir.relative_to(dataset_root)
    parts = relative.parts
    if len(parts) < 4:
        raise ValueError(f"invalid trial path：{relative}")
    fabric_id = parse_fabric_id(parts[0])
    primitive = normalize_primitive(parts[1])
    position = ""
    orientation = -1
    track = ""
    direction = ""

    if not parts[-1].startswith("trial_"):
        raise ValueError(f"invalid trial directory name：{parts[-1]}")
    repeat_text = parts[-1].removeprefix("trial_")
    if not repeat_text.isdigit():
        raise ValueError(f"invalid trial directory name：{parts[-1]}")
    repeat = int(repeat_text)

    if primitive == "press":
        if (
            len(parts) != 5
            or not parts[2].startswith("position_")
            or not parts[3].startswith("orientation_")
        ):
            raise ValueError(f"invalid pressing trial path：{relative}")
        position = parts[2].removeprefix("position_")
        orientation_text = parts[3].removeprefix("orientation_")
        if not orientation_text.isdigit():
            raise ValueError(f"invalid pressing orientation：{parts[3]}")
        orientation = int(orientation_text)
    elif primitive == "slide":
        if len(parts) != 4 or not parts[2].startswith("track_"):
            raise ValueError(f"invalid sliding trial path：{relative}")
        track_parts = parts[2].removeprefix("track_").split("_", 1)
        if len(track_parts) != 2 or not track_parts[0].isdigit():
            raise ValueError(f"invalid sliding trajectory：{parts[2]}")
        track = track_parts[1]
    elif primitive == "twist":
        if (
            len(parts) != 5
            or not parts[2].startswith("position_")
            or not parts[3].startswith("direction_")
        ):
            raise ValueError(f"invalid twisting trial path：{relative}")
        position = parts[2].removeprefix("position_")
        direction = parts[3].removeprefix("direction_")

    record = TrialRecord(
        fabric_id=fabric_id,
        primitive=primitive,
        position=position,
        press_orientation_deg=orientation,
        slide_track=track,
        twist_direction=direction,
        repeat=repeat,
        trial_uid=relative.as_posix(),
        trial_dir=trial_dir,
    )
    validate_label(record)
    return record

def validate_label(record: TrialRecord) -> None:
    if record.primitive == "press":
        if record.position not in POSITION_TO_ID:
            raise ValueError(f"{record.trial_uid}：invalid press position")
        if record.press_orientation_deg < 0:
            raise ValueError(f"{record.trial_uid}：is missing press angle")
    elif record.primitive == "slide":
        if record.slide_track not in TRACK_TO_ID:
            raise ValueError(f"{record.trial_uid}：invalid slide trajectory")
    elif record.primitive == "twist":
        if record.position not in POSITION_TO_ID:
            raise ValueError(f"{record.trial_uid}：invalid twist position")
        if record.twist_direction not in TWIST_DIRECTION_TO_ID:
            raise ValueError(f"{record.trial_uid}：invalid twist direction")

def required_trial_paths(record: TrialRecord) -> list[Path]:
    return [
        record.trial_dir / AUDIO_MAIN_RELATIVE_PATH,
        record.trial_dir / AUDIO_SECONDARY_RELATIVE_PATH,
        record.trial_dir / ACCEL_RELATIVE_PATH,
        record.trial_dir / GELSIGHT_IMAGE_DIR_RELATIVE_PATH,
    ]

def discover_trials(
    dataset_root: Path, fabric_ids: list[int]
) -> list[TrialRecord]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root directory does not exist：{dataset_root}")

    fabric_dirs = [
        dataset_root / f"fabric_{fabric_id:03d}" for fabric_id in fabric_ids
    ]
    missing = [path for path in fabric_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "configured classification fabric directories are missing："
            + ", ".join(str(path) for path in missing)
        )

    records: list[TrialRecord] = []
    for fabric_dir in fabric_dirs:
        trial_dirs = sorted(
            (
                path
                for pattern in (
                    "continuous_pressing/position_*/orientation_*/trial_*",
                    "stable_sliding/track_*/trial_*",
                    "stable_twisting/position_*/direction_*/trial_*",
                )
                for path in fabric_dir.glob(pattern)
                if path.is_dir()
            ),
            key=lambda path: natural_key(path.relative_to(fabric_dir).as_posix()),
        )
        fabric_records = [
            parse_trial_directory(path, dataset_root) for path in trial_dirs
        ]
        if STRICT_VALIDATION and len(fabric_records) != EXPECTED_TRIALS_PER_FABRIC:
            raise RuntimeError(
                f"{fabric_dir.name}：expected {EXPECTED_TRIALS_PER_FABRIC}  "
                f"trial，actually found {len(fabric_records)} "
            )
        counts = {
            primitive: sum(
                record.primitive == primitive for record in fabric_records
            )
            for primitive in PRIMITIVE_TO_ID
        }
        if STRICT_VALIDATION and counts != {"press": 36, "slide": 24, "twist": 24}:
            raise RuntimeError(
                f"{fabric_dir.name}：expected primitive counts for press/slide/twist "
                f"36/24/24，actual {counts}"
            )
        for record in fabric_records:
            missing = [path for path in required_trial_paths(record) if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{record.trial_uid}：is missing required paths："
                    + ", ".join(str(path) for path in missing)
                )
        records.extend(fabric_records)
    return records

def assign_trial_splits(records: list[TrialRecord]) -> list[TrialRecord]:
    by_fabric: dict[int, list[TrialRecord]] = {}
    for record in records:
        by_fabric.setdefault(record.fabric_id, []).append(record)

    result: list[TrialRecord] = []
    expected = (
        TRAIN_TRIALS_PER_FABRIC
        + VAL_TRIALS_PER_FABRIC
        + TEST_TRIALS_PER_FABRIC
    )
    if expected != EXPECTED_TRIALS_PER_FABRIC:
        raise ValueError("configured dataset split counts do not sum to the expected trial count")

    for fabric_id in sorted(by_fabric):
        fabric_records = sorted(by_fabric[fabric_id], key=lambda row: row.trial_uid)
        rng = random.Random(RANDOM_SEED + fabric_id)
        rng.shuffle(fabric_records)
        test_rows = fabric_records[:TEST_TRIALS_PER_FABRIC]
        val_start = TEST_TRIALS_PER_FABRIC
        val_end = val_start + VAL_TRIALS_PER_FABRIC
        val_rows = fabric_records[val_start:val_end]
        train_rows = fabric_records[val_end:]
        split_rows = {
            "train": train_rows,
            "val": val_rows,
            "test": test_rows,
        }
        for split, rows in split_rows.items():
            for row in rows:
                result.append(replace(row, split=split))
    return sorted(
        result,
        key=lambda row: (SPLITS.index(row.split), row.fabric_id, row.trial_uid),
    )

def assert_no_trial_leakage(records: list[TrialRecord]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        previous = seen.setdefault(record.trial_uid, record.split)
        if previous != record.split:
            raise RuntimeError(
                f"detected trial data leakage：{record.trial_uid} is present in both "
                f"{previous} and {record.split}"
            )

def write_split_manifest(records: list[TrialRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_uid",
        "split",
        "fabric_id",
        "primitive",
        "position",
        "press_orientation_deg",
        "slide_track",
        "twist_direction",
        "repeat",
    ]
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: getattr(record, key)
                    for key in fieldnames
                }
            )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, output_path)

def audio_metadata(record: TrialRecord) -> tuple[float, int]:
    return 0.0, EXPECTED_AUDIO_SAMPLE_RATE_HZ

def load_audio(record: TrialRecord) -> tuple[np.ndarray, float, int]:
    main_path = record.trial_dir / AUDIO_MAIN_RELATIVE_PATH
    secondary_path = record.trial_dir / AUDIO_SECONDARY_RELATIVE_PATH
    main_rows = np.loadtxt(
        main_path,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1),
        dtype=np.float64,
        ndmin=2,
    )
    secondary_rows = np.loadtxt(
        secondary_path,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1),
        dtype=np.float64,
        ndmin=2,
    )
    if (
        main_rows.ndim != 2
        or main_rows.shape[1] != 2
        or main_rows.shape != secondary_rows.shape
        or main_rows.shape[0] == 0
        or not np.all(np.isfinite(main_rows))
        or not np.all(np.isfinite(secondary_rows))
        or not np.array_equal(main_rows[:, 0], secondary_rows[:, 0])
        or not np.array_equal(main_rows[:, 0], np.arange(main_rows.shape[0]))
    ):
        raise ValueError(f"{record.trial_uid}：invalid or misaligned audio data")
    values = np.asarray(
        main_rows[:, 1] - AUDIO_SECONDARY_GAIN * secondary_rows[:, 1],
        dtype=np.float32,
    )
    start_sec, sample_rate = audio_metadata(record)
    return values, start_sec, sample_rate

def audio_sample_count(record: TrialRecord) -> int:
    counts = []
    for relative_path in (
        AUDIO_MAIN_RELATIVE_PATH,
        AUDIO_SECONDARY_RELATIVE_PATH,
    ):
        path = record.trial_dir / relative_path
        with path.open("r", encoding="utf-8", newline="") as file:
            counts.append(max(sum(1 for _ in file) - 1, 0))
    if counts[0] == 0 or counts[0] != counts[1]:
        raise ValueError(f"{record.trial_uid}：audio channel lengths are inconsistent")
    return counts[0]

def load_accel(record: TrialRecord) -> tuple[np.ndarray, np.ndarray]:
    accel_path = record.trial_dir / ACCEL_RELATIVE_PATH
    rows = np.loadtxt(
        accel_path,
        delimiter=",",
        skiprows=1,
        usecols=(0, 1, 2, 3),
        dtype=np.float64,
        ndmin=2,
    )
    if rows.ndim != 2 or rows.shape[1] != 4 or rows.shape[0] < 2:
        raise ValueError(f"{record.trial_uid}：acceleration data is too short")
    timestamps_ns = rows[:, 0]
    values = rows[:, 1:4].astype(np.float32)
    value_times = (timestamps_ns - timestamps_ns[0]) / 1e9
    if (
        not np.all(np.isfinite(rows))
        or not np.all(np.diff(value_times) > 0)
    ):
        raise ValueError(f"{record.trial_uid}：invalid acceleration data")
    return value_times, values

def calculate_sync_info(
    audio_size: int,
    audio_start_sec: float,
    audio_rate_hz: int,
    accel_times: np.ndarray,
) -> SyncInfo:
    audio_end = audio_start_sec + audio_size / float(audio_rate_hz)
    accel_step = float(np.median(np.diff(accel_times)))
    accel_end = float(accel_times[-1] + accel_step)
    sync_start = max(audio_start_sec, float(accel_times[0]))
    sync_end = min(audio_end, accel_end)
    usable = sync_end - sync_start
    if usable + 1e-9 < WINDOW_SECONDS:
        raise ValueError(
            f"the shared audio and acceleration time interval is only {usable:.6f} seconds"
        )
    num_windows = (
        int(math.floor((usable - WINDOW_SECONDS + 1e-9) / WINDOW_STRIDE_SECONDS))
        + 1
    )
    return SyncInfo(
        audio_start_host_sec=audio_start_sec,
        audio_end_host_sec=audio_end,
        accel_start_host_sec=float(accel_times[0]),
        accel_end_host_sec=accel_end,
        sync_start_host_sec=sync_start,
        sync_end_host_sec=sync_end,
        num_windows=num_windows,
    )

def fixed_slice(
    values: np.ndarray, start_index: int, length: int, context: str
) -> np.ndarray:
    end_index = start_index + length
    if start_index < 0 or end_index > values.size:
        raise ValueError(
            f"{context}：requested interval [{start_index}:{end_index}] "
            f"exceeds the data length {values.size}"
        )
    return np.asarray(values[start_index:end_index], dtype=np.float32)

def temporal_windows(
    record: TrialRecord,
) -> tuple[list[np.ndarray], list[np.ndarray], SyncInfo]:
    audio, audio_start, source_audio_rate = load_audio(record)
    accel_times, accel_values = load_accel(record)
    sync = calculate_sync_info(
        audio.size, audio_start, source_audio_rate, accel_times
    )
    if (
        STRICT_VALIDATION
        and source_audio_rate != EXPECTED_AUDIO_SAMPLE_RATE_HZ
    ):
        raise ValueError(
            f"{record.trial_uid}：expected raw audio sample rate is "
            f"{EXPECTED_AUDIO_SAMPLE_RATE_HZ} Hz，actual {source_audio_rate} Hz"
        )

    audio_length = int(round(WINDOW_SECONDS * source_audio_rate))
    audio_windows: list[np.ndarray] = []
    accel_windows: list[np.ndarray] = []

    for window_index in range(sync.num_windows):
        relative_start = window_index * WINDOW_STRIDE_SECONDS
        window_start = sync.sync_start_host_sec + relative_start
        audio_start_index = int(
            round(
                (window_start - sync.audio_start_host_sec)
                * source_audio_rate
            )
        )
        audio_window = fixed_slice(
            audio,
            audio_start_index,
            audio_length,
            f"{record.trial_uid} audio window {window_index}",
        )
        if AUDIO_PEAK_NORMALIZE_PER_WINDOW:
            peak = float(np.max(np.abs(audio_window)))
            if peak > NORMALIZATION_EPSILON:
                audio_window = audio_window / peak

        window_end = window_start + WINDOW_SECONDS
        accel_mask = (accel_times >= window_start) & (
            accel_times < window_end
        )
        accel_window = np.asarray(
            accel_values[accel_mask], dtype=np.float32
        )
        if accel_window.shape[0] == 0:
            raise ValueError(
                f"{record.trial_uid}：acceleration window {window_index} "
                "contains no raw sample points"
            )
        if ACCEL_ZSCORE_PER_WINDOW:
            mean = accel_window.mean(axis=0, keepdims=True)
            std = accel_window.std(axis=0, keepdims=True)
            accel_window = (accel_window - mean) / np.maximum(
                std, NORMALIZATION_EPSILON
            )
        audio_windows.append(audio_window)
        accel_windows.append(accel_window)
    return audio_windows, accel_windows, sync

class TemporalH5Writer:
    def __init__(self, path: Path, overwrite: bool) -> None:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError(
                "writing HDF5 requires h5py，install the project dependencies first。"
            ) from error

        self.h5py = h5py
        self.final_path = path
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"output file already exists：{path}。use this option to replace them --overwrite。"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = path.with_suffix(path.suffix + ".tmp")
        self.temp_path.unlink(missing_ok=True)
        self.file = h5py.File(self.temp_path, "w")
        self.count = 0
        self._create_datasets()

    def _create_datasets(self) -> None:
        audio_length = int(
            round(WINDOW_SECONDS * EXPECTED_AUDIO_SAMPLE_RATE_HZ)
        )
        compression_args = {
            "compression": HDF5_COMPRESSION,
            "compression_opts": HDF5_COMPRESSION_LEVEL,
        }
        self.audio = self.file.create_dataset(
            "audio",
            shape=(0, audio_length),
            maxshape=(None, audio_length),
            chunks=(1, audio_length),
            dtype="float32",
            **compression_args,
        )
        accel_vlen_dtype = self.h5py.vlen_dtype(np.dtype("float32"))
        self.accel = self.file.create_dataset(
            "accel",
            shape=(0,),
            maxshape=(None,),
            chunks=True,
            dtype=accel_vlen_dtype,
        )
        labels_group = self.file.create_group("labels")
        metadata_group = self.file.create_group("metadata")
        self.labels = {
            name: labels_group.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                dtype="int32",
                **compression_args,
            )
            for name in (
                "fabric_id",
                "primitive",
                "position",
                "press_orientation_deg",
                "slide_track",
                "twist_direction",
                "repeat",
            )
        }
        string_dtype = self.h5py.string_dtype(encoding="utf-8")
        self.trial_uid = metadata_group.create_dataset(
            "trial_uid",
            shape=(0,),
            maxshape=(None,),
            chunks=True,
            dtype=string_dtype,
        )
        self.window_index = metadata_group.create_dataset(
            "window_index",
            shape=(0,),
            maxshape=(None,),
            chunks=True,
            dtype="int32",
            **compression_args,
        )
        self.window_start = metadata_group.create_dataset(
            "window_start_host_sec",
            shape=(0,),
            maxshape=(None,),
            chunks=True,
            dtype="float64",
            **compression_args,
        )
        self.window_end = metadata_group.create_dataset(
            "window_end_host_sec",
            shape=(0,),
            maxshape=(None,),
            chunks=True,
            dtype="float64",
            **compression_args,
        )
        self.accel_sample_count = metadata_group.create_dataset(
            "accel_sample_count",
            shape=(0,),
            maxshape=(None,),
            chunks=True,
            dtype="int32",
            **compression_args,
        )
        self.file.attrs.update(
            {
                "random_seed": RANDOM_SEED,
                "window_seconds": WINDOW_SECONDS,
                "window_stride_seconds": WINDOW_STRIDE_SECONDS,
                "audio_sample_rate_hz": EXPECTED_AUDIO_SAMPLE_RATE_HZ,
                "audio_resampled": False,
                "accel_resampled": False,
                "accel_storage": "variable_length_interleaved_xyz",
                "accel_source": ACCEL_RELATIVE_PATH.as_posix(),
                "accel_channels": 3,
                "accel_axis_order": "x,y,z",
                "audio_peak_normalized_per_window": (
                    AUDIO_PEAK_NORMALIZE_PER_WINDOW
                ),
                "accel_zscore_per_window": ACCEL_ZSCORE_PER_WINDOW,
                "conditional_label_missing_value": -1,
                "primitive_mapping": json.dumps(PRIMITIVE_TO_ID),
                "position_mapping": json.dumps(POSITION_TO_ID),
                "slide_track_mapping": json.dumps(TRACK_TO_ID),
                "twist_direction_mapping": json.dumps(
                    TWIST_DIRECTION_TO_ID
                ),
            }
        )

    @staticmethod
    def _resize(dataset: Any, size: int) -> None:
        dataset.resize(size, axis=0)

    def append_trial(
        self,
        record: TrialRecord,
        audio_windows: list[np.ndarray],
        accel_windows: list[np.ndarray],
        sync: SyncInfo,
    ) -> None:
        size = len(audio_windows)
        if size != len(accel_windows) or size != sync.num_windows:
            raise ValueError(f"{record.trial_uid}：modality window counts are inconsistent")
        start = self.count
        end = start + size
        for dataset in (
            self.audio,
            self.accel,
            self.trial_uid,
            self.window_index,
            self.window_start,
            self.window_end,
            self.accel_sample_count,
            *self.labels.values(),
        ):
            self._resize(dataset, end)

        self.audio[start:end] = np.stack(audio_windows)
        for offset, accel_window in enumerate(accel_windows):
            if accel_window.ndim != 2 or accel_window.shape[1] != 3:
                raise ValueError(
                    f"{record.trial_uid}：acceleration window must be [T, 3]，"
                    f"actual {accel_window.shape}"
                )
            self.accel[start + offset] = accel_window.reshape(-1)
        self.accel_sample_count[start:end] = [
            window.shape[0] for window in accel_windows
        ]
        labels = record.label_values()
        for name, dataset in self.labels.items():
            dataset[start:end] = labels[name]
        self.trial_uid[start:end] = [record.trial_uid] * size
        indices = np.arange(size, dtype=np.int32)
        starts = (
            sync.sync_start_host_sec
            + indices.astype(np.float64) * WINDOW_STRIDE_SECONDS
        )
        self.window_index[start:end] = indices
        self.window_start[start:end] = starts
        self.window_end[start:end] = starts + WINDOW_SECONDS
        self.count = end

    def close(self, success: bool) -> None:
        self.file.attrs["num_samples"] = self.count
        self.file.flush()
        self.file.close()
        if success:
            os.replace(self.temp_path, self.final_path)
        else:
            self.temp_path.unlink(missing_ok=True)

def process_temporal(
    records: list[TrialRecord], output_root: Path, overwrite: bool
) -> tuple[dict[str, SyncInfo], dict[str, int]]:
    sync_by_trial: dict[str, SyncInfo] = {}
    counts: dict[str, int] = {}
    for split in SPLITS:
        split_records = [row for row in records if row.split == split]
        output_path = output_root / f"temporal_{split}.h5"
        writer = TemporalH5Writer(output_path, overwrite=overwrite)
        success = False
        try:
            for index, record in enumerate(split_records, start=1):
                audio_windows, accel_windows, sync = temporal_windows(record)
                writer.append_trial(
                    record, audio_windows, accel_windows, sync
                )
                sync_by_trial[record.trial_uid] = sync
                if index % 10 == 0 or index == len(split_records):
                    print(
                        f"[temporal:{split}] processed {index}/{len(split_records)} "
                        f" trial，total {writer.count} windows",
                        flush=True,
                    )
            counts[split] = writer.count
            success = True
        finally:
            writer.close(success=success)
    return sync_by_trial, counts

def read_gelsight_rows(record: TrialRecord) -> list[tuple[int, Path, float]]:
    image_dir = record.trial_dir / GELSIGHT_IMAGE_DIR_RELATIVE_PATH
    image_paths = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: natural_key(path.name),
    )
    rows: list[tuple[int, Path, float]] = []
    for image_path in image_paths:
        if not image_path.stem.isdigit():
            raise ValueError(f"invalid GelSight image name：{image_path.name}")
        sample_idx = int(image_path.stem)
        rows.append(
            (sample_idx, image_path, sample_idx / GELSIGHT_SAMPLE_RATE_HZ)
        )
    if not rows:
        raise RuntimeError(f"{record.trial_uid}：contains no GelSight images")
    if STRICT_VALIDATION and [row[0] for row in rows] != list(range(len(rows))):
        raise RuntimeError(
            f"{record.trial_uid}：GelSight image indices are not continuous"
        )
    return rows

def resize_rgb_image(path: Path) -> np.ndarray:
    from PIL import Image

    resampling = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }[GELSIGHT_RESAMPLE]
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(IMAGE_SIZE, resample=resampling)
        array = np.asarray(resized, dtype=np.uint8)
    return np.transpose(array, (2, 0, 1))

def compute_sync_only(record: TrialRecord) -> SyncInfo:
    audio_size = audio_sample_count(record)
    audio_start, audio_rate = audio_metadata(record)
    accel_times, _ = load_accel(record)
    return calculate_sync_info(
        audio_size, audio_start, audio_rate, accel_times
    )

def nearest_window_index(host_ts: float, sync: SyncInfo) -> int:
    if host_ts < sync.sync_start_host_sec or host_ts > sync.sync_end_host_sec:
        return -1
    relative = host_ts - sync.sync_start_host_sec
    index = int(round((relative - WINDOW_SECONDS / 2) / WINDOW_STRIDE_SECONDS))
    return min(max(index, 0), sync.num_windows - 1)

def atomic_torch_save(payload: dict[str, Any], path: Path, overwrite: bool) -> None:
    import torch

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output file already exists：{path}。use this option to replace them --overwrite。"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(
        payload,
        temporary,
        _use_new_zipfile_serialization=PT_USE_NEW_ZIPFILE_SERIALIZATION,
    )
    os.replace(temporary, path)

def process_gelsight(
    records: list[TrialRecord],
    output_root: Path,
    overwrite: bool,
    sync_by_trial: dict[str, SyncInfo],
) -> dict[str, int]:
    import torch

    counts: dict[str, int] = {}
    channels, height, width = 3, IMAGE_SIZE[1], IMAGE_SIZE[0]
    for split in SPLITS:
        split_records = [row for row in records if row.split == split]
        frame_rows = {
            record.trial_uid: read_gelsight_rows(record)
            for record in split_records
        }
        total = sum(len(rows) for rows in frame_rows.values())
        counts[split] = total
        output_root.mkdir(parents=True, exist_ok=True)
        descriptor, mmap_name = tempfile.mkstemp(
            dir=output_root,
            prefix=f".gelsight_{split}.",
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
            for trial_index, record in enumerate(split_records):
                sync = sync_by_trial.get(record.trial_uid)
                if sync is None:
                    sync = compute_sync_only(record)
                    sync_by_trial[record.trial_uid] = sync
                label_values = record.label_values()
                for source_frame_index, image_path, host_ts in frame_rows[
                    record.trial_uid
                ]:
                    images[cursor] = resize_rgb_image(image_path)
                    for name in label_names:
                        labels[name][cursor] = label_values[name]
                    trial_indices[cursor] = trial_index
                    source_frame_indices[cursor] = source_frame_index
                    host_timestamps[cursor] = host_ts
                    relative_timestamps[cursor] = (
                        host_ts - sync.sync_start_host_sec
                    )
                    window_indices[cursor] = nearest_window_index(host_ts, sync)
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
                    name: torch.from_numpy(values) for name, values in labels.items()
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
                    "image_size": list(IMAGE_SIZE),
                    "image_dtype": "uint8",
                    "image_layout": "NCHW_RGB",
                    "primitive_mapping": PRIMITIVE_TO_ID,
                    "position_mapping": POSITION_TO_ID,
                    "slide_track_mapping": TRACK_TO_ID,
                    "twist_direction_mapping": TWIST_DIRECTION_TO_ID,
                    "window_seconds": WINDOW_SECONDS,
                    "window_stride_seconds": WINDOW_STRIDE_SECONDS,
                },
            }
            atomic_torch_save(
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

def split_statistics(records: list[TrialRecord]) -> dict[str, Any]:
    fabrics = sorted({record.fabric_id for record in records})
    stats: dict[str, Any] = {
        "num_fabrics": len(fabrics),
        "fabric_ids": fabrics,
        "total_trials": len(records),
        "splits": {},
    }
    for split in SPLITS:
        rows = [record for record in records if record.split == split]
        stats["splits"][split] = {
            "trials": len(rows),
            "per_fabric": {
                str(fabric_id): sum(row.fabric_id == fabric_id for row in rows)
                for fabric_id in fabrics
            },
            "primitive_counts": {
                primitive: sum(row.primitive == primitive for row in rows)
                for primitive in PRIMITIVE_TO_ID
            },
        }
    return stats

def check_output_targets(
    output_root: Path, modes: set[str], overwrite: bool
) -> None:
    paths: list[Path] = []
    if "temporal" in modes:
        paths.extend(output_root / f"temporal_{split}.h5" for split in SPLITS)
    if "gelsight" in modes:
        paths.extend(output_root / f"gelsight_{split}.pt" for split in SPLITS)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "the following outputs already exist，use this option to replace them --overwrite："
            + ", ".join(str(path) for path in existing)
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="offline-preprocess synchronized temporal modalities and GelSight modality。"
    )
    parser.add_argument(
        "--mode",
        choices=("all", "temporal", "gelsight", "scan"),
        default="all",
        help="select modalities to generate；scan perform read-only validation only。",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan only、validate and print statistics，do not write files。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_OUTPUTS,
        help="overwrite existing generated results using atomic replacement。",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_reproducible_seed(RANDOM_SEED)
    fabric_ids = load_classification_fabric_ids(args.config)
    print(f"scanning dataset：{args.dataset_root}", flush=True)
    records = assign_trial_splits(
        discover_trials(args.dataset_root, fabric_ids)
    )
    assert_no_trial_leakage(records)
    stats = split_statistics(records)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run or args.mode == "scan":
        print("read-only validation completed：no files were written。", flush=True)
        return

    modes = (
        {"temporal", "gelsight"}
        if args.mode == "all"
        else {args.mode}
    )
    check_output_targets(args.output_root, modes, args.overwrite)
    write_split_manifest(
        records, args.output_root / SPLIT_MANIFEST_NAME
    )

    sync_by_trial: dict[str, SyncInfo] = {}
    if "temporal" in modes:
        sync_by_trial, temporal_counts = process_temporal(
            records, args.output_root, args.overwrite
        )
        stats["temporal_windows"] = temporal_counts
    if "gelsight" in modes:
        gelsight_counts = process_gelsight(
            records,
            args.output_root,
            args.overwrite,
            sync_by_trial,
        )
        stats["gelsight_frames"] = gelsight_counts

    summary_path = args.output_root / SUMMARY_NAME
    summary_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"processing completed，output directory：{args.output_root}", flush=True)

if __name__ == "__main__":
    main()
