from __future__ import annotations
from pathlib import Path
from typing import Any
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
SPLITS = ("train", "val", "test")
SUPPORTED_MODALITIES = {"audio", "accel", "gelsight", "visual"}
def _log_mel_spectrogram(
    segment: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    top_db: float = 80.0,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=top_db)
    return log_mel.astype(np.float32)

def resolve_offline_root(
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> Path:
    value = config.get("preprocessed_root", config.get("dataset_root"))
    if value is None:
        raise KeyError("offline data configuration is missing preprocessed_root")
    path = Path(value)
    if not path.is_absolute() and project_root is not None:
        path = Path(project_root) / path
    return path

def load_offline_trial_tables(
    input_root: str | Path,
    manifest_name: str = "trial_split_manifest.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(input_root) / manifest_name
    if not path.is_file():
        raise FileNotFoundError(f"cannot find offline trial split manifest：{path}")
    manifest = pd.read_csv(path)
    required = {"trial_uid", "split", "fabric_id"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing fields：{sorted(missing)}")
    invalid_splits = set(manifest["split"].astype(str)).difference(SPLITS)
    if invalid_splits:
        raise ValueError(f"{path} contains unknown split：{sorted(invalid_splits)}")

    metadata = manifest.rename(columns={"trial_uid": "trial_id"}).copy()
    metadata["label"] = metadata["fabric_id"].astype(int).astype(str)
    split_df = metadata[["trial_id", "split", "label"]].copy()
    metadata = metadata.drop(columns=["split"])
    return metadata, split_df

def load_offline_classification_tables(
    config: dict[str, Any],
    project_root: str | Path | None = None,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    root = resolve_offline_root(config, project_root)
    offline_cfg = config.get("offline", {})
    metadata, split_df = load_offline_trial_tables(
        root,
        str(offline_cfg.get("trial_manifest", "trial_split_manifest.csv")),
    )
    return root, metadata, split_df

def _decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]

def _torch_load_mmap(path: Path) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")

class OfflineClassificationDataset(Dataset):

    def __init__(
        self,
        input_root: str | Path,
        split: str,
        label_to_index: dict[str, int],
        modalities: list[str],
        config: dict[str, Any],
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split：{split}")
        normalized = [str(modality).lower() for modality in modalities]
        unknown = set(normalized).difference(SUPPORTED_MODALITIES)
        if unknown:
            raise ValueError(f"unknown modality：{sorted(unknown)}")
        if not normalized:
            raise ValueError("modalities cannot be empty")

        self.input_root = Path(input_root)
        self.split = split
        self.label_to_index = label_to_index
        self.modalities = normalized
        self.config = config
        self.offline_cfg = config.get("offline", {})
        self._h5_file: Any | None = None
        self._pt_payloads: dict[str, dict[str, Any]] = {}

        temporal_pattern = str(
            self.offline_cfg.get("temporal_pattern", "temporal_{split}.h5")
        )
        self.temporal_path = self.input_root / temporal_pattern.format(split=split)
        self.gelsight_path = self.input_root / str(
            self.offline_cfg.get(
                "gelsight_pattern", "gelsight_{split}.pt"
            )
        ).format(split=split)
        self.visual_path = self.input_root / str(
            self.offline_cfg.get("visual_pattern", "visual_{split}.pt")
        ).format(split=split)
        self._validate_paths()
        self._read_temporal_index()

        self.gelsight_indices: np.ndarray | None = None
        self.visual_indices: np.ndarray | None = None
        if "gelsight" in self.modalities:
            self.gelsight_frames_per_window = int(
                config.get("gelsight", {}).get(
                    "frames_per_window", 3
                )
            )
            if self.gelsight_frames_per_window < 1:
                raise ValueError(
                    "gelsight.frames_per_window must be greater than or equal to 1"
                )
            self.gelsight_indices, gelsight_mask = self._build_gelsight_pairs()
            self._filter_temporal_rows(gelsight_mask)
            self.gelsight_indices = self.gelsight_indices[gelsight_mask]
        if "visual" in self.modalities:
            self.visual_indices = self._build_visual_pairs()

    def _validate_paths(self) -> None:
        required = [self.temporal_path]
        if "gelsight" in self.modalities:
            required.append(self.gelsight_path)
        if "visual" in self.modalities:
            required.append(self.visual_path)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing offline-preprocessed files：{missing}")

    @staticmethod
    def _import_h5py() -> Any:
        try:
            import h5py
        except ImportError as error:
            raise ImportError(
                "reading temporal_*.h5 requires h5py，run first pip install h5py。"
            ) from error
        return h5py

    def _read_temporal_index(self) -> None:
        h5py = self._import_h5py()
        with h5py.File(self.temporal_path, "r") as file:
            required = (
                "audio",
                "accel",
                "labels/fabric_id",
                "metadata/trial_uid",
                "metadata/window_index",
                "metadata/window_start_host_sec",
                "metadata/window_end_host_sec",
                "metadata/accel_sample_count",
            )
            missing = [name for name in required if name not in file]
            if missing:
                raise ValueError(
                    f"{self.temporal_path} is missing dataset：{missing}"
                )
            self.fabric_ids = np.asarray(
                file["labels/fabric_id"][:], dtype=np.int64
            )
            self.trial_ids = _decode_strings(
                file["metadata/trial_uid"][:]
            )
            self.window_indices = np.asarray(
                file["metadata/window_index"][:], dtype=np.int64
            )
            self.window_starts = np.asarray(
                file["metadata/window_start_host_sec"][:],
                dtype=np.float64,
            )
            self.window_ends = np.asarray(
                file["metadata/window_end_host_sec"][:],
                dtype=np.float64,
            )
            self.accel_channels = int(file.attrs.get("accel_channels", 1))
            self.accel_axis_order = str(file.attrs.get("accel_axis_order", ""))
            self.accel_sample_counts = np.asarray(
                file["metadata/accel_sample_count"][:], dtype=np.int64
            )
            if "accel" in self.modalities and (
                self.accel_channels != 3
                or self.accel_axis_order != "x,y,z"
            ):
                raise ValueError(
                    f"{self.temporal_path} is not strictly aligned raw three-axis acceleration H5；"
                    "rerun preprocess_class_temporal_gelsight.py"
                )
            self.audio_sample_rate = int(
                file.attrs.get(
                    "audio_sample_rate_hz",
                    self.config.get("audio", {}).get("sample_rate", 44_100),
                )
            )
            self.window_seconds = float(
                file.attrs.get(
                    "window_seconds",
                    self.config.get("window", {}).get("length_sec", 0.5),
                )
            )
            self.window_stride_seconds = float(
                file.attrs.get(
                    "window_stride_seconds",
                    self.window_seconds
                    * (
                        1.0
                        - float(
                            self.config.get("window", {}).get(
                                "overlap", 0.5
                            )
                        )
                    ),
                )
            )
            size = len(file["audio"])
        lengths = {
            len(self.fabric_ids),
            len(self.trial_ids),
            len(self.window_indices),
            len(self.window_starts),
            len(self.window_ends),
            size,
        }
        if len(lengths) != 1:
            raise ValueError(f"{self.temporal_path} index lengths are inconsistent")
        self.row_indices = np.arange(size, dtype=np.int64)

    def _filter_temporal_rows(self, mask: np.ndarray) -> None:
        if mask.dtype != np.bool_ or mask.shape != self.row_indices.shape:
            raise ValueError("offline temporal filter mask has invalid shape")
        if bool(np.all(mask)):
            return
        if not bool(np.any(mask)):
            raise ValueError(
                f"{self.gelsight_path} has no temporal windows with GelSight frames"
            )
        self.fabric_ids = self.fabric_ids[mask]
        self.trial_ids = [
            trial_id for trial_id, keep in zip(self.trial_ids, mask) if keep
        ]
        self.window_indices = self.window_indices[mask]
        self.window_starts = self.window_starts[mask]
        self.window_ends = self.window_ends[mask]
        self.accel_sample_counts = self.accel_sample_counts[mask]
        self.row_indices = self.row_indices[mask]

    def _get_h5(self) -> Any:
        if self._h5_file is None:
            h5py = self._import_h5py()
            self._h5_file = h5py.File(self.temporal_path, "r")
        return self._h5_file

    def _get_pt(self, modality: str) -> dict[str, Any]:
        if modality not in self._pt_payloads:
            path = (
                self.gelsight_path
                if modality == "gelsight"
                else self.visual_path
            )
            self._pt_payloads[modality] = _torch_load_mmap(path)
        return self._pt_payloads[modality]

    def _build_gelsight_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        payload = self._get_pt("gelsight")
        metadata = payload.get("metadata", {})
        required = {
            "trial_index",
            "time_from_sync_start_sec",
            "trial_uids",
        }
        missing = required.difference(metadata)
        if missing:
            raise ValueError(
                f"{self.gelsight_path} metadata is missing fields：{sorted(missing)}"
            )
        trial_uids = [str(value) for value in metadata["trial_uids"]]
        trial_indices = metadata["trial_index"].cpu().numpy()
        relative_times = metadata[
            "time_from_sync_start_sec"
        ].cpu().numpy()

        frames_by_trial: dict[str, list[tuple[float, int]]] = {}
        for frame_index in range(len(trial_indices)):
            trial_uid = trial_uids[int(trial_indices[frame_index])]
            relative_time = float(relative_times[frame_index])
            frames_by_trial.setdefault(trial_uid, []).append(
                (relative_time, frame_index)
            )
        for frames in frames_by_trial.values():
            frames.sort(key=lambda item: (item[0], item[1]))

        frames_per_window = self.gelsight_frames_per_window
        pairs = np.full(
            (len(self), frames_per_window),
            fill_value=-1,
            dtype=np.int64,
        )
        valid_mask = np.ones(len(self), dtype=np.bool_)
        selection_seed = int(
            self.config.get("gelsight", {}).get(
                "selection_seed",
                self.offline_cfg.get("pairing_seed", 20260730),
            )
        )
        split_code = SPLITS.index(self.split)
        for index, (trial_uid, window_index, fabric_id) in enumerate(
            zip(
                self.trial_ids,
                self.window_indices,
                self.fabric_ids,
            )
        ):
            frames = frames_by_trial.get(trial_uid, [])
            if not frames:
                valid_mask[index] = False
                continue
            window_start = (
                int(window_index) * self.window_stride_seconds
            )
            window_end = window_start + self.window_seconds
            candidates = [
                frame
                for frame in frames
                if window_start <= frame[0] < window_end
            ]
            if not candidates:
                valid_mask[index] = False
                continue

            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        selection_seed,
                        split_code,
                        index,
                        int(window_index),
                        int(fabric_id),
                    ]
                )
            )
            positions = rng.choice(
                len(candidates),
                size=frames_per_window,
                replace=len(candidates) < frames_per_window,
            )
            selected = sorted(
                (candidates[int(position)] for position in positions),
                key=lambda item: (item[0], item[1]),
            )
            pairs[index] = [frame_index for _, frame_index in selected]
        if not bool(np.any(valid_mask)):
            raise ValueError(
                f"{self.gelsight_path} has no temporal windows with GelSight frames"
            )
        return pairs, valid_mask

    def _build_visual_pairs(self) -> np.ndarray:
        payload = self._get_pt("visual")
        try:
            visual_fabrics = payload["labels"]["fabric_id"].cpu().numpy()
        except KeyError as error:
            raise ValueError(
                f"{self.visual_path} is missing labels/fabric_id"
            ) from error
        by_fabric: dict[int, np.ndarray] = {
            int(fabric_id): np.flatnonzero(visual_fabrics == fabric_id)
            for fabric_id in np.unique(visual_fabrics)
        }
        seed = int(self.offline_cfg.get("pairing_seed", 20260730))
        pairs = np.empty(len(self), dtype=np.int64)
        for index, fabric_id in enumerate(self.fabric_ids):
            candidates = by_fabric.get(int(fabric_id))
            if candidates is None or len(candidates) == 0:
                raise ValueError(
                    f"{self.visual_path} is missing fabric_id={fabric_id}"
                )
            pairs[index] = candidates[(index + seed) % len(candidates)]
        return pairs

    @staticmethod
    def _normalize_image(image: torch.Tensor) -> torch.Tensor:
        tensor = image.to(dtype=torch.float32).div_(255.0)
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return (tensor - mean) / std

    def _normalize_accel_for_deep(self, accel: np.ndarray) -> np.ndarray:
        model_accel_cfg = (
            self.config.get("model", {})
            .get("modalities", {})
            .get("accel", {})
        )
        mode = str(model_accel_cfg.get("normalization", "none")).lower()
        if mode in {"", "none", "false"}:
            return accel.astype(np.float32, copy=False)
        if mode not in {"zscore", "z-score", "standard", "standardize"}:
            raise ValueError(f"unsupported accel normalization: {mode}")

        epsilon = float(model_accel_cfg.get("normalization_epsilon", 1e-8))
        mean = np.mean(accel, axis=0, keepdims=True)
        std = np.std(accel, axis=0, keepdims=True)
        return ((accel - mean) / np.maximum(std, epsilon)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.fabric_ids)

    def sample_metadata(self, index: int) -> dict[str, Any]:
        trial_id = self.trial_ids[index]
        window_index = int(self.window_indices[index])
        return {
            "trial_id": trial_id,
            "window_id": f"{trial_id}::window_{window_index:04d}",
            "window_index": window_index,
            "window_start": float(self.window_starts[index]),
            "window_end": float(self.window_ends[index]),
            "start_sec": window_index * self.window_stride_seconds,
            "end_sec": (
                window_index * self.window_stride_seconds
                + self.window_seconds
            ),
            "fabric_id": int(self.fabric_ids[index]),
            "split": self.split,
            "label": str(int(self.fabric_ids[index])),
            "session_id": "",
            "operator_id": "",
        }

    def paired_source_index(self, index: int, modality: str) -> int:
        return int(self.paired_source_indices(index, modality)[0])

    def paired_source_indices(
        self,
        index: int,
        modality: str,
    ) -> np.ndarray:
        if modality == "gelsight" and self.gelsight_indices is not None:
            return np.asarray(
                self.gelsight_indices[index],
                dtype=np.int64,
            ).reshape(-1)
        if modality == "visual" and self.visual_indices is not None:
            return np.asarray(
                [self.visual_indices[index]],
                dtype=np.int64,
            )
        return np.asarray([self.row_indices[index]], dtype=np.int64)

    def load_raw_modality(self, index: int, modality: str) -> np.ndarray:
        row_index = int(self.row_indices[index])
        if modality == "audio":
            return np.asarray(
                self._get_h5()["audio"][row_index], dtype=np.float32
            )
        if modality == "accel":
            flattened = np.asarray(
                self._get_h5()["accel"][row_index], dtype=np.float32
            )
            sample_count = int(self.accel_sample_counts[index])
            expected_size = sample_count * self.accel_channels
            if flattened.size != expected_size:
                raise ValueError(
                    f"{self.temporal_path} of accel[{index}] length is "
                    f"{flattened.size}，expected {expected_size}"
                )
            return flattened.reshape(sample_count, self.accel_channels)
        if modality == "gelsight":
            source_indices = self.paired_source_indices(index, modality)
            images = self._get_pt(modality)["images"][
                torch.from_numpy(source_indices)
            ]
            return np.ascontiguousarray(
                images.cpu().numpy().transpose(0, 2, 3, 1)
            )
        if modality == "visual":
            source_index = self.paired_source_index(index, modality)
            image = self._get_pt(modality)["images"][source_index]
            return np.ascontiguousarray(
                image.cpu().numpy().transpose(1, 2, 0)
            )
        raise ValueError(f"unknown modality：{modality}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        metadata = self.sample_metadata(index)
        label_name = metadata["label"]
        if label_name not in self.label_to_index:
            raise KeyError(f"label mapping does not contain fabric_id={label_name}")
        item: dict[str, Any] = {
            "label": torch.tensor(
                self.label_to_index[label_name], dtype=torch.long
            ),
            "trial_id": metadata["trial_id"],
            "window_id": metadata["window_id"],
        }
        for modality in self.modalities:
            if modality == "audio":
                audio_cfg = self.config.get("audio", {})
                waveform = self.load_raw_modality(index, modality)
                feature = _log_mel_spectrogram(
                    waveform,
                    sample_rate=self.audio_sample_rate,
                    n_mels=int(audio_cfg.get("n_mels", 64)),
                    n_fft=int(audio_cfg.get("n_fft", 1024)),
                    hop_length=int(audio_cfg.get("hop_length", 256)),
                    win_length=int(audio_cfg.get("win_length", 1024)),
                    top_db=float(audio_cfg.get("top_db", 80.0)),
                )
                item["audio"] = torch.from_numpy(feature)
            elif modality == "accel":
                accel = self.load_raw_modality(index, modality)
                accel = self._normalize_accel_for_deep(accel)
                item["accel"] = torch.from_numpy(accel.T.copy())
                item["accel_length"] = int(accel.shape[0])
            elif modality == "gelsight":
                source_indices = self.paired_source_indices(
                    index, modality
                )
                images = self._get_pt(modality)["images"][
                    torch.from_numpy(source_indices)
                ]
                item[modality] = self._normalize_image(images)
            else:
                source_index = self.paired_source_index(index, modality)
                image = self._get_pt(modality)["images"][source_index]
                item[modality] = self._normalize_image(image)
        return item

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_pt_payloads"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

def offline_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch or "accel" not in batch[0]:
        return default_collate(batch)
    lengths = torch.tensor(
        [int(item["accel"].shape[-1]) for item in batch],
        dtype=torch.long,
    )
    max_length = int(lengths.max())
    channels = int(batch[0]["accel"].shape[0])
    padded = batch[0]["accel"].new_zeros(
        (len(batch), channels, max_length)
    )
    for row, item in enumerate(batch):
        length = int(item["accel"].shape[-1])
        padded[row, :, max_length - length :] = item["accel"]

    remaining = [
        {
            key: value
            for key, value in item.items()
            if key not in {"accel", "accel_length"}
        }
        for item in batch
    ]
    output = default_collate(remaining)
    output["accel"] = padded
    output["accel_length"] = lengths
    return output
