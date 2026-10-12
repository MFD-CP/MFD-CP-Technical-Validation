from __future__ import annotations
import copy
import re
from pathlib import Path
from typing import Any
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from src.utils.config import save_yaml

def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_")


def seed_list(value: Any) -> list[int]:
    return [int(x) for x in value] if isinstance(value, list) else [int(value)]


def apply_encoder_override(
    config: dict[str, Any],
    encoder_by_modality: dict[str, Any],
) -> dict[str, Any]:
    run_cfg = copy.deepcopy(config)
    run_cfg.setdefault("model", {})
    run_cfg["model"]["encoder_by_modality"] = {
        str(modality).lower(): encoder
        for modality, encoder in encoder_by_modality.items()
    }
    return run_cfg

def prepare_run_dirs(
    project_root: Path,
    experiment_name: str,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Path]:
    safe_name = sanitize_name(experiment_name)
    checkpoint_root = project_root / config.get("checkpoint_dir", "checkpoints")
    log_root = project_root / config.get("log_dir", "logs")
    run_dir = checkpoint_root / safe_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_copy_path = run_dir / "config.yaml"
    save_yaml(config, config_copy_path)
    return {
        "safe_name": Path(safe_name),
        "checkpoint_dir": run_dir,
        "tensorboard_dir": log_root / "tensorboard",
        "csv_log_dir": log_root / "csv",
        "config_copy": config_copy_path,
    }

def build_loggers(
    project_root: Path,
    experiment_name: str,
    seed: int,
    config: dict[str, Any],
):
    safe_name = sanitize_name(experiment_name)
    log_root = project_root / config.get("log_dir", "logs")
    version = f"seed_{seed}"
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(log_root / "tensorboard"),
        name=safe_name,
        version=version,
    )
    csv_logger = CSVLogger(
        save_dir=str(log_root / "csv"),
        name=safe_name,
        version=version,
    )
    return [tensorboard_logger, csv_logger]

def build_callbacks(checkpoint_dir: Path, config: dict[str, Any]) -> list[Any]:
    callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-{epoch:03d}-{val_f1_trial:.4f}",
            monitor="val_f1_trial",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    patience = int(config.get("early_stopping_patience", 15))
    if patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_f1_trial",
                mode="max",
                patience=patience,
                verbose=True,
            )
        )
    return callbacks

def find_resume_checkpoint(checkpoint_dir: Path, config: dict[str, Any]) -> str | None:
    resume_cfg = config.get("resume", {})
    if not bool(resume_cfg.get("auto", True)):
        explicit = resume_cfg.get("path")
        return str(explicit) if explicit else None
    explicit = resume_cfg.get("path")
    if explicit:
        path = Path(explicit)
        return str(path if path.is_absolute() else checkpoint_dir / path)
    last_ckpt = checkpoint_dir / "last.ckpt"
    return str(last_ckpt) if last_ckpt.exists() else None

def metric_from_test_result(test_result: dict[str, Any], key: str) -> float:
    return float(test_result.get(f"test_{key}", test_result.get(key, 0.0)))

def torch_precision_setting(config: dict[str, Any]) -> str | int:
    precision = config.get("precision", "32-true")
    return int(precision) if isinstance(precision, int) else str(precision)

def compute_num_devices(config: dict[str, Any]) -> int | str:
    devices = config.get("devices", "auto")
    return devices
