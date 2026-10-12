from __future__ import annotations
import argparse
import sys
from functools import reduce
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from src.features.feature_cache import (
    BASE_COLUMNS,
    feature_columns,
    load_feature_table,
)
from src.training.metrics import compute_window_and_trial_metrics
from src.utils.config import load_yaml
from src.utils.io import write_csv
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

METRICS = (
    "accuracy_window",
    "precision_window",
    "recall_window",
    "f1_window",
    "accuracy_trial",
    "precision_trial",
    "recall_trial",
    "f1_trial",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train Interaction XGBoost")
    parser.add_argument("--config", required=True)
    return parser.parse_args()

def resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def seed_list(value: Any) -> list[int]:
    return [int(seed) for seed in value] if isinstance(value, list) else [int(value)]

def prepare_combo(cache_dir: Path, combo: list[str]) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for modality in combo:
        frame = load_feature_table(cache_dir, modality)
        columns = feature_columns(frame)
        table = frame[BASE_COLUMNS + columns].copy()
        table = table.rename(
            columns={name: f"{modality}_{name}" for name in columns}
        )
        tables.append(table)
    return reduce(
        lambda left, right: pd.merge(
            left, right, on=BASE_COLUMNS, how="inner"
        ),
        tables,
    )

def split_data(
    frame: pd.DataFrame, encoder: LabelEncoder
) -> dict[str, dict[str, Any]]:
    columns = [name for name in frame.columns if name not in BASE_COLUMNS]
    data: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        part = frame[frame["split"] == split].reset_index(drop=True)
        if part.empty:
            raise ValueError(f"{split} features are empty")
        data[split] = {
            "x": part[columns].to_numpy(dtype=np.float32),
            "y": encoder.transform(part["label"].astype(str)),
            "trial_id": part["trial_id"].astype(str).tolist(),
        }
    return data

def fit_xgboost(
    train: dict[str, Any],
    val: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> Pipeline:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train["x"])
    x_val = scaler.transform(val["x"])
    model = XGBClassifier(
        n_estimators=int(config.get("n_estimators", 500)),
        max_depth=int(config.get("max_depth", 4)),
        learning_rate=float(config.get("learning_rate", 0.05)),
        subsample=float(config.get("subsample", 0.8)),
        colsample_bytree=float(config.get("colsample_bytree", 0.8)),
        objective="multi:softprob",
        eval_metric="mlogloss",
        early_stopping_rounds=int(
            config.get("early_stopping_rounds", 30)
        ),
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(
        x_train,
        train["y"],
        eval_set=[(x_val, val["y"])],
        verbose=False,
    )
    return Pipeline([("scaler", scaler), ("clf", model)])

def load_existing(path: Path, seeds: list[int]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    required = {"seed", "task_name", "modality_combo"}
    if not required.issubset(frame.columns):
        legacy = path.with_name(f"{path.stem}_legacy.csv")
        if not legacy.exists():
            write_csv(frame, legacy)
        return []
    return frame[frame["seed"].astype(int).isin(seeds)].to_dict("records")

def save_progress(
    rows: list[dict[str, Any]], output_path: Path, all_path: Path
) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["task_name", "seed", "modality_combo"]
        ).reset_index(drop=True)
    write_csv(frame, output_path)

    task_name = str(rows[0]["task_name"]) if rows else ""
    if all_path.exists():
        old = pd.read_csv(all_path)
        if {"task_name", "method_type"}.issubset(old.columns):
            old = old[
                ~(
                    (old["task_name"].astype(str) == task_name)
                    & (old["method_type"] == "traditional_ml")
                )
            ]
    else:
        old = pd.DataFrame()
    current = frame.copy()
    if not current.empty:
        current["method_type"] = "traditional_ml"
        current["model"] = "XGBoost"
    write_csv(pd.concat([old, current], ignore_index=True), all_path)

def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    task_name = str(config["task_name"])
    seeds = seed_list(config.get("seed", [0, 1, 2, 3, 4]))
    combos = [
        [str(value).lower() for value in combo]
        for combo in config.get("modality_combos", [])
    ]
    if not combos:
        raise ValueError("modality_combos cannot be empty")
    if any("visual" in combo for combo in combos):
        raise ValueError("Interaction does not use Visual modality")

    cache_dir = resolve(
        config.get("feature_cache_dir", "cache/interaction")
    ) / task_name
    output_path = resolve(config["output_csv"])
    all_path = resolve(
        config.get(
            "all_results_csv",
            "interaction/results/tables/all_interaction_results.csv",
        )
    )
    model_root = resolve(
        config.get(
            "model_dir", f"checkpoints/interaction/{task_name}/ml"
        )
    )
    model_root.mkdir(parents=True, exist_ok=True)
    rows = load_existing(output_path, seeds)
    logger = setup_logger(
        f"train_interaction_ml_{task_name}",
        PROJECT_ROOT
        / "interaction/results/logs"
        / task_name
        / f"train_ml_{task_name}.log",
    )

    for combo in combos:
        frame = prepare_combo(cache_dir, combo)
        if frame.empty:
            raise ValueError(f"task {task_name} combination {combo} has no aligned windows")
        encoder = LabelEncoder().fit(frame["label"].astype(str))
        data = split_data(frame, encoder)
        combo_name = "+".join(combo)
        for seed in seeds:
            key = (seed, combo_name)
            existing = {
                (int(row["seed"]), str(row["modality_combo"]))
                for row in rows
            }
            if key in existing:
                logger.info("skipping existing result：combo=%s seed=%s", combo_name, seed)
                continue
            set_seed(seed)
            model = fit_xgboost(
                data["train"], data["val"], config.get("xgboost", {}), seed
            )
            val_metrics = compute_window_and_trial_metrics(
                y_true=data["val"]["y"],
                scores=model.predict_proba(data["val"]["x"]),
                trial_ids=data["val"]["trial_id"],
                average=str(config.get("average", "macro")),
            )
            metrics = compute_window_and_trial_metrics(
                y_true=data["test"]["y"],
                scores=model.predict_proba(data["test"]["x"]),
                trial_ids=data["test"]["trial_id"],
                average=str(config.get("average", "macro")),
            )
            row = {
                "task_name": task_name,
                "seed": seed,
                "evaluation_mode": "fixed_trial_split",
                "modality_combo": combo_name,
                "classifier": "XGBoost",
                "fusion": "none" if len(combo) == 1 else "feature_concat",
                "val_f1_trial": val_metrics["f1_trial"],
                **{name: metrics[name] for name in METRICS},
            }
            rows.append(row)
            seed_dir = model_root / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, seed_dir / f"xgboost_{combo_name}.joblib")
            save_progress(rows, output_path, all_path)
            logger.info(
                "completed：task=%s combo=%s seed=%s trial_f1=%.4f",
                task_name,
                combo_name,
                seed,
                metrics["f1_trial"],
            )

    save_progress(rows, output_path, all_path)

if __name__ == "__main__":
    main()
