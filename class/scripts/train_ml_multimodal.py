from __future__ import annotations
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from functools import reduce
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("install xgboost。") from exc
from src.features.feature_cache import BASE_COLUMNS, feature_columns, load_feature_table
from src.training.metrics import compute_window_and_trial_metrics
from src.utils.config import load_yaml
from src.utils.io import write_csv
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()

def _prepare_combo(cache_dir: Path, combo: list[str]) -> pd.DataFrame:
    tables = []
    for modality in combo:
        df = load_feature_table(cache_dir, modality)
        feat_cols = feature_columns(df)
        table = df[BASE_COLUMNS + feat_cols].copy()
        table = table.rename(columns={c: f"{modality}_{c}" for c in feat_cols})
        tables.append(table)
    return reduce(lambda l, r: pd.merge(l, r, on=BASE_COLUMNS, how="inner"), tables)

def _split_data(df: pd.DataFrame, encoder: LabelEncoder) -> dict[str, Any]:
    feat_cols = [c for c in df.columns if c not in BASE_COLUMNS]
    data = {}
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split].reset_index(drop=True)
        data[split] = {
            "x": part[feat_cols].to_numpy(dtype=np.float32),
            "y": encoder.transform(part["label"].astype(str)),
            "trial_id": part["trial_id"].astype(str).tolist(),
        }
    return data

def _fit_xgb(train: dict[str, Any], val: dict[str, Any], cfg: dict[str, Any], seed: int) -> Any:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train["x"])
    x_val = scaler.transform(val["x"])
    model = XGBClassifier(
        n_estimators=int(cfg.get("n_estimators", 500)),
        max_depth=int(cfg.get("max_depth", 4)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        subsample=float(cfg.get("subsample", 0.8)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.8)),
        objective="multi:softprob",
        eval_metric="mlogloss",
        early_stopping_rounds=int(cfg.get("early_stopping_rounds", 30)),
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

def _seed_list(value: Any) -> list[int]:
    return [int(seed) for seed in value] if isinstance(value, list) else [int(value)]

def _load_existing_rows(output_path: Path, seeds: list[int]) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    frame = pd.read_csv(output_path)
    if "seed" not in frame.columns:
        legacy_path = output_path.with_name(f"{output_path.stem}_seed42_legacy.csv")
        if not legacy_path.exists():
            legacy = frame.copy()
            legacy.insert(0, "seed", 42)
            write_csv(legacy, legacy_path)
        return []
    frame = frame[frame["seed"].astype(int).isin(seeds)]
    return frame.to_dict("records")

def _save_progress(rows: list[dict[str, Any]], output_path: Path, all_path: Path) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["seed", "modality_combo"]).reset_index(drop=True)
    write_csv(frame, output_path)

    experiment_name = "ML-Multimodal-XGBoost-FeatureConcat"
    if all_path.exists():
        old = pd.read_csv(all_path)
        old = old[old["experiment_name"] != experiment_name]
    else:
        old = pd.DataFrame()
    current = pd.DataFrame(
        [
            {
                "seed": int(row["seed"]),
                "experiment_name": experiment_name,
                "method_type": "traditional_ml",
                "model": "XGBoost",
                "modality_combo": row["modality_combo"],
                "fusion": row["fusion"],
                **{key: row[key] for key in [
                    "accuracy_window", "precision_window", "recall_window", "f1_window",
                    "accuracy_trial", "precision_trial", "recall_trial", "f1_trial",
                ]},
            }
            for row in rows
        ]
    )
    write_csv(pd.concat([old, current], ignore_index=True), all_path)

def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seeds = _seed_list(cfg.get("seed", 42))
    logger = setup_logger("train_ml_multimodal", PROJECT_ROOT / "class/results/logs/train_ml_multimodal.log")
    cache_dir = PROJECT_ROOT / cfg.get("feature_cache_dir", "cache/class")
    output_path = PROJECT_ROOT / cfg.get("output_csv", "class/results/tables/multimodal_ml_xgboost.csv")
    all_path = PROJECT_ROOT / cfg.get("all_results_csv", "class/results/tables/all_results.csv")
    rows = _load_existing_rows(output_path, seeds)
    model_dir = PROJECT_ROOT / "checkpoints/class/ml_multimodal"
    model_dir.mkdir(parents=True, exist_ok=True)

    for combo in cfg.get("modality_combos", []):
        df = _prepare_combo(cache_dir, combo)
        if df.empty:
            logger.warning("%s has no aligned samples，skip。", combo)
            continue
        encoder = LabelEncoder().fit(df["label"].astype(str))
        data = _split_data(df, encoder)
        combo_name = "+".join(combo)
        for seed in seeds:
            result_key = (seed, combo_name)
            saved_keys = {
                (int(row["seed"]), str(row["modality_combo"]))
                for row in rows
            }
            if result_key in saved_keys:
                logger.info("skipping existing result: combo=%s seed=%s", combo_name, seed)
                continue

            set_seed(seed)
            model = _fit_xgb(data["train"], data["val"], cfg.get("xgboost", {}), seed)
            val_metrics = compute_window_and_trial_metrics(
                y_true=data["val"]["y"],
                scores=model.predict_proba(data["val"]["x"]),
                trial_ids=data["val"]["trial_id"],
                average=cfg.get("average", "macro"),
            )
            scores = model.predict_proba(data["test"]["x"])
            metrics = compute_window_and_trial_metrics(
                y_true=data["test"]["y"],
                scores=scores,
                trial_ids=data["test"]["trial_id"],
                average=cfg.get("average", "macro"),
            )
            row = {
                "seed": seed,
                "modality_combo": combo_name,
                "fusion": "feature_concat",
                "classifier": "XGBoost",
                "val_f1_trial": val_metrics["f1_trial"],
                **{key: metrics[key] for key in [
                    "accuracy_window", "precision_window", "recall_window", "f1_window",
                    "accuracy_trial", "precision_trial", "recall_trial", "f1_trial",
                ]},
            }
            rows.append(row)
            seed_model_dir = model_dir / f"seed_{seed}"
            seed_model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, seed_model_dir / f"xgboost_{combo_name}.joblib")
            _save_progress(rows, output_path, all_path)
            logger.info(
                "completed: combo=%s seed=%s trial_f1=%.4f",
                combo_name,
                seed,
                metrics["f1_trial"],
            )

    _save_progress(rows, output_path, all_path)

if __name__ == "__main__":
    main()
