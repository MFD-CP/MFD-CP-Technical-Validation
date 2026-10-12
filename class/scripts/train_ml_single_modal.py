from __future__ import annotations
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
from src.features.feature_cache import feature_columns, load_feature_table
from src.training.metrics import compute_window_and_trial_metrics
from src.utils.config import load_yaml
from src.utils.io import write_csv
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()

def _split_data(df: pd.DataFrame, encoder: LabelEncoder) -> dict[str, Any]:
    feat_cols = feature_columns(df)
    df = df.dropna(subset=feat_cols)
    data = {}
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split].reset_index(drop=True)
        data[split] = {
            "x": part[feat_cols].to_numpy(dtype=np.float32),
            "y": encoder.transform(part["label"].astype(str)),
            "trial_id": part["trial_id"].astype(str).tolist(),
        }
    return data

def _scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)
    decision = model.decision_function(x)
    if decision.ndim == 1:
        decision = np.stack([-decision, decision], axis=1)
    return decision

def _fit_svm(train: dict[str, Any], val: dict[str, Any], cfg: dict[str, Any]) -> Any:
    x = np.concatenate([train["x"], val["x"]], axis=0)
    y = np.concatenate([train["y"], val["y"]], axis=0)
    fold = np.concatenate([-np.ones(len(train["y"]), dtype=int), np.zeros(len(val["y"]), dtype=int)])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight=cfg.get("class_weight", "balanced"))),
    ])
    grid = {"clf__C": cfg.get("C", [0.1, 1.0, 10.0]), "clf__gamma": cfg.get("gamma", ["scale", "auto"])}
    search = GridSearchCV(
        pipe,
        grid,
        scoring=make_scorer(f1_score, average="macro"),
        cv=PredefinedSplit(fold),
        refit=True,
        n_jobs=-1,
    )
    search.fit(x, y)
    return search.best_estimator_

def _fit_rf(train: dict[str, Any], val: dict[str, Any], cfg: dict[str, Any], seed: int) -> Any:
    x = np.concatenate([train["x"], val["x"]], axis=0)
    y = np.concatenate([train["y"], val["y"]], axis=0)
    fold = np.concatenate([-np.ones(len(train["y"]), dtype=int), np.zeros(len(val["y"]), dtype=int)])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=int(cfg.get("n_estimators", 500)),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    grid = {"clf__max_depth": cfg.get("max_depth", [10, 20, None])}
    search = GridSearchCV(
        pipe,
        grid,
        scoring=make_scorer(f1_score, average="macro"),
        cv=PredefinedSplit(fold),
        refit=True,
        n_jobs=-1,
    )
    search.fit(x, y)
    return search.best_estimator_

def _fit_xgb(train: dict[str, Any], val: dict[str, Any], cfg: dict[str, Any], seed: int) -> Any:
    if XGBClassifier is None:
        raise ImportError("install xgboost。")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train["x"])
    x_val = scaler.transform(val["x"])
    max_depths = cfg.get("max_depth", [4, 6])
    if not isinstance(max_depths, list):
        max_depths = [max_depths]
    best_model = None
    best_score = -1.0
    for max_depth in max_depths:
        model = XGBClassifier(
            n_estimators=int(cfg.get("n_estimators", 500)),
            max_depth=int(max_depth),
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
        val_metrics = compute_window_and_trial_metrics(
            y_true=val["y"],
            scores=model.predict_proba(x_val),
            trial_ids=val["trial_id"],
            average="macro",
        )
        score = val_metrics["f1_trial"]
        if score > best_score:
            best_score = score
            best_model = model
    return Pipeline([("scaler", scaler), ("clf", best_model)])

def _experiment_name(modality: str, classifier: str) -> str:
    name_map = {"audio": "Audio", "accel": "Accel", "gelsight": "GelSight", "visual": "Visual"}
    return f"ML-Single-{name_map.get(modality, modality)}-{classifier}"

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

def _save_progress(
    rows: list[dict[str, Any]],
    output_path: Path,
    all_path: Path,
    experiment_names: set[str],
) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["seed", "modality", "classifier"]).reset_index(drop=True)
    write_csv(frame, output_path)

    if all_path.exists():
        old = pd.read_csv(all_path)
        old = old[~old["experiment_name"].isin(experiment_names)]
    else:
        old = pd.DataFrame()
    current = pd.DataFrame(
        [
            {
                "seed": int(row["seed"]),
                "experiment_name": _experiment_name(str(row["modality"]), str(row["classifier"])),
                "method_type": "traditional_ml",
                "model": row["classifier"],
                "modality_combo": row["modality"],
                "fusion": "none",
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
    logger = setup_logger("train_ml_single", PROJECT_ROOT / "class/results/logs/train_ml_single.log")
    cache_dir = PROJECT_ROOT / cfg.get("feature_cache_dir", "cache/class")
    output_path = PROJECT_ROOT / cfg.get("output_csv", "class/results/tables/single_modal_ml.csv")
    all_path = PROJECT_ROOT / cfg.get("all_results_csv", "class/results/tables/all_results.csv")
    rows = _load_existing_rows(output_path, seeds)
    model_dir = PROJECT_ROOT / "checkpoints/class/ml_single"
    model_dir.mkdir(parents=True, exist_ok=True)
    classifiers = ["XGBoost"]
    experiment_names = {
        _experiment_name(modality, classifier)
        for modality in cfg.get("modalities", ["audio", "accel", "gelsight", "visual"])
        for classifier in classifiers
    }

    for modality in cfg.get("modalities", ["audio", "accel", "gelsight", "visual"]):
        df = load_feature_table(cache_dir, modality)
        encoder = LabelEncoder().fit(df["label"].astype(str))
        data = _split_data(df, encoder)
        for seed in seeds:
            result_key = (seed, modality, "XGBoost")
            saved_keys = {
                (int(row["seed"]), str(row["modality"]), str(row["classifier"]))
                for row in rows
            }
            if result_key in saved_keys:
                logger.info("skipping existing result: modality=%s classifier=XGBoost seed=%s", modality, seed)
                continue

            set_seed(seed)
            classifier = "XGBoost"
            model = _fit_xgb(data["train"], data["val"], cfg.get("xgboost", {}), seed)
            val_metrics = compute_window_and_trial_metrics(
                y_true=data["val"]["y"],
                scores=_scores(model, data["val"]["x"]),
                trial_ids=data["val"]["trial_id"],
                average=cfg.get("average", "macro"),
            )
            scores = _scores(model, data["test"]["x"])
            metrics = compute_window_and_trial_metrics(
                y_true=data["test"]["y"],
                scores=scores,
                trial_ids=data["test"]["trial_id"],
                average=cfg.get("average", "macro"),
            )
            row = {
                "seed": seed,
                "modality": modality,
                "classifier": classifier,
                "val_f1_trial": val_metrics["f1_trial"],
                **{k: metrics[k] for k in [
                    "accuracy_window", "precision_window", "recall_window", "f1_window",
                    "accuracy_trial", "precision_trial", "recall_trial", "f1_trial",
                ]},
            }
            rows.append(row)
            seed_model_dir = model_dir / f"seed_{seed}"
            seed_model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, seed_model_dir / f"{modality}_{classifier}.joblib")
            _save_progress(rows, output_path, all_path, experiment_names)
            logger.info(
                "completed: modality=%s classifier=%s seed=%s trial_f1=%.4f",
                modality,
                classifier,
                seed,
                metrics["f1_trial"],
            )

    _save_progress(rows, output_path, all_path, experiment_names)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
