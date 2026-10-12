from __future__ import annotations
import argparse
import copy
import gc
import sys
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import joblib
import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import train_ml_multimodal as ml_multimodal
import train_ml_single_modal as ml_single
from src.networks.multimodal_late_fusion_net import MultiModalLateFusionNet
from src.networks.single_modal_net import SingleModalNet
from src.training.evaluation import save_confusion_matrix
from src.training.lightning_module import FabricClassificationLightningModule
from src.training.lightning_utils import (
    apply_encoder_override,
    compute_num_devices,
    sanitize_name,
    torch_precision_setting,
)
from src.training.metrics import compute_metrics
from src.training.train_utils import (
    compute_class_weights,
    load_classification_data,
    make_dataloaders,
)
from src.utils.config import load_yaml
from src.utils.io import load_json, write_csv
from src.utils.logger import setup_logger

EXPECTED_SEEDS = [0, 1, 2, 3, 4]
TEST_METRICS = [
    "accuracy_window",
    "precision_window",
    "recall_window",
    "f1_window",
    "accuracy_trial",
    "precision_trial",
    "recall_trial",
    "f1_trial",
]
RESULT_TABLES = {
    "ml_single": PROJECT_ROOT / "class/results/tables/single_modal_ml.csv",
    "ml_multimodal": PROJECT_ROOT / "class/results/tables/multimodal_ml_xgboost.csv",
    "deep_single": PROJECT_ROOT / "class/results/tables/single_modal_deep.csv",
    "deep_multimodal": PROJECT_ROOT / "class/results/tables/multimodal_deep_late_concat.csv",
}
SINGLE_ML_CONFIG = PROJECT_ROOT / "class/configs/ml/single_modal.yaml"
MULTIMODAL_ML_CONFIG = PROJECT_ROOT / "class/configs/ml/multimodal_xgboost.yaml"
SINGLE_DEEP_CONFIG = PROJECT_ROOT / "class/configs/deep/single_modal_all_encoders.yaml"
MULTIMODAL_DEEP_CONFIG = PROJECT_ROOT / "class/configs/deep/multimodal_late_concat.yaml"
RESULTS_ROOT = PROJECT_ROOT / "class/results"
LABEL_MAPPING = PROJECT_ROOT / "checkpoints/class/label_mapping.json"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--rerun-containing",
        default=None,
        help=(
            "only for modalities rerun inference for experiments containing this modality in；"
            "reuse existing predictions for other experiments、reports and best seed record。"
        ),
    )
    parser.add_argument(
        "--rerun-group",
        choices=(
            "ml_single",
            "ml_multimodal",
            "deep_single",
            "deep_multimodal",
        ),
        default=None,
        help=(
            "rerun inference only for the specified experiment groups；reuse existing predictions for other experiment groups、"
            "reports and best seed record。"
        ),
    )
    parser.add_argument(
        "--reuse-metrics-if-artifacts-missing",
        action="store_true",
        help=(
            "for selective reporting，if predictions for other experiments/reports are missing，then only reuse"
            "existing five seed metrics and checkpoint selection information，rerunning inference is prohibited。"
        ),
    )
    return parser.parse_args()

def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def _load_merged_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    deep_cfg = load_yaml(config_path)
    dataset_cfg = load_yaml(_resolve_project_path(deep_cfg["dataset_config"]))
    merged = copy.deepcopy(dataset_cfg)
    merged.update(deep_cfg)
    for key in ["audio", "accel", "gelsight", "visual", "window"]:
        if key in dataset_cfg:
            merged[key] = dataset_cfg[key]
    return deep_cfg, merged

def _find_experiment(deep_cfg: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    for experiment in deep_cfg.get("experiments", []):
        if str(experiment.get("name")) == experiment_name:
            return experiment
    raise KeyError(f"experiment does not exist in the configuration: {experiment_name}")

def _find_latest_best_checkpoint(
    deep_cfg: dict[str, Any],
    experiment_name: str,
    seed: int,
) -> Path:
    checkpoint_root = _resolve_project_path(deep_cfg.get("checkpoint_dir", "checkpoints/class"))
    run_dir = checkpoint_root / experiment_name / f"seed_{seed}"
    checkpoints = sorted(run_dir.glob("best-*.ckpt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"no best checkpoint found: {run_dir}")
    return checkpoints[-1]

def _load_label_mapping() -> tuple[dict[str, int], list[str]]:
    mapping = load_json(LABEL_MAPPING)
    label_to_index = {str(key): int(value) for key, value in mapping["label_to_index"].items()}
    class_names = [str(value) for value in mapping.get("class_names", [])]
    if not class_names:
        inverse = {index: label for label, index in label_to_index.items()}
        class_names = [inverse[index] for index in range(len(inverse))]
    return label_to_index, class_names

def _build_inference_objects(
    selected_run: pd.Series,
    num_workers: int,
    batch_size_override: int | None,
) -> tuple[
    dict[str, Any],
    FabricClassificationLightningModule,
    Any,
    Path,
    list[str],
]:
    deep_cfg, cfg = _load_merged_config(Path(str(selected_run["config_path"])))
    experiment_name = str(selected_run["experiment_name"])
    seed = int(selected_run["seed"])
    experiment = _find_experiment(deep_cfg, experiment_name)
    label_to_index, class_names = _load_label_mapping()
    dataset_root, metadata, split_df = load_classification_data(cfg, PROJECT_ROOT)

    if str(selected_run["model_family"]) == "single":
        modality = str(experiment["modality"]).lower()
        encoder = str(experiment["encoder"]).lower()
        modalities = [modality]
        run_cfg = apply_encoder_override(cfg, {modality: encoder})
        model = SingleModalNet(modality, len(class_names), run_cfg)
    else:
        modalities = [str(value).lower() for value in experiment["modality_combo"]]
        encoders = {
            str(modality).lower(): encoder
            for modality, encoder in experiment["encoders"].items()
            if str(modality).lower() in modalities
        }
        run_cfg = apply_encoder_override(cfg, encoders)
        model = MultiModalLateFusionNet(modalities, len(class_names), run_cfg)

    batch_size = batch_size_override or int(deep_cfg.get("batch_size", 32))
    loaders = make_dataloaders(
        metadata=metadata,
        split_df=split_df,
        dataset_root=dataset_root,
        modalities=modalities,
        config=run_cfg,
        label_to_index=label_to_index,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    class_weights = None
    if bool(deep_cfg.get("class_weighted_loss", True)):
        train_meta = metadata.merge(split_df[["trial_id", "split"]], on="trial_id")
        train_meta = train_meta[train_meta["split"] == "train"]
        class_weights = compute_class_weights(train_meta, label_to_index)

    lit_module = FabricClassificationLightningModule(
        model=model,
        num_classes=len(class_names),
        learning_rate=float(deep_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(deep_cfg.get("weight_decay", 1e-4)),
        scheduler_name=str(deep_cfg.get("scheduler", "reduce_on_plateau")),
        scheduler_patience=int(deep_cfg.get("scheduler_patience", 5)),
        class_weights=class_weights,
        average=str(deep_cfg.get("average", "macro")),
        trial_aggregation=str(deep_cfg.get("trial_aggregation", "average_logits")),
    )
    checkpoint = _find_latest_best_checkpoint(deep_cfg, experiment_name, seed)
    return deep_cfg, lit_module, loaders["test"], checkpoint, class_names

def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def _prediction_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    trial_ids: np.ndarray,
    class_names: list[str],
    num_windows: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "trial_id": trial_ids.astype(str),
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "true_label": [class_names[index] for index in y_true],
            "pred_label": [class_names[index] for index in y_pred],
            "confidence": probabilities.max(axis=1),
        }
    )
    if num_windows is not None:
        frame.insert(1, "num_windows", num_windows.astype(int))
    for index, class_name in enumerate(class_names):
        frame[f"prob_class_{index}_{class_name}"] = probabilities[:, index]
    return frame

def _aggregate_trials(
    y_true: np.ndarray,
    scores: np.ndarray,
    trial_ids: list[str],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trial_array = np.asarray(trial_ids, dtype=str)
    unique_trials = np.unique(trial_array)
    true_rows: list[int] = []
    pred_rows: list[int] = []
    score_rows: list[np.ndarray] = []
    window_counts: list[int] = []
    for trial_id in unique_trials:
        mask = trial_array == trial_id
        labels = y_true[mask].astype(int)
        mean_scores = np.mean(scores[mask], axis=0)
        if method == "majority_vote":
            window_predictions = np.argmax(scores[mask], axis=1)
            prediction = int(np.bincount(window_predictions).argmax())
        else:
            prediction = int(np.argmax(mean_scores))
        true_rows.append(int(np.bincount(labels).argmax()))
        pred_rows.append(prediction)
        score_rows.append(mean_scores)
        window_counts.append(int(mask.sum()))
    return (
        np.asarray(true_rows, dtype=int),
        np.asarray(pred_rows, dtype=int),
        np.stack(score_rows),
        unique_trials,
        np.asarray(window_counts, dtype=int),
    )

def _combo_title(combo: str) -> str:
    names = {"audio": "Audio", "accel": "Accel", "gelsight": "GelSight", "visual": "Visual"}
    return "".join(names.get(part, part.title()) for part in combo.split("+"))

def _validate_five_seeds(frame: pd.DataFrame, key_columns: list[str], name: str) -> pd.DataFrame:
    if "seed" not in frame.columns:
        raise ValueError(f"{name} is missing seed field，old seed=42 table cannot be used for the official report")
    formal = frame[frame["seed"].astype(int).isin(EXPECTED_SEEDS)].copy()
    if formal.duplicated([*key_columns, "seed"]).any():
        raise ValueError(f"{name} contains duplicate experiment configurations + seed")
    counts = formal.groupby(key_columns)["seed"].nunique()
    if formal.empty:
        raise ValueError(f"{name} does not have seed 0-4 official results")
    incomplete = counts[counts != len(EXPECTED_SEEDS)]
    if not incomplete.empty:
        details = {str(key): int(value) for key, value in incomplete.items()}
        raise ValueError(f"{name} has not completed all five seed: {details}")
    return formal

def _load_normalized_results() -> pd.DataFrame:
    for path in RESULT_TABLES.values():
        if not path.exists():
            raise FileNotFoundError(path)

    single_ml = _validate_five_seeds(
        pd.read_csv(RESULT_TABLES["ml_single"]), ["modality", "classifier"], "single-modal ML"
    )
    if "val_f1_trial" not in single_ml.columns or single_ml["val_f1_trial"].isna().any():
        raise ValueError("single-modal ML is missing val_f1_trial，must use the new five seed script must be rerun")
    single_ml["experiment_name"] = single_ml.apply(
        lambda row: ml_single._experiment_name(str(row["modality"]), str(row["classifier"])),
        axis=1,
    )
    single_ml["experiment_group"] = "ml_single"
    single_ml["modalities"] = single_ml["modality"].astype(str)
    single_ml["model_description"] = single_ml["classifier"].astype(str)
    single_ml["config_path"] = ""

    multimodal_ml = _validate_five_seeds(
        pd.read_csv(RESULT_TABLES["ml_multimodal"]),
        ["modality_combo", "classifier", "fusion"],
        "multimodal ML",
    )
    if "val_f1_trial" not in multimodal_ml.columns or multimodal_ml["val_f1_trial"].isna().any():
        raise ValueError("multimodal ML is missing val_f1_trial，must use the new five seed script must be rerun")
    multimodal_ml["experiment_name"] = multimodal_ml["modality_combo"].map(
        lambda combo: f"ML-Multimodal-XGBoost-FeatureConcat-{_combo_title(str(combo))}"
    )
    multimodal_ml["experiment_group"] = "ml_multimodal"
    multimodal_ml["modalities"] = multimodal_ml["modality_combo"].astype(str)
    multimodal_ml["model_description"] = multimodal_ml["classifier"].astype(str)
    multimodal_ml["config_path"] = ""

    single_deep = _validate_five_seeds(
        pd.read_csv(RESULT_TABLES["deep_single"]), ["experiment_name"], "single-modal DL"
    )
    single_deep["experiment_group"] = "deep_single"
    single_deep["modalities"] = single_deep["modality"].astype(str)
    single_deep["model_description"] = single_deep["encoder"].astype(str)
    single_deep["config_path"] = str(SINGLE_DEEP_CONFIG)
    single_deep["val_f1_trial"] = np.nan

    multimodal_deep = _validate_five_seeds(
        pd.read_csv(RESULT_TABLES["deep_multimodal"]), ["experiment_name"], "multimodal DL"
    )
    multimodal_deep["experiment_group"] = "deep_multimodal"
    multimodal_deep["modalities"] = multimodal_deep["modality_combo"].astype(str)
    multimodal_deep["model_description"] = multimodal_deep["encoders"].astype(str)
    multimodal_deep["config_path"] = str(MULTIMODAL_DEEP_CONFIG)
    multimodal_deep["val_f1_trial"] = np.nan

    columns = [
        "seed",
        "experiment_name",
        "experiment_group",
        "modalities",
        "model_description",
        "config_path",
        "val_f1_trial",
        *TEST_METRICS,
    ]
    results = pd.concat(
        [single_ml[columns], multimodal_ml[columns], single_deep[columns], multimodal_deep[columns]],
        ignore_index=True,
    )
    expected_by_group = {
        "ml_single": {
            ml_single._experiment_name(str(modality), "XGBoost")
            for modality in load_yaml(SINGLE_ML_CONFIG).get("modalities", [])
        },
        "ml_multimodal": {
            f"ML-Multimodal-XGBoost-FeatureConcat-{_combo_title('+'.join(combo))}"
            for combo in load_yaml(MULTIMODAL_ML_CONFIG).get("modality_combos", [])
        },
        "deep_single": {
            str(experiment["name"])
            for experiment in load_yaml(SINGLE_DEEP_CONFIG).get("experiments", [])
        },
        "deep_multimodal": {
            str(experiment["name"])
            for experiment in load_yaml(MULTIMODAL_DEEP_CONFIG).get("experiments", [])
        },
    }
    for group_name, expected_names in expected_by_group.items():
        actual_names = set(
            results.loc[results["experiment_group"] == group_name, "experiment_name"].astype(str)
        )
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        if missing or unexpected:
            raise ValueError(
                f"{group_name} experiment set is inconsistent with the configuration: missing={missing}, unexpected={unexpected}"
            )
    return results

def _aggregate_test_statistics(results: pd.DataFrame) -> pd.DataFrame:
    metadata = ["experiment_name", "experiment_group", "modalities", "model_description"]
    aggregations: dict[str, tuple[str, str]] = {"runs": ("seed", "nunique")}
    for metric in TEST_METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    return results.groupby(metadata, as_index=False).agg(**aggregations)

def _find_best_checkpoint(config_path: Path, experiment_name: str, seed: int) -> Path:
    deep_cfg, _ = _load_merged_config(config_path)
    checkpoint_root = _resolve_project_path(
        deep_cfg.get("checkpoint_dir", "checkpoints/class")
    )
    run_dir = checkpoint_root / experiment_name / f"seed_{seed}"
    checkpoints = list(run_dir.glob("best-*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"no best checkpoint found: {run_dir}")
    return max(checkpoints, key=_checkpoint_val_f1)

def _checkpoint_val_f1(checkpoint: Path) -> float:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for state in payload.get("callbacks", {}).values():
        if not isinstance(state, dict) or state.get("monitor") != "val_f1_trial":
            continue
        score = state.get("best_model_score")
        if score is not None:
            return float(score.item() if hasattr(score, "item") else score)
    raise ValueError(f"checkpoint does not contain val_f1_trial: {checkpoint}")

def _select_best_seed(group: pd.DataFrame) -> tuple[pd.Series, float, Path]:
    experiment_group = str(group.iloc[0]["experiment_group"])
    if experiment_group.startswith("ml_"):
        selected = group.sort_values(
            ["val_f1_trial", "seed"], ascending=[False, True], kind="mergesort"
        ).iloc[0]
        if experiment_group == "ml_single":
            model_path = (
                PROJECT_ROOT
                / "checkpoints/class/ml_single"
                / f"seed_{int(selected['seed'])}"
                / f"{selected['modalities']}_XGBoost.joblib"
            )
        else:
            model_path = (
                PROJECT_ROOT
                / "checkpoints/class/ml_multimodal"
                / f"seed_{int(selected['seed'])}"
                / f"xgboost_{selected['modalities']}.joblib"
            )
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        return selected, float(selected["val_f1_trial"]), model_path

    scored: list[tuple[float, int, Path]] = []
    config_path = Path(str(group.iloc[0]["config_path"]))
    experiment_name = str(group.iloc[0]["experiment_name"])
    for seed in EXPECTED_SEEDS:
        checkpoint = _find_best_checkpoint(config_path, experiment_name, seed)
        scored.append((_checkpoint_val_f1(checkpoint), seed, checkpoint))
    best_score, best_seed, checkpoint = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
    selected = group[group["seed"].astype(int) == best_seed].iloc[0]
    return selected, best_score, checkpoint

def _run_ml_test(selected: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    group = str(selected["experiment_group"])
    cache_dir = PROJECT_ROOT / "cache/class"
    if group == "ml_single":
        modality = str(selected["modalities"])
        table = ml_single.load_feature_table(cache_dir, modality)
        encoder = LabelEncoder().fit(table["label"].astype(str))
        data = ml_single._split_data(table, encoder)
        model_path = (
            PROJECT_ROOT
            / "checkpoints/class/ml_single"
            / f"seed_{int(selected['seed'])}"
            / f"{modality}_XGBoost.joblib"
        )
        model = joblib.load(model_path)
        scores = ml_single._scores(model, data["test"]["x"])
    else:
        combo = str(selected["modalities"])
        table = ml_multimodal._prepare_combo(cache_dir, combo.split("+"))
        encoder = LabelEncoder().fit(table["label"].astype(str))
        data = ml_multimodal._split_data(table, encoder)
        model_path = (
            PROJECT_ROOT
            / "checkpoints/class/ml_multimodal"
            / f"seed_{int(selected['seed'])}"
            / f"xgboost_{combo}.joblib"
        )
        model = joblib.load(model_path)
        scores = model.predict_proba(data["test"]["x"])
    return (
        data["test"]["y"],
        scores,
        data["test"]["trial_id"],
        [str(value) for value in encoder.classes_],
    )

def _run_deep_test(
    selected: pd.Series,
    num_workers: int,
    batch_size: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], Path]:
    family = "single" if selected["experiment_group"] == "deep_single" else "multimodal"
    model_row = selected.copy()
    model_row["model_family"] = family
    deep_cfg, lit_module, test_loader, checkpoint, class_names = _build_inference_objects(
        model_row,
        num_workers=num_workers,
        batch_size_override=batch_size,
    )
    trainer = Trainer(
        accelerator=str(deep_cfg.get("accelerator", "auto")),
        devices=compute_num_devices(deep_cfg),
        precision=torch_precision_setting(deep_cfg),
        deterministic=bool(deep_cfg.get("deterministic", True)),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(PROJECT_ROOT),
    )
    trainer.test(lit_module, dataloaders=test_loader, ckpt_path=str(checkpoint), verbose=False)
    if lit_module.last_test_y_true is None or lit_module.last_test_scores is None:
        raise RuntimeError("deep-model testing did not return per-window predictions")
    return (
        lit_module.last_test_y_true,
        lit_module.last_test_scores,
        lit_module.last_test_trial_ids,
        class_names,
        checkpoint,
    )

def _save_level_outputs(
    base_name: str,
    level: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_dir = RESULTS_ROOT / "reports" / base_name
    matrix_dir = RESULTS_ROOT / "confusion_matrices" / base_name
    report_dir.mkdir(parents=True, exist_ok=True)
    per_class = pd.DataFrame(
        [
            {
                "class_index": index,
                "class_name": class_name,
                "support": int(report[class_name]["support"]),
                "precision": float(report[class_name]["precision"]),
                "recall": float(report[class_name]["recall"]),
                "f1_score": float(report[class_name]["f1-score"]),
            }
            for index, class_name in enumerate(class_names)
        ]
    )
    write_csv(per_class, report_dir / f"{level}_per_class_metrics.csv")
    text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    (report_dir / f"{level}_classification_report.txt").write_text(text, encoding="utf-8")
    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        matrix_dir / f"{level}_confusion_matrix.csv",
        matrix_dir / f"{level}_confusion_matrix.png",
    )
    return {"level": level, "sample_count": len(y_true), **compute_metrics(y_true, y_pred)}

def _save_experiment_outputs(
    selected: pd.Series,
    val_f1: float,
    model_path: Path,
    statistics: pd.Series,
    y_true: np.ndarray,
    scores: np.ndarray,
    trial_ids: list[str],
    class_names: list[str],
    trial_aggregation: str,
) -> None:
    experiment_name = str(selected["experiment_name"])
    base_name = sanitize_name(experiment_name)
    scores_are_probabilities = str(selected["experiment_group"]).startswith("ml_")
    probability = scores if scores_are_probabilities else _softmax(scores)
    window_pred = np.argmax(scores, axis=1).astype(int)
    prediction_dir = RESULTS_ROOT / "predictions" / base_name
    write_csv(
        _prediction_frame(
            y_true,
            window_pred,
            probability,
            np.asarray(trial_ids),
            class_names,
        ),
        prediction_dir / "window_predictions.csv",
    )
    trial_true, trial_pred, trial_scores, unique_trials, window_counts = _aggregate_trials(
        y_true,
        scores,
        trial_ids,
        trial_aggregation,
    )
    write_csv(
        _prediction_frame(
            trial_true,
            trial_pred,
            trial_scores if scores_are_probabilities else _softmax(trial_scores),
            unique_trials,
            class_names,
            num_windows=window_counts,
        ),
        prediction_dir / "trial_predictions.csv",
    )

    report_dir = RESULTS_ROOT / "reports" / base_name
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame([statistics.to_dict()]), report_dir / "five_seed_test_mean_std.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "experiment_name": experiment_name,
                    "selected_seed": int(selected["seed"]),
                    "selection_metric": "val_f1_trial",
                    "best_val_f1_trial": val_f1,
                    "model_path": str(model_path),
                }
            ]
        ),
        report_dir / "best_seed_selection.csv",
    )
    selected_metrics = [
        _save_level_outputs(base_name, "window", y_true, window_pred, class_names),
        _save_level_outputs(base_name, "trial", trial_true, trial_pred, class_names),
    ]
    write_csv(pd.DataFrame(selected_metrics), report_dir / "selected_seed_test_metrics.csv")

def _write_master_report(statistics: pd.DataFrame, selections: pd.DataFrame) -> None:
    selected_map = selections.set_index("experiment_name")

    def selection_text(selection: pd.Series) -> tuple[str, str]:
        seed = selection["selected_seed"]
        val_f1 = selection["best_val_f1_trial"]
        seed_text = "N/A" if pd.isna(seed) else str(int(seed))
        val_text = "N/A" if pd.isna(val_f1) else f"{float(val_f1):.6f}"
        return seed_text, val_text

    lines = [
        "# Part A Five-seed Test Set Summary Report",
        "",
        "official statistics use only seed 0、1、2、3、4。the table uses sample standard deviation；",
        "the best for each experiment seed based only on validation-set trial-level Macro F1 select。",
        "",
        "## Window-level test results",
        "",
        "| Experiment | Runs | Accuracy | Macro Precision | Macro Recall | Macro F1 | Best Seed | Best Val Trial F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in statistics.sort_values("experiment_name").iterrows():
        selection = selected_map.loc[row["experiment_name"]]
        seed_text, val_text = selection_text(selection)
        lines.append(
            f"| {row['experiment_name']} | {int(row['runs'])} | "
            f"{row['accuracy_window_mean']:.6f} ± {row['accuracy_window_std']:.6f} | "
            f"{row['precision_window_mean']:.6f} ± {row['precision_window_std']:.6f} | "
            f"{row['recall_window_mean']:.6f} ± {row['recall_window_std']:.6f} | "
            f"{row['f1_window_mean']:.6f} ± {row['f1_window_std']:.6f} | "
            f"{seed_text} | {val_text} |"
        )
    lines.extend(
        [
            "",
            "## Trial-level test results",
            "",
            "| Experiment | Runs | Accuracy | Macro Precision | Macro Recall | Macro F1 | Best Seed | Best Val Trial F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in statistics.sort_values("experiment_name").iterrows():
        selection = selected_map.loc[row["experiment_name"]]
        seed_text, val_text = selection_text(selection)
        lines.append(
            f"| {row['experiment_name']} | {int(row['runs'])} | "
            f"{row['accuracy_trial_mean']:.6f} ± {row['accuracy_trial_std']:.6f} | "
            f"{row['precision_trial_mean']:.6f} ± {row['precision_trial_std']:.6f} | "
            f"{row['recall_trial_mean']:.6f} ± {row['recall_trial_std']:.6f} | "
            f"{row['f1_trial_mean']:.6f} ± {row['f1_trial_std']:.6f} | "
            f"{seed_text} | {val_text} |"
        )
    path = RESULTS_ROOT / "reports/five_seed_test_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _reuse_existing_selection(
    experiment_name: str,
    existing_selections: pd.DataFrame,
) -> dict[str, Any]:
    if "experiment_name" not in existing_selections.columns:
        raise ValueError("existing best seed table is empty or missing experiment_name column")
    matches = existing_selections[
        existing_selections["experiment_name"].astype(str) == experiment_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"cannot uniquely reuse the best seed record: {experiment_name}"
        )
    base_name = sanitize_name(experiment_name)
    required = [
        RESULTS_ROOT / "predictions" / base_name / "window_predictions.csv",
        RESULTS_ROOT / "predictions" / base_name / "trial_predictions.csv",
        RESULTS_ROOT / "reports" / base_name / "best_seed_selection.csv",
        RESULTS_ROOT / "reports" / base_name / "five_seed_test_mean_std.csv",
        RESULTS_ROOT / "reports" / base_name / "selected_seed_test_metrics.csv",
        RESULTS_ROOT / "reports" / base_name / "window_classification_report.txt",
        RESULTS_ROOT / "reports" / base_name / "trial_classification_report.txt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"experiment {experiment_name} is missing reusable report files: {missing}"
        )
    row = matches.iloc[0]
    return {
        "experiment_name": experiment_name,
        "experiment_group": row["experiment_group"],
        "modalities": row["modalities"],
        "selected_seed": int(row["selected_seed"]),
        "selection_metric": row["selection_metric"],
        "best_val_f1_trial": float(row["best_val_f1_trial"]),
        "model_path": str(row["model_path"]),
    }

def main() -> None:
    args = parse_args()
    logger = setup_logger(
        "create_experiment_reports",
        RESULTS_ROOT / "logs/create_experiment_reports.log",
    )
    results = _load_normalized_results()
    statistics = _aggregate_test_statistics(results)
    write_csv(statistics, RESULTS_ROOT / "tables/all_experiments_test_mean_std.csv")

    selection_path = RESULTS_ROOT / "tables/best_seed_by_experiment.csv"
    existing_selections = (
        pd.read_csv(selection_path)
        if selection_path.is_file()
        else pd.DataFrame()
    )
    rerun_containing = (
        str(args.rerun_containing).strip().lower()
        if args.rerun_containing
        else None
    )
    rerun_group = str(args.rerun_group) if args.rerun_group else None
    if rerun_containing and rerun_group:
        raise ValueError(
            "--rerun-containing and --rerun-group cannot be used simultaneously"
        )
    if (
        (rerun_containing or rerun_group)
        and existing_selections.empty
        and not args.reuse_metrics_if_artifacts_missing
    ):
        raise FileNotFoundError(
            "selective report generation requires an existing best_seed_by_experiment.csv"
        )

    selections: list[dict[str, Any]] = []
    for experiment_name, group in results.groupby("experiment_name", sort=True):
        modalities = {
            part.strip().lower()
            for part in str(group.iloc[0]["modalities"]).split("+")
        }
        experiment_group = str(group.iloc[0]["experiment_group"])
        reuse_existing = (
            rerun_containing is not None
            and rerun_containing not in modalities
        ) or (
            rerun_group is not None
            and rerun_group != experiment_group
        )
        if reuse_existing:
            try:
                reused = _reuse_existing_selection(
                    str(experiment_name), existing_selections
                )
                logger.info(
                    "reuse existing predictions and reports，do not rerun inference: experiment=%s",
                    experiment_name,
                )
            except (FileNotFoundError, ValueError):
                if not args.reuse_metrics_if_artifacts_missing:
                    raise
                try:
                    selected, val_f1, model_path = _select_best_seed(group)
                    reused = {
                        "experiment_name": experiment_name,
                        "experiment_group": selected["experiment_group"],
                        "modalities": selected["modalities"],
                        "selected_seed": int(selected["seed"]),
                        "selection_metric": "val_f1_trial",
                        "best_val_f1_trial": val_f1,
                        "model_path": str(model_path),
                    }
                    logger.info(
                        "reuse the existing five seed metrics and model selection，do not rerun inference: experiment=%s",
                        experiment_name,
                    )
                except FileNotFoundError:
                    metadata = group.iloc[0]
                    reused = {
                        "experiment_name": experiment_name,
                        "experiment_group": metadata["experiment_group"],
                        "modalities": metadata["modalities"],
                        "selected_seed": None,
                        "selection_metric": "unavailable_checkpoint_missing",
                        "best_val_f1_trial": np.nan,
                        "model_path": "",
                    }
                    logger.warning(
                        "reuse only the existing five seed test metrics；checkpoint missing，cannot recover the validation-set best seed: "
                        "experiment=%s",
                        experiment_name,
                    )
            selections.append(reused)
            continue

        selected, val_f1, model_path = _select_best_seed(group)
        logger.info(
            "best seed: experiment=%s seed=%s val_f1_trial=%.6f",
            experiment_name,
            int(selected["seed"]),
            val_f1,
        )
        if str(selected["experiment_group"]).startswith("ml_"):
            y_true, scores, trial_ids, class_names = _run_ml_test(selected)
            aggregation = "average_logits"
        else:
            y_true, scores, trial_ids, class_names, model_path = _run_deep_test(
                selected,
                num_workers=max(args.num_workers, 0),
                batch_size=args.batch_size,
            )
            deep_cfg = load_yaml(Path(str(selected["config_path"])))
            aggregation = str(deep_cfg.get("trial_aggregation", "average_logits"))

        statistic = statistics[statistics["experiment_name"] == experiment_name].iloc[0]
        _save_experiment_outputs(
            selected,
            val_f1,
            model_path,
            statistic,
            y_true,
            scores,
            trial_ids,
            class_names,
            aggregation,
        )
        selections.append(
            {
                "experiment_name": experiment_name,
                "experiment_group": selected["experiment_group"],
                "modalities": selected["modalities"],
                "selected_seed": int(selected["seed"]),
                "selection_metric": "val_f1_trial",
                "best_val_f1_trial": val_f1,
                "model_path": str(model_path),
            }
        )
        del y_true, scores, trial_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selection_frame = pd.DataFrame(selections)
    write_csv(selection_frame, selection_path)
    _write_master_report(statistics, selection_frame)
    logger.info("all experiment reports have been saved: %s", RESULTS_ROOT)

if __name__ == "__main__":
    main()
