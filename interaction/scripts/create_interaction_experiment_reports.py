from __future__ import annotations
import argparse
import copy
import gc
import sys
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import joblib
import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from interaction.offline_adapter import (
    OfflineInteractionDataset,
    build_label_mapping,
    load_interaction_manifest,
    task_split_from_manifest,
)
from interaction.scripts.train_interaction_deep import encoder_name_map
from interaction.scripts.train_interaction_ml import prepare_combo, split_data
from src.networks.multimodal_late_fusion_net import MultiModalLateFusionNet
from src.data.offline_dataset import offline_collate_fn
from src.training.evaluation import save_confusion_matrix
from src.training.lightning_module import FabricClassificationLightningModule
from src.training.lightning_utils import (
    apply_encoder_override,
    compute_num_devices,
    sanitize_name,
    torch_precision_setting,
)
from src.training.metrics import compute_metrics
from src.training.train_utils import compute_class_weights
from src.utils.config import load_yaml
from src.utils.io import write_csv
from src.utils.logger import setup_logger

EXPECTED_SEEDS = [0, 1, 2, 3, 4]
TEST_METRICS = (
    "accuracy_window",
    "precision_window",
    "recall_window",
    "f1_window",
    "accuracy_trial",
    "precision_trial",
    "recall_trial",
    "f1_trial",
)
RESULTS_ROOT = PROJECT_ROOT / "interaction/results"
TASK_CONFIGS = {
    "primitive": {
        "ml": "interaction/configs/ml/primitive_xgboost.yaml",
        "deep": (
            "interaction/configs/deep/"
            "primitive_late_concat_lstm_efficientnet.yaml"
        ),
    },
    "press_angle": {
        "ml": (
            "interaction/configs/ml/"
            "press_angle_within_fabric_xgboost.yaml"
        ),
        "deep": (
            "interaction/configs/deep/"
            "press_angle_within_fabric_late_concat_lstm_efficientnet.yaml"
        ),
    },
    "slide_trajectory": {
        "ml": (
            "interaction/configs/ml/"
            "slide_trajectory_within_fabric_xgboost.yaml"
        ),
        "deep": (
            "interaction/configs/deep/"
            "slide_trajectory_within_fabric_late_concat_lstm_efficientnet.yaml"
        ),
    },
    "twist_position": {
        "ml": (
            "interaction/configs/ml/"
            "twist_position_within_fabric_xgboost.yaml"
        ),
        "deep": (
            "interaction/configs/deep/"
            "twist_position_within_fabric_late_concat_lstm_efficientnet.yaml"
        ),
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="generate task-stratified Interaction official experiment report"
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        choices=tuple(TASK_CONFIGS),
        default=list(TASK_CONFIGS),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="use only existing statistics tables and best seed table to rebuild Markdown summary report",
    )
    return parser.parse_args()

def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

def load_deep_configs(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deep_config = load_yaml(resolve(path))
    dataset_config = load_yaml(resolve(deep_config["dataset_config"]))
    merged = copy.deepcopy(dataset_config)
    merged.update(deep_config)
    for key in ("audio", "accel", "gelsight", "window"):
        if key in dataset_config:
            merged[key] = dataset_config[key]
    return deep_config, merged

def validate_five_seeds(
    frame: pd.DataFrame,
    keys: list[str],
    table_name: str,
) -> pd.DataFrame:
    required = {"seed", "val_f1_trial", *keys, *TEST_METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{table_name} is missing fields：{sorted(missing)}")
    formal = frame[frame["seed"].astype(int).isin(EXPECTED_SEEDS)].copy()
    if formal.empty:
        raise ValueError(f"{table_name} does not have seed 0-4 official results")
    if formal["val_f1_trial"].isna().any():
        raise ValueError(f"{table_name} of val_f1_trial is incomplete")
    if formal.duplicated([*keys, "seed"]).any():
        raise ValueError(f"{table_name} contains duplicate experiment configurations + seed")
    counts = formal.groupby(keys)["seed"].nunique()
    incomplete = counts[counts != len(EXPECTED_SEEDS)]
    if not incomplete.empty:
        details = {str(key): int(value) for key, value in incomplete.items()}
        raise ValueError(f"{table_name} has not completed all five seed：{details}")
    return formal

def load_task_results(
    task_name: str,
    ml_config: dict[str, Any],
    deep_config: dict[str, Any],
) -> pd.DataFrame:
    ml = validate_five_seeds(
        pd.read_csv(resolve(ml_config["output_csv"])),
        ["modality_combo", "classifier", "fusion"],
        f"{task_name} ML",
    )
    ml["method"] = "ml"
    ml["experiment_name"] = ml["modality_combo"].map(
        lambda combo: f"ML-Interaction-{task_name}-XGBoost-{combo}"
    )
    ml["model_description"] = ml["classifier"].astype(str)

    deep = validate_five_seeds(
        pd.read_csv(resolve(deep_config["output_csv"])),
        ["experiment_name"],
        f"{task_name} Deep",
    )
    deep["method"] = "deep"
    deep["model_description"] = deep["encoders"].astype(str)
    columns = [
        "task_name",
        "seed",
        "experiment_name",
        "method",
        "modality_combo",
        "model_description",
        "fusion",
        "val_f1_trial",
        *TEST_METRICS,
    ]
    return pd.concat([ml[columns], deep[columns]], ignore_index=True)

def aggregate_statistics(results: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "task_name",
        "experiment_name",
        "method",
        "modality_combo",
        "model_description",
        "fusion",
    ]
    aggregations: dict[str, tuple[str, str]] = {"runs": ("seed", "nunique")}
    for metric in TEST_METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    return results.groupby(groups, as_index=False).agg(**aggregations)

def select_best_seed(group: pd.DataFrame) -> pd.Series:
    return group.sort_values(
        ["val_f1_trial", "seed"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]

def semantic_class_names(
    split_df: pd.DataFrame, class_labels: list[str]
) -> list[str]:
    names = (
        split_df[["label", "task_label_name"]]
        .drop_duplicates("label")
        .set_index("label")["task_label_name"]
        .astype(str)
        .to_dict()
    )
    return [names.get(label, label) for label in class_labels]

def run_ml_test(
    selected: pd.Series,
    task_name: str,
    config: dict[str, Any],
    split_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], Path]:
    combo = str(selected["modality_combo"])
    cache_root = resolve(
        config.get("feature_cache_dir", "cache/interaction")
    ) / task_name
    table = prepare_combo(cache_root, combo.split("+"))
    encoder = LabelEncoder().fit(table["label"].astype(str))
    data = split_data(table, encoder)
    model_root = resolve(
        config.get(
            "model_dir", f"checkpoints/interaction/{task_name}/ml"
        )
    )
    model_path = (
        model_root
        / f"seed_{int(selected['seed'])}"
        / f"xgboost_{combo}.joblib"
    )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    scores = model.predict_proba(data["test"]["x"])
    class_labels = [str(value) for value in encoder.classes_]
    return (
        data["test"]["y"],
        scores,
        data["test"]["trial_id"],
        semantic_class_names(split_df, class_labels),
        model_path,
    )

def find_best_checkpoint(
    config: dict[str, Any], experiment_name: str, seed: int
) -> Path:
    run_dir = (
        resolve(config.get("checkpoint_dir", "checkpoints/interaction"))
        / sanitize_name(experiment_name)
        / f"seed_{seed}"
    )
    checkpoints = sorted(
        run_dir.glob("best-*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"no best checkpoint found：{run_dir}")
    return checkpoints[0]

def run_deep_test(
    selected: pd.Series,
    task_name: str,
    deep_config: dict[str, Any],
    config: dict[str, Any],
    input_root: Path,
    split_df: pd.DataFrame,
    label_to_index: dict[str, int],
    class_labels: list[str],
    num_workers: int,
    batch_size: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], Path]:
    combo = str(selected["modality_combo"]).split("+")
    defaults = encoder_name_map(config)
    encoders = {modality: defaults[modality] for modality in combo}
    run_config = apply_encoder_override(config, encoders)
    dataset = OfflineInteractionDataset(
        input_root=input_root,
        split="test",
        split_df=split_df,
        label_to_index=label_to_index,
        modalities=combo,
        config=run_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size or int(deep_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=max(num_workers, 0),
        pin_memory=torch.cuda.is_available(),
        collate_fn=offline_collate_fn,
    )
    train_labels = split_df[split_df["split"] == "train"][["label"]]
    class_weights = (
        compute_class_weights(train_labels, label_to_index)
        if bool(deep_config.get("class_weighted_loss", True))
        else None
    )
    model = MultiModalLateFusionNet(combo, len(class_labels), run_config)
    module = FabricClassificationLightningModule(
        model=model,
        num_classes=len(class_labels),
        learning_rate=float(deep_config.get("learning_rate", 3e-4)),
        weight_decay=float(deep_config.get("weight_decay", 1e-4)),
        scheduler_name=str(deep_config.get("scheduler", "reduce_on_plateau")),
        scheduler_patience=int(deep_config.get("scheduler_patience", 5)),
        class_weights=class_weights,
        average=str(deep_config.get("average", "macro")),
        trial_aggregation=str(
            deep_config.get("trial_aggregation", "average_logits")
        ),
    )
    checkpoint = find_best_checkpoint(
        deep_config,
        str(selected["experiment_name"]),
        int(selected["seed"]),
    )
    L.seed_everything(int(selected["seed"]), workers=True)
    trainer = Trainer(
        accelerator=str(deep_config.get("accelerator", "auto")),
        devices=compute_num_devices(deep_config),
        precision=torch_precision_setting(deep_config),
        deterministic=bool(deep_config.get("deterministic", True)),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=str(PROJECT_ROOT),
    )
    try:
        trainer.test(
            module,
            dataloaders=loader,
            ckpt_path=str(checkpoint),
            verbose=False,
        )
        if module.last_test_y_true is None or module.last_test_scores is None:
            raise RuntimeError("deep-model testing did not return per-window predictions")
        return (
            module.last_test_y_true,
            module.last_test_scores,
            module.last_test_trial_ids,
            semantic_class_names(split_df, class_labels),
            checkpoint,
        )
    finally:
        dataset.close()

def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def aggregate_trials(
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
            predictions = np.argmax(scores[mask], axis=1)
            prediction = int(np.bincount(predictions).argmax())
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

def prediction_frame(
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

def save_level_outputs(
    report_dir: Path,
    matrix_dir: Path,
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
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{level}_classification_report.txt").write_text(
        text, encoding="utf-8"
    )
    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        matrix_dir / f"{level}_confusion_matrix.csv",
        matrix_dir / f"{level}_confusion_matrix.png",
    )
    return {"level": level, "sample_count": len(y_true), **compute_metrics(y_true, y_pred)}

def save_experiment_outputs(
    task_name: str,
    selected: pd.Series,
    statistics: pd.Series,
    model_path: Path,
    y_true: np.ndarray,
    scores: np.ndarray,
    trial_ids: list[str],
    class_names: list[str],
    trial_aggregation: str,
) -> None:
    method = str(selected["method"])
    combo_slug = sanitize_name(str(selected["modality_combo"]))
    prediction_dir = RESULTS_ROOT / "predictions" / task_name / method / combo_slug
    report_dir = RESULTS_ROOT / "reports" / task_name / method / combo_slug
    matrix_dir = (
        RESULTS_ROOT / "confusion_matrices" / task_name / method / combo_slug
    )
    probabilities = scores if method == "ml" else softmax(scores)
    window_pred = np.argmax(scores, axis=1).astype(int)
    write_csv(
        prediction_frame(
            y_true,
            window_pred,
            probabilities,
            np.asarray(trial_ids),
            class_names,
        ),
        prediction_dir / "window_predictions.csv",
    )
    trial_true, trial_pred, trial_scores, unique_trials, counts = aggregate_trials(
        y_true, scores, trial_ids, trial_aggregation
    )
    trial_probabilities = trial_scores if method == "ml" else softmax(trial_scores)
    write_csv(
        prediction_frame(
            trial_true,
            trial_pred,
            trial_probabilities,
            unique_trials,
            class_names,
            num_windows=counts,
        ),
        prediction_dir / "trial_predictions.csv",
    )
    write_csv(
        pd.DataFrame([statistics.to_dict()]),
        report_dir / "five_seed_test_mean_std.csv",
    )
    write_csv(
        pd.DataFrame(
            [
                {
                    "task_name": task_name,
                    "experiment_name": selected["experiment_name"],
                    "method": method,
                    "modality_combo": selected["modality_combo"],
                    "selected_seed": int(selected["seed"]),
                    "selection_metric": "val_f1_trial",
                    "best_val_f1_trial": float(selected["val_f1_trial"]),
                    "model_path": str(model_path),
                }
            ]
        ),
        report_dir / "best_seed_selection.csv",
    )
    metrics = [
        save_level_outputs(
            report_dir, matrix_dir, "window", y_true, window_pred, class_names
        ),
        save_level_outputs(
            report_dir, matrix_dir, "trial", trial_true, trial_pred, class_names
        ),
    ]
    write_csv(
        pd.DataFrame(metrics), report_dir / "selected_seed_test_metrics.csv"
    )

def write_task_summary(
    task_name: str,
    statistics: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    selected = selections.set_index("experiment_name")
    lines = [
        f"# Interaction Task: {task_name}",
        "",
        "official statistics use only seed 0、1、2、3、4；the standard deviation is the sample standard deviation。",
        "best seed based only on validation-set trial-level Macro F1 select。",
        "",
        "## Window-level test results",
        "",
        (
            "| Experiment | Modalities | Runs | Window Accuracy | "
            "Window Macro Precision | Window Macro Recall | Window Macro F1 |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    ordered = statistics.sort_values("experiment_name")
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['experiment_name']} | {row['modality_combo']} | "
            f"{int(row['runs'])} | "
            f"{row['accuracy_window_mean']:.6f} ± "
            f"{row['accuracy_window_std']:.6f} | "
            f"{row['precision_window_mean']:.6f} ± "
            f"{row['precision_window_std']:.6f} | "
            f"{row['recall_window_mean']:.6f} ± "
            f"{row['recall_window_std']:.6f} | "
            f"{row['f1_window_mean']:.6f} ± {row['f1_window_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Trial-level test results",
            "",
            (
                "| Experiment | Modalities | Runs | Trial Accuracy | "
                "Trial Macro F1 | Best Seed | Best Val Trial F1 |"
            ),
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in ordered.iterrows():
        choice = selected.loc[row["experiment_name"]]
        lines.append(
            f"| {row['experiment_name']} | {row['modality_combo']} | "
            f"{int(row['runs'])} | "
            f"{row['accuracy_trial_mean']:.6f} ± {row['accuracy_trial_std']:.6f} | "
            f"{row['f1_trial_mean']:.6f} ± {row['f1_trial_std']:.6f} | "
            f"{int(choice['selected_seed'])} | "
            f"{float(choice['best_val_f1_trial']):.6f} |"
        )
    output = RESULTS_ROOT / "reports" / task_name / "five_seed_test_summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    args = parse_args()
    logger = setup_logger(
        "create_interaction_experiment_reports",
        RESULTS_ROOT / "logs/create_experiment_reports.log",
    )
    all_statistics: list[pd.DataFrame] = []
    all_selections: list[pd.DataFrame] = []
    if args.summary_only:
        for task_name in args.tasks:
            task_table_dir = RESULTS_ROOT / "tables" / task_name
            statistics = pd.read_csv(
                task_table_dir / "five_seed_test_mean_std.csv"
            )
            selections = pd.read_csv(
                task_table_dir / "best_seed_by_experiment.csv"
            )
            write_task_summary(task_name, statistics, selections)
            logger.info("rebuilt the summary report from existing statistics tables：task=%s", task_name)
        return

    for task_name in args.tasks:
        paths = TASK_CONFIGS[task_name]
        ml_config = load_yaml(resolve(paths["ml"]))
        deep_config, merged_config = load_deep_configs(paths["deep"])
        input_root, manifest = load_interaction_manifest(
            merged_config, PROJECT_ROOT
        )
        split_df = task_split_from_manifest(
            manifest, task_name, str(deep_config["target_label"])
        )
        label_to_index, class_labels = build_label_mapping(split_df)
        results = load_task_results(task_name, ml_config, deep_config)
        statistics = aggregate_statistics(results)
        task_table_dir = RESULTS_ROOT / "tables" / task_name
        write_csv(statistics, task_table_dir / "five_seed_test_mean_std.csv")

        selections: list[dict[str, Any]] = []
        for experiment_name, group in results.groupby(
            "experiment_name", sort=True
        ):
            selected = select_best_seed(group)
            logger.info(
                "best seed：task=%s experiment=%s seed=%s val_f1=%.6f",
                task_name,
                experiment_name,
                int(selected["seed"]),
                float(selected["val_f1_trial"]),
            )
            if selected["method"] == "ml":
                y_true, scores, trial_ids, class_names, model_path = run_ml_test(
                    selected, task_name, ml_config, split_df
                )
            else:
                y_true, scores, trial_ids, class_names, model_path = run_deep_test(
                    selected,
                    task_name,
                    deep_config,
                    merged_config,
                    input_root,
                    split_df,
                    label_to_index,
                    class_labels,
                    args.num_workers,
                    args.batch_size,
                )
            statistic = statistics[
                statistics["experiment_name"] == experiment_name
            ].iloc[0]
            save_experiment_outputs(
                task_name,
                selected,
                statistic,
                model_path,
                y_true,
                scores,
                trial_ids,
                class_names,
                str(deep_config.get("trial_aggregation", "average_logits")),
            )
            selections.append(
                {
                    "task_name": task_name,
                    "experiment_name": experiment_name,
                    "method": selected["method"],
                    "modality_combo": selected["modality_combo"],
                    "selected_seed": int(selected["seed"]),
                    "selection_metric": "val_f1_trial",
                    "best_val_f1_trial": float(selected["val_f1_trial"]),
                    "model_path": str(model_path),
                }
            )
            del y_true, scores, trial_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        selection_frame = pd.DataFrame(selections)
        write_csv(selection_frame, task_table_dir / "best_seed_by_experiment.csv")
        write_task_summary(task_name, statistics, selection_frame)
        all_statistics.append(statistics)
        all_selections.append(selection_frame)

    write_csv(
        pd.concat(all_statistics, ignore_index=True),
        RESULTS_ROOT / "tables/all_experiments_test_mean_std.csv",
    )
    write_csv(
        pd.concat(all_selections, ignore_index=True),
        RESULTS_ROOT / "tables/best_seed_by_experiment.csv",
    )
    logger.info("Interaction task-specific reports have all been saved：%s", RESULTS_ROOT)

if __name__ == "__main__":
    main()
