from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }

def aggregate_trial_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    trial_ids: list[str] | np.ndarray,
    method: str = "average_logits",
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    trial_ids = np.asarray(trial_ids)
    true_rows = []
    pred_rows = []
    for trial_id in np.unique(trial_ids):
        mask = trial_ids == trial_id
        labels = y_true[mask].astype(int)
        if method == "majority_vote":
            window_pred = np.argmax(scores[mask], axis=1)
            pred = int(np.bincount(window_pred).argmax())
        else:
            pred = int(np.argmax(np.mean(scores[mask], axis=0)))
        true = int(np.bincount(labels).argmax())
        true_rows.append(true)
        pred_rows.append(pred)
    return np.asarray(true_rows, dtype=int), np.asarray(pred_rows, dtype=int)

def compute_window_and_trial_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    trial_ids: list[str] | np.ndarray,
    average: str = "macro",
    trial_aggregation: str = "average_logits",
) -> dict[str, float]:
    window_pred = np.argmax(scores, axis=1)
    window_metrics = compute_metrics(y_true, window_pred, average=average)
    trial_true, trial_pred = aggregate_trial_predictions(
        y_true=y_true,
        scores=scores,
        trial_ids=trial_ids,
        method=trial_aggregation,
    )
    trial_metrics = compute_metrics(trial_true, trial_pred, average=average)
    return {
        "accuracy_window": window_metrics["accuracy"],
        "precision_window": window_metrics["precision"],
        "recall_window": window_metrics["recall"],
        "f1_window": window_metrics["f1"],
        "weighted_precision_window": window_metrics["weighted_precision"],
        "weighted_recall_window": window_metrics["weighted_recall"],
        "weighted_f1_window": window_metrics["weighted_f1"],
        "accuracy_trial": trial_metrics["accuracy"],
        "precision_trial": trial_metrics["precision"],
        "recall_trial": trial_metrics["recall"],
        "f1_trial": trial_metrics["f1"],
        "weighted_precision_trial": trial_metrics["weighted_precision"],
        "weighted_recall_trial": trial_metrics["weighted_recall"],
        "weighted_f1_trial": trial_metrics["weighted_f1"],
    }
