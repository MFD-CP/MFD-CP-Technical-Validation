from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    csv_path: str | Path,
    png_path: str | Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    csv_path = Path(csv_path)
    png_path = Path(png_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.5), max(5, len(class_names) * 0.5)))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

