from __future__ import annotations
from typing import Any
import lightning as L
import numpy as np
import torch
from torch import nn
from src.training.metrics import compute_window_and_trial_metrics

class FabricClassificationLightningModule(L.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        scheduler_name: str = "reduce_on_plateau",
        scheduler_patience: int = 5,
        class_weights: torch.Tensor | None = None,
        average: str = "macro",
        trial_aggregation: str = "average_logits",
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.scheduler_patience = scheduler_patience
        self.average = average
        self.trial_aggregation = trial_aggregation

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        self.validation_outputs: list[dict[str, Any]] = []
        self.test_outputs: list[dict[str, Any]] = []
        self.last_val_metrics: dict[str, float] = {}
        self.last_test_metrics: dict[str, float] = {}
        self.last_test_y_true: np.ndarray | None = None
        self.last_test_scores: np.ndarray | None = None
        self.last_test_trial_ids: list[str] = []

        self.save_hyperparameters(ignore=["model", "class_weights"])

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch)

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        labels = batch["label"]
        logits = self(batch)
        loss = self.criterion(logits, labels)
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=(stage in {"train", "val"}),
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=labels.size(0),
        )
        if stage in {"val", "test"}:
            output = {
                "labels": labels.detach().cpu(),
                "scores": logits.detach().cpu(),
                "trial_ids": [str(x) for x in batch["trial_id"]],
            }
            if stage == "val":
                self.validation_outputs.append(output)
            else:
                self.test_outputs.append(output)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def _epoch_metrics(self, outputs: list[dict[str, Any]], stage: str) -> dict[str, float]:
        if not outputs:
            return {}
        y_true = torch.cat([item["labels"] for item in outputs], dim=0).numpy()
        scores = torch.cat([item["scores"] for item in outputs], dim=0).numpy()
        trial_ids: list[str] = []
        for item in outputs:
            trial_ids.extend(item["trial_ids"])

        metrics = compute_window_and_trial_metrics(
            y_true=y_true,
            scores=scores,
            trial_ids=trial_ids,
            average=self.average,
            trial_aggregation=self.trial_aggregation,
        )
        for key, value in metrics.items():
            self.log(f"{stage}_{key}", value, prog_bar=key in {"f1_trial", "accuracy_trial"}, sync_dist=False)
        if stage == "test":
            self.last_test_y_true = y_true
            self.last_test_scores = scores
            self.last_test_trial_ids = trial_ids
        return metrics

    def on_validation_epoch_end(self) -> None:
        self.last_val_metrics = self._epoch_metrics(self.validation_outputs, "val")
        self.validation_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self.last_test_metrics = self._epoch_metrics(self.test_outputs, "test")
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.scheduler_name == "none":
            return optimizer
        if self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(int(self.trainer.max_epochs), 1),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=self.scheduler_patience,
            factor=0.5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1_trial",
                "interval": "epoch",
                "frequency": 1,
            },
        }
