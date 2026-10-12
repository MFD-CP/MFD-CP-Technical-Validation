from __future__ import annotations
from typing import Any
import torch
from torch import nn
from src.networks.encoder_factory import build_encoder
from src.networks.classifier.mlp_classifier import MLPClassifier

class SingleModalNet(nn.Module):

    def __init__(
        self,
        modality: str,
        num_classes: int,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = config or {}
        model_cfg = cfg.get("model", cfg)
        self.modality = modality.lower()
        self.feature_dim = int(model_cfg.get("feature_dim", 256))

        self.encoder, self.encoder_name = build_encoder(
            modality=self.modality,
            config=cfg,
            output_dim=self.feature_dim,
        )
        clf_cfg = model_cfg.get("classifier", {})
        self.classifier = MLPClassifier(
            feature_dim=self.feature_dim,
            num_classes=num_classes,
            hidden_dims=clf_cfg.get("hidden_dims", [512, 256]),
            dropout=float(clf_cfg.get("dropout", 0.3)),
            use_batch_norm=bool(clf_cfg.get("use_batch_norm", True)),
        )

    def forward(self, batch: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        x = batch[self.modality] if isinstance(batch, dict) else batch
        feature = self.encoder(x)
        return self.classifier(feature)
