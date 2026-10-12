from __future__ import annotations
from typing import Any
import torch
from torch import nn
from src.networks.encoder_factory import build_encoder
from src.networks.classifier.mlp_classifier import MLPClassifier

class MultiModalLateFusionNet(nn.Module):

    def __init__(
        self,
        modality_combo: list[str],
        num_classes: int,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        cfg = config or {}
        model_cfg = cfg.get("model", cfg)
        self.feature_dim = int(model_cfg.get("feature_dim", 256))
        self.modality_combo = [m.lower() for m in modality_combo]
        self.encoders = nn.ModuleDict()
        self.encoder_names: dict[str, str] = {}

        for modality in self.modality_combo:
            encoder, encoder_name = build_encoder(
                modality=modality,
                config=cfg,
                output_dim=self.feature_dim,
            )
            self.encoders[modality] = encoder
            self.encoder_names[modality] = encoder_name

        clf_cfg = model_cfg.get("classifier", {})
        self.classifier = MLPClassifier(
            feature_dim=self.feature_dim * len(self.modality_combo),
            num_classes=num_classes,
            hidden_dims=clf_cfg.get("hidden_dims", [512, 256]),
            dropout=float(clf_cfg.get("dropout", 0.3)),
            use_batch_norm=bool(clf_cfg.get("use_batch_norm", True)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        for modality in self.modality_combo:
            if batch.get(modality) is None:
                raise ValueError(f"batch is missing modality: {modality}")
            features.append(self.encoders[modality](batch[modality]))
        fused = torch.cat(features, dim=1)
        return self.classifier(fused)
