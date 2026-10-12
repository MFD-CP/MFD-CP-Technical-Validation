from __future__ import annotations
import torch
from torch import nn

def _load_torchvision_models():
    try:
        from torchvision import models
    except Exception as exc:
        raise ImportError(
            "cannot import torchvision.models。ensure torch and torchvision versions are compatible。"
        ) from exc
    return models

def _get_weights(models, pretrained: bool):
    if not pretrained:
        return None
    return models.EfficientNet_B0_Weights.IMAGENET1K_V1

class ImageEncoder(nn.Module):

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        output_dim: int = 256,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        models = _load_torchvision_models()
        architecture = architecture.lower()
        self.architecture = architecture

        if architecture != "efficientnet_b0":
            raise ValueError("architecture must be efficientnet_b0")
        try:
            backbone = models.efficientnet_b0(
                weights=_get_weights(models, pretrained)
            )
        except Exception:
            backbone = models.efficientnet_b0(pretrained=pretrained)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            batch_size, frames, channels, height, width = x.shape
            x = x.reshape(batch_size * frames, channels, height, width)
            feat = self.backbone(x)
            feat = feat.reshape(batch_size, frames, -1).mean(dim=1)
        elif x.ndim == 4:
            feat = self.backbone(x)
        else:
            raise ValueError("ImageEncoder input must be [B,3,H,W] or [B,K,3,H,W]")
        return self.projection(feat)
