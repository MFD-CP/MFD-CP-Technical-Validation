from __future__ import annotations
import torch
from torch import nn

class MLPClassifier(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)
