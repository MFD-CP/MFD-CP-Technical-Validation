from __future__ import annotations
import torch
from torch import nn

class TCNBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if out.shape[-1] != residual.shape[-1]:
            min_len = min(out.shape[-1], residual.shape[-1])
            out = out[..., :min_len]
            residual = residual[..., :min_len]
        return self.relu(out + residual)

class TCNEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int] | tuple[int, ...] = (64, 128, 256),
        output_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.2,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        blocks = []
        current = in_channels
        for idx, out_channels in enumerate(hidden_channels):
            blocks.append(
                TCNBlock(
                    current,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilations[idx % len(dilations)],
                    dropout=dropout,
                )
            )
            current = out_channels
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(current, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = self.pool(out).squeeze(-1)
        return self.projection(out)
