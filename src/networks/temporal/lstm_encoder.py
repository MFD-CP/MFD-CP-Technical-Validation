from __future__ import annotations
import torch
from torch import nn

class LSTMEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_dim: int = 256,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be greater than or equal to 1")
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size * self.num_directions),
            nn.Linear(hidden_size * self.num_directions, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last = hidden[-1]
        return self.projection(last)
