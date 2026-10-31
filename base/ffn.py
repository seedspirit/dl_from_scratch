import torch
import torch.nn as nn

from base.norm_gelu import GELU


class FFN(nn.Module):
    layers: nn.Sequential

    def __init__(
        self, x_dim: int, hidden_dim: int | None = None, dropout_rate: float = 0.1
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * x_dim
        self.layers = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, x_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
