import math

import torch
import torch.nn as nn

from multi_head import MultiHeadAttention

eps = 1e-5


class LayerNorm(nn.Module):
    gamma: nn.Parameter
    beta: nn.Parameter

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + eps)
        return self.gamma * x + self.beta


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tanh 근사식: 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
        return (
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        )
