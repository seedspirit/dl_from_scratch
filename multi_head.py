import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embed_dim: int, n_head: int, head_dim: int, dropout_rate: float
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate

        self.W_q = nn.Linear(embed_dim, n_head * head_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, n_head * head_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, n_head * head_dim, bias=False)
        self.W_o = nn.Linear(n_head * head_dim, embed_dim, bias=False)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, context_len, _ = x.shape
        Q = self.W_q(x)
        V = self.W_v(x)
        K = self.W_k(x)

        # 이미 헤드별로 따로 계산된 결과를 헤드 차원으로 변환하고 전치
        Q = Q.view(batch_size, context_len, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, context_len, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, context_len, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        # 마스크 적용 (삼각 마스크)
        mask = torch.tril(torch.ones(context_len, context_len, device=scores.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # 어텐션 가중치 계산
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        hidden = torch.matmul(weights, V)

        # 헤드 결합과 출력 계산
        hidden = (
            hidden.transpose(1, 2)
            .contiguous()
            .view(batch_size, context_len, self.n_head * self.head_dim)
        )
        out = self.W_o(hidden)

        # 출력 드롭아웃 적용
        out = self.output_dropout(out)
        return out
