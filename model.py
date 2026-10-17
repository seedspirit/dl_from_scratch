from typing import Self

import torch
from pydantic import Field
from pydantic.dataclasses import dataclass
from torch import nn

from ffn import FFN
from multi_head import MultiHeadAttention
from norm_gelu import LayerNorm


@dataclass
class BlockConfig:
    """모든 트랜스포머 블록이 공통으로 사용하는 구조와 드롭아웃 설정."""

    embed_dim: int = Field(
        gt=0,
        description="토큰 벡터 차원. n_head로 나누어떨어져야 합니다.",
    )
    n_head: int = Field(
        gt=0,
        description="어텐션 헤드 수. 헤드별 차원은 embed_dim // n_head입니다.",
    )
    ff_dim: int | None = Field(
        default=None,
        gt=0,
        description="FFN의 은닉층 차원. None이면 embed_dim의 4배를 사용합니다.",
    )
    dropout_rate: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="임베딩·어텐션·FFN에 적용하는 드롭아웃 확률. 평가 시 비활성화됩니다.",
    )

    def __post_init__(self) -> None:
        if self.embed_dim % self.n_head != 0:
            raise ValueError("embed_dim은 n_head로 나누어떨어져야 합니다.")


@dataclass
class GPTConfig:
    """모델 전체의 어휘·문맥·깊이와 공통 블록 설정."""

    vocab_size: int = Field(
        gt=0,
        description="특수 토큰을 포함한 어휘 크기. 모든 토큰 ID는 이 값 미만이어야 합니다.",
    )
    max_context_len: int = Field(
        gt=0,
        description="한 번에 처리할 최대 토큰 수이자 위치 임베딩 테이블의 길이입니다.",
    )
    n_layer: int = Field(
        gt=0,
        description="쌓을 트랜스포머 블록 수. 각 블록의 가중치는 독립적입니다.",
    )
    block: BlockConfig = Field(
        description="모든 블록의 공통 설정. 임베딩 차원과 드롭아웃은 모델 전체에도 적용합니다.",
    )


class Block(nn.Module):
    norm1: LayerNorm
    attention: MultiHeadAttention
    norm2: LayerNorm
    ffn: FFN

    def __init__(self, config: BlockConfig) -> None:
        super().__init__()
        embed_dim = config.embed_dim
        n_head = config.n_head
        ff_dim = config.ff_dim
        dropout_rate = config.dropout_rate
        head_dim = embed_dim // n_head
        self.norm1 = LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(
            embed_dim, n_head=n_head, head_dim=head_dim, dropout_rate=dropout_rate
        )
        self.norm2 = LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ff_dim, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm1_out = self.norm1(x)  # 레이어 정규화를 통해 입력의 분포를 정규화
        attention_out = self.attention(
            norm1_out
        )  # 멀티 헤드 어텐션을 통해 입력의 상관관계를 학습
        x = x + attention_out  # 잔차 연결로 입력과 어텐션 출력 합쳐 학습 안정화
        norm2_out = self.norm2(x)
        ffn_out = self.ffn(norm2_out)
        x = x + ffn_out  # 잔차 연결로 입력과 FFN 출력 합쳐 학습 안정화
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.max_context_len = config.max_context_len
        embed_dim = config.block.embed_dim

        # 임베딩 레이어와 포지셔널 인코딩 초기화
        self.embed = nn.Embedding(config.vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(config.max_context_len, embed_dim)
        self.dropout = nn.Dropout(config.block.dropout_rate)

        # 트랜스포머 블록 초기화
        self.blocks = nn.ModuleList(
            [Block(config.block) for _ in range(config.n_layer)]
        )

        # 출력 레이어 초기화
        self.norm = nn.LayerNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, config.vocab_size, bias=False)
        self.embed.weight = self.unembed.weight # 임베딩과 출력 레이어의 가중치를 공유하여 파라미터 수 절약

        self.apply_init_weights()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        batch_size, context_len = ids.shape
        device = ids.device

        # 임베딩
        pos = torch.arange(start=0, end=context_len, dtype=torch.long, device=device)
        origin_embed = self.embed(ids)
        positional_embed = self.pos_embed(pos)
        embeding = (
            origin_embed + positional_embed
        )  # 토큰의 위치 정보를 나타내는 포지셔널 인코딩 생성 후 embeding과 합산
        x = self.dropout(embeding)  # 드롭아웃 적용

        # 트랜스포머 블록 통과
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # 출력 레이어
        logits = self.unembed(x)
        return logits

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def apply_init_weights(self) -> None:
        self.apply(self._init_weights)

    def save(self, file_path: str) -> None:
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.config.vocab_size,
            'max_context_len': self.config.max_context_len,
            'embed_dim': self.config.block.embed_dim,
            'n_head': self.config.block.n_head,
            'n_layer': self.config.n_layer,
            'ff_dim': self.config.block.ff_dim,
            'dropout_rate': self.config.block.dropout_rate,
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def load_from(cls, file_path: str, device: str = "cpu") -> Self:
        checkpoint = torch.load(file_path, map_location=device)

        block_config = BlockConfig(
            embed_dim=checkpoint['embed_dim'],
            n_head=checkpoint['n_head'],
            ff_dim=checkpoint['ff_dim'],
            dropout_rate=checkpoint['dropout_rate'],
        )
        config = GPTConfig(
            vocab_size=checkpoint['vocab_size'],
            max_context_len=checkpoint['max_context_len'],
            n_layer=checkpoint['n_layer'],
            block=block_config,
        )
        model = cls(config)

        # 체크포인트에서 로드한 가중치를 모델에 적용
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device) # 모델을 지정한 디바이스로 이동
        return model
