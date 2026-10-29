import torch
import torch.nn.functional as F

from base.model import GPT
from base.profiling import timed


def token_probs(model: GPT, ids: torch.Tensor) -> torch.Tensor:
    """각 위치에서 실제 다음 토큰에 모델이 준 확률. (B, C) -> (B, C-1)"""
    probs = F.softmax(model(ids)[:, :-1, :], dim=-1)  # (B, C-1, V)
    next_ids = ids[:, 1:].unsqueeze(-1)  # (B, C-1, 1)
    return probs.gather(dim=-1, index=next_ids).squeeze(-1)


@timed("loss")
def grpo_loss(
    model: GPT,
    old_model: GPT,
    ids: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    probs = token_probs(model, ids)
    with torch.no_grad():
        old_probs = token_probs(old_model, ids)

    # 토큰별 확률비(0으로 나눠지는 일을 막기 위해 작은 값을 더함)
    ratio = probs / (old_probs + 1e-8)
    advantages = advantages.unsqueeze(-1)  # (N, 1): 응답 전체 토큰에 같은 값 적용
    objective = torch.min(
        ratio * advantages,
        ratio.clamp(1 - epsilon, 1 + epsilon) * advantages,
    )

    # 응답 토큰만 남기고(마스크도 한 칸 시프트) 샘플 수로 정규화
    objective = objective * mask[:, 1:]
    return -objective.sum() / ids.size(0)
