import re
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from base.generate import Tokenizer, generate
from base.model import GPT


@dataclass
class Rollout:
    """프롬프트별로 group_size개씩 생성한 응답과 각 응답의 어드밴티지."""

    prompts: list[str]
    responses: list[str]
    advantages: torch.Tensor  # (N,)

    def to_batch(
        self, tokenizer: Tokenizer, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """프롬프트+응답 토큰 ID와 응답 위치만 1인 마스크를 패딩해서 반환."""
        seqs, masks = [], []
        for prompt, response in zip(self.prompts, self.responses, strict=True):
            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response)
            seqs.append(torch.tensor(prompt_ids + response_ids))
            masks.append(torch.tensor([0.0] * len(prompt_ids) + [1.0] * len(response_ids)))
        ids = pad_sequence(seqs, batch_first=True, padding_value=0)  # (N, L)
        mask = pad_sequence(masks, batch_first=True, padding_value=0.0)  # (N, L)
        return ids.to(device), mask.to(device)


def calculate_reward(ground_truth: int, response: str) -> float:
    """응답의 마지막 정수가 정답이면 1, 아니면 0."""
    numbers = re.findall(r"-?\d+", response)
    return 1.0 if numbers and int(numbers[-1]) == ground_truth else 0.0


def generate_response(model: GPT, tokenizer: Tokenizer, prompt: str, temperature: float) -> str:
    """프롬프트를 뺀 응답 부분만 반환."""
    return generate(model, tokenizer, prompt, temperature=temperature)[len(prompt) :]


def collect_rollout(
    model: GPT,
    tokenizer: Tokenizer,
    prompts: list[str],
    ground_truths: list[int],
    group_size: int,
) -> Rollout:
    all_prompts: list[str] = []
    all_responses: list[str] = []
    all_advantages: list[torch.Tensor] = []
    for prompt, ground_truth in zip(prompts, ground_truths, strict=True):
        responses = [
            generate_response(model, tokenizer, prompt, temperature=1.0) for _ in range(group_size)
        ]
        rewards = torch.tensor([calculate_reward(ground_truth, r) for r in responses])
        all_prompts += [prompt] * group_size
        all_responses += responses
        all_advantages.append(rewards - rewards.mean())  # 그룹 평균 대비 상대 보상
    return Rollout(all_prompts, all_responses, torch.cat(all_advantages))
