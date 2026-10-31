import re
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from base.generate import Tokenizer, generate_batch
from base.model import GPT
from base.profiling import timed


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
        # 같은 프롬프트·응답이 많이 반복되므로 고유 문자열만 토큰화
        cache = {text: tokenizer.encode(text) for text in {*self.prompts, *self.responses}}
        seqs, masks = [], []
        for prompt, response in zip(self.prompts, self.responses, strict=True):
            prompt_ids = cache[prompt]
            response_ids = cache[response]
            seqs.append(torch.tensor(prompt_ids + response_ids))
            masks.append(torch.tensor([0.0] * len(prompt_ids) + [1.0] * len(response_ids)))
        ids = pad_sequence(seqs, batch_first=True, padding_value=0)  # (N, L)
        mask = pad_sequence(masks, batch_first=True, padding_value=0.0)  # (N, L)
        return ids.to(device), mask.to(device)


@timed("reward")
def calculate_reward(ground_truth: int, response: str) -> float:
    """응답의 마지막 정수가 정답이면 1, 아니면 0."""
    numbers = re.findall(r"-?\d+", response)
    return 1.0 if numbers and int(numbers[-1]) == ground_truth else 0.0


def generate_responses(
    model: GPT, tokenizer: Tokenizer, prompts: list[str], temperature: float
) -> list[str]:
    """프롬프트들을 한 번에 생성하고 프롬프트를 뺀 응답 부분만 반환."""
    outputs = generate_batch(model, tokenizer, prompts, temperature=temperature)
    return [output[len(prompt) :] for prompt, output in zip(prompts, outputs, strict=True)]


def collect_rollout(
    model: GPT,
    tokenizer: Tokenizer,
    prompts: list[str],
    ground_truths: list[int],
    group_size: int,
) -> Rollout:
    # 프롬프트마다 group_size개씩 복제해서 전체를 한 배치로 생성
    all_prompts = [prompt for prompt in prompts for _ in range(group_size)]
    all_responses = generate_responses(model, tokenizer, all_prompts, temperature=1.0)

    all_gts = [gt for gt in ground_truths for _ in range(group_size)]
    rewards = torch.tensor(
        [calculate_reward(gt, r) for gt, r in zip(all_gts, all_responses, strict=True)]
    ).view(-1, group_size)  # (프롬프트 수, group_size)
    advantages = rewards - rewards.mean(dim=1, keepdim=True)  # 그룹 평균 대비 상대 보상
    return Rollout(all_prompts, all_responses, advantages.flatten())
