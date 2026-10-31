from typing import Protocol

import torch

from base.model import GPT
from base.profiling import timed
from base.tokenizer import BPETokenizer
from base.utils import get_device


class Tokenizer(Protocol):
    end_token_id: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_token_length: int = 50,
    temperature: float = 1.0,
) -> str:
    model.eval()  # 평가 모드로 전환

    # 프롬프트 토큰화
    device = next(model.parameters()).device  # 모델 파라미터의 디바이스 가져오기
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

    generated_ids = ids.clone()  # 생성된 토큰 시퀀스 초기화
    max_context_len = model.max_context_len

    for _ in range(max_token_length):
        if ids.size(1) > max_context_len:
            ids = ids[:, -max_context_len:]  # 컨텍스트 길이를 초과하면 앞부분 자르기

        logits = model(ids)[:, -1, :]  # 마지막 토큰의 로짓만 사용
        if temperature == 0:  # 온도가 0이면 가장 높은 확률의 토큰 선택
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:  # 온도가 0이 아니면 샘플링
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.end_token_id:
            break

        ids = torch.cat([ids, next_id], dim=-1)
        generated_ids = torch.cat([generated_ids, next_id], dim=-1)

    output = tokenizer.decode(generated_ids[0].tolist())
    return output


@torch.no_grad()
def generate_batch(
    model: GPT,
    tokenizer: Tokenizer,
    prompts: list[str],
    max_token_length: int = 50,
    temperature: float = 1.0,
) -> list[str]:
    """여러 프롬프트를 한 번에 생성해서 generate()와 같은 형식(프롬프트 + 응답)으로 반환.

    모델에 패딩 마스크가 없으므로 토큰 길이가 같은 프롬프트끼리 묶어서 배치로 처리한다.
    """
    model.eval()
    device = next(model.parameters()).device
    # 같은 프롬프트가 반복되는 경우가 많으므로(GRPO 그룹) 고유 프롬프트만 토큰화
    unique_ids = {prompt: tokenizer.encode(prompt) for prompt in set(prompts)}
    encoded = [unique_ids[prompt] for prompt in prompts]

    # 토큰 길이별로 프롬프트 인덱스 묶기
    groups: dict[int, list[int]] = {}
    for idx, prompt_ids in enumerate(encoded):
        groups.setdefault(len(prompt_ids), []).append(idx)

    outputs = [""] * len(prompts)
    for indices in groups.values():
        ids = torch.tensor([encoded[i] for i in indices], dtype=torch.long, device=device)
        new_ids = _sample_batch(model, ids, tokenizer.end_token_id, max_token_length, temperature)
        for i, row in zip(indices, new_ids, strict=True):
            outputs[i] = tokenizer.decode(encoded[i] + row)
    return outputs


@timed("generate")
def _sample_batch(
    model: GPT,
    ids: torch.Tensor,
    end_token_id: int,
    max_token_length: int,
    temperature: float,
) -> list[list[int]]:
    """(B, L) 프롬프트에서 행마다 종료 토큰 전까지의 새 토큰 ID 리스트를 반환."""
    finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
    new_tokens: list[torch.Tensor] = []

    for _ in range(max_token_length):
        logits = model(ids[:, -model.max_context_len :])[:, -1, :]
        if temperature == 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        finished |= next_id.squeeze(-1) == end_token_id
        if finished.all():  # 모든 행이 종료 토큰을 냈으면 중단
            break

        ids = torch.cat([ids, next_id], dim=-1)
        new_tokens.append(next_id)

    if not new_tokens:
        return [[] for _ in range(ids.size(0))]

    # 행마다 첫 종료 토큰 이후는 버리기
    rows = torch.cat(new_tokens, dim=-1).tolist()
    return [row[: row.index(end_token_id)] if end_token_id in row else row for row in rows]


if __name__ == "__main__":
    device = get_device()
    model_path = "./codebot/model_pretrain.pt"
    tokenizer_path = "./codebot/merge_rules.pkl"
    prompt = "def"  # 생성을 시작할 프롬프트
    max_new_tokens = 200  # 생성할 토큰 수의 상한
    temperature = 1.0  # 온도 파라미터
    # 모델과 토크나이저 불러오기
    tokenizer = BPETokenizer.load_from(tokenizer_path)
    model = GPT.load_from(model_path, device=device)
    # 텍스트 생성
    for i in range(5):
        print(f"--- 샘플 {i + 1} ---")
        generated_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_token_length=max_new_tokens,
            temperature=temperature,
        )
        print(generated_text)
