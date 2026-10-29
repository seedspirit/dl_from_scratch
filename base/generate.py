from typing import Protocol

import torch

from base.model import GPT
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
