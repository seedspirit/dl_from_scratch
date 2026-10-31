#!/usr/bin/env python3

from base.generate import generate
from base.model import GPT
from base.tokenizer import BPETokenizer
from base.utils import get_device

# 설정
device = get_device()
model_path = "codebot/model_sft.pt"
tokenizer_path = "codebot/merge_rules.pkl"
max_new_tokens = 200
temperature = 1.0


def format_prompt(user_message: str) -> str:
    return f"### Instruction:\n{user_message}\n\n### Response:\n"


# ❶ 모델과 토크나이저 불러오기
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(model_path, device=device)

while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue

    # ❷ 프롬프트 포맷팅과 텍스트 생성
    prompt = format_prompt(user_input)
    response = generate(model, tokenizer, prompt, max_new_tokens, temperature)

    # ❸ 비서의 응답 부분 추출
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    # 줄바꿈 포함 여부에 따라 출력 형식 전환
    if "\n" in response:
        print(f"Bot:\n{response}")
    else:
        print(f"Bot: {response}")
