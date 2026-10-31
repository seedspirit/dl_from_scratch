import json

import torch
from torch.utils.data import Dataset

from base.tokenizer import BPETokenizer


class SFTDataset(Dataset):
    tokenizer: BPETokenizer
    context_len: int
    samples: list[tuple[list[int], list[int]]]

    # ❶ 생성자
    def __init__(self, data_path: str, tokenizer: BPETokenizer, context_len: int) -> None:
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.samples = []
        with open(data_path) as f:
            data = json.load(f)
        for item in data:
            ids, labels = self._create_sample(item["instruction"], item["response"])
            self.samples.append((ids, labels))

    # ❷ 샘플 생성
    def _create_sample(self, instruction: str, response: str) -> tuple[list[int], list[int]]:
        # 프롬프트(지시)와 응답 포맷팅
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        response = f"{response}<|endoftext|>"
        # 토큰화
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)
        # 입력 시퀀스와 레이블 생성(프롬프트 부분은 -100으로 마스킹)
        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        # 언어 모델용으로 시프트(입력과 정답이 한 칸씩 어긋나게)
        ids = ids[:-1]
        labels = labels[1:]
        # context_len에 맞춰 패딩 혹은 잘라내기
        pad_len = self.context_len - len(ids)
        if pad_len > 0:
            ids = ids + [0] * pad_len  # 패딩 ID로 0 사용
            labels = labels + [-100] * pad_len
        elif pad_len < 0:
            ids = ids[: self.context_len]
            labels = labels[: self.context_len]
        return ids, labels

    # ❸ DataLoader용 메서드
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, labels = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
