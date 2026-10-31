from itertools import cycle
from typing import cast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.model import GPT
from base.tokenizer import BPETokenizer
from base.utils import get_device
from sft.dataset import SFTDataset

# 설정
device = get_device()
data_path = "codebot/tiny_codes_sft.json"
tokenizer_path = "codebot/merge_rules.pkl"
pretrain_model_path = "codebot/model_pretrain.pt"
sft_model_save_path = "codebot/model_sft.pt"
# 하이퍼파라미터
context_len = 256
batch_size = 32
learning_rate = 3e-4
max_iters = 500  # ❶

# 토크나이저와 데이터셋 준비
tokenizer = BPETokenizer.load_from(tokenizer_path)
dataset = SFTDataset(data_path, tokenizer, context_len)  # ❷
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델과 옵티마이저
model = GPT.load_from(pretrain_model_path, device=str(device))  # ❸
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 학습 루프
losses: list[float] = []
data_iter = cycle(dataloader)
pbar = tqdm(range(max_iters))

for _ in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x = cast(torch.Tensor, batch_x)
    batch_y = cast(torch.Tensor, batch_y)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    logits = model(batch_x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch_y.view(-1),
        ignore_index=-100,  # ❹ 레이블이 -100인 위치를 손실 계산에서 제외
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

# 모델 저장
model.save(sft_model_save_path)
