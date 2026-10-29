from itertools import cycle
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.model import GPT, BlockConfig, GPTConfig
from base.utils import get_device
from pretrain.dataset import TokenDataset

# 설정
device = get_device()
print(device)
data_path = "codebot/tiny_codes.bin"
tokenizer_path = "codebot/merge_rules.pkl"
model_save_path = "codebot/model_pretrain.pt"
# 하이퍼파라미터
context_len = 256
vocab_size = 1000
batch_size = 32
learning_rate = 3e-4
max_iters = 20000
embed_dim = 384
n_head = 6
n_layer = 6
ff_dim = 4 * embed_dim  # 일반적으로 임베딩 차원의 4배로 설정
dropout_rate = 0.1

# nn.Embedding은 int64 인덱스를 받으므로 uint16 토큰을 변환
ids = torch.from_numpy(np.fromfile(data_path, dtype=np.uint16).astype(np.int64))
dataset = TokenDataset(ids, context_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

config = GPTConfig(
    vocab_size=vocab_size,
    max_context_len=context_len,
    n_layer=n_layer,
    block=BlockConfig(
        embed_dim=embed_dim,
        n_head=n_head,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate,
    ),
)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

losses: list[float] = []
data_iter = cycle(dataloader)
pbar = tqdm(range(max_iters))

for _ in pbar:
    batch_x, batch_y = next(data_iter)
    batch_x = cast(torch.Tensor, batch_x)
    batch_y = cast(torch.Tensor, batch_y)
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    # 예측
    # logits: [batch_size, context_len, vocab_size]
    logits = model(batch_x)
    # 손실 계산을 위해 logits와 batch_y를 적절한 형태로 변환
    input_ = logits.view(-1, logits.size(-1))
    target_ = batch_y.view(-1)
    # 손실 계산
    loss = F.cross_entropy(input_, target_)

    # 기울기 계산과 파라미터 갱신
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    pbar.set_postfix({"loss": f"{loss.item():.4f}"})

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_pretrain.png")
model.save(model_save_path)
