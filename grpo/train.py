import time
from itertools import cycle

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.model import GPT
from base.profiling import report, set_enabled, timer
from base.tokenizer import BPETokenizer
from base.utils import get_device
from grpo.dataset import GRPODataset
from grpo.loss import grpo_loss
from grpo.rollout import calculate_reward, collect_rollout, generate_responses

# 설정
device = get_device()
tokenizer_path = "codebot/merge_rules.pkl"
sft_model_path = "codebot/model_sft.pt"
grpo_model_save_path = "codebot/model_grpo.pt"
# 하이퍼파라미터
learning_rate = 7e-6
max_iters = 500
n_update_per_generation = 2  # 같은 생성 데이터로 갱신하는 횟수
eval_interval = 10
epsilon = 0.2  # 클리핑 범위
group_size = 8  # 그룹 크기
batch_size = 32
profile = True  # True면 단계별 소요 시간을 측정해서 마지막에 출력


def evaluate(model: GPT, tokenizer: BPETokenizer, dataset: GRPODataset) -> float:
    """전체 문제를 greedy로 풀어 정답률(%)을 반환."""
    prompts = [prompt for prompt, _ in dataset.data]
    responses = generate_responses(model, tokenizer, prompts, temperature=0)
    model.train()  # generate_batch()가 eval 모드로 바꾸므로 되돌리기
    correct = sum(
        calculate_reward(gt, response)
        for (_, gt), response in zip(dataset.data, responses, strict=True)
    )
    return correct / len(dataset) * 100


# 초기화
set_enabled(profile)
tokenizer = BPETokenizer.load_from(tokenizer_path)
model = GPT.load_from(sft_model_path, device=device)
old_model = GPT.load_from(sft_model_path, device=device)  # 옛 모델
old_model.eval()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
dataset = GRPODataset()
data_iter = cycle(DataLoader(dataset, batch_size=batch_size, shuffle=True))

# 학습 루프
accuracies: list[float] = []
accuracy = 0.0
pbar = tqdm(range(max_iters))
loop_start = time.perf_counter()

for i in pbar:
    # ❶ 배치 데이터 가져오기
    prompts, ground_truths = next(data_iter)

    # ❷ 옛 모델(old_model) 갱신
    old_model.load_state_dict(model.state_dict())

    # ❸ 옛 모델로 여러 샘플을 생성하고, 보상과 어드밴티지 계산
    rollout = collect_rollout(
        old_model, tokenizer, list(prompts), ground_truths.tolist(), group_size
    )

    # ❹ 배치 데이터 만들기
    ids, mask = rollout.to_batch(tokenizer, device)
    advantages = rollout.advantages.to(device)

    # ❺ 생성 데이터로 여러 번 갱신
    for _ in range(n_update_per_generation):
        optimizer.zero_grad()
        loss = grpo_loss(model, old_model, ids, mask, advantages, epsilon)
        with timer("update"):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 기울기 클리핑
            optimizer.step()

    # ❻ 주기적으로 평가
    if i % eval_interval == 0:
        accuracy = evaluate(model, tokenizer, dataset)
        accuracies.append(accuracy)

    pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{accuracy:.1f}%"})

# 단계별 소요 시간
if profile:
    print(report(time.perf_counter() - loop_start))

# 학습된 모델 저장
model.save(grpo_model_save_path)

# 정답률 추이 그래프 (평가는 eval_interval마다 수행)
plt.figure(figsize=(10, 6))
plt.plot(range(0, max_iters, eval_interval), accuracies, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.savefig("accuracy_grpo.png")
