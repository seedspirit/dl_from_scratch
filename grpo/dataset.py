from torch.utils.data import Dataset


class GRPODataset(Dataset):
    """한 자리 수 덧셈 문제(1+1 ~ 9+9)와 정답 쌍."""

    data: list[tuple[str, int]]

    def __init__(self) -> None:
        self.data = [
            (f"### Instruction:\n{i}+{j}=\n\n### Response:\n", i + j)
            for i in range(1, 10)
            for j in range(1, 10)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.data[idx]
