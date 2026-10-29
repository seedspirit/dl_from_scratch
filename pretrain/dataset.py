import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    tokens: torch.Tensor
    context_len: int

    def __init__(self, tokens: torch.Tensor, context_len: int) -> None:
        self.tokens = tokens
        self.context_len = context_len

    def __len__(self) -> int:
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx : idx + self.context_len]
        y = self.tokens[idx + 1 : idx + 1 + self.context_len]
        return x, y
