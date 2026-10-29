import torch


def get_device():
    """사용 가능한 디바이스를 자동으로 찾아 반환"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
