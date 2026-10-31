"""단계별 실행 시간 측정.

set_enabled(True)를 호출했을 때만 측정한다. 꺼져 있으면(기본값) 아무 일도 하지 않으므로
평소 학습 속도에는 영향이 없다.

    set_enabled(True)

    @timed("tokenize")
    def encode(...): ...

    with timer("update"):
        loss.backward()
        optimizer.step()

    print(report())
"""

import time
from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

import torch

P = ParamSpec("P")
R = TypeVar("R")

_enabled = False

_total_seconds: dict[str, float] = defaultdict(float)
_call_counts: dict[str, int] = defaultdict(int)


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def _sync() -> None:
    # GPU 연산은 비동기라서 동기화하지 않으면 실제 연산 시간이 다음 단계로 넘어가 측정된다
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timer(name: str) -> Generator[None]:
    if not _enabled:
        yield
        return
    _sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        _total_seconds[name] += time.perf_counter() - start
        _call_counts[name] += 1


def timed(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with timer(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def report(wall_seconds: float | None = None) -> str:
    """단계별 호출 수·총 시간·평균 시간·비율을 표로 반환. 비율은 wall_seconds 기준."""
    if not _total_seconds:
        return "(측정된 단계 없음: set_enabled(True)로 측정을 켜세요)"
    total = wall_seconds or sum(_total_seconds.values())
    lines = [f"{'stage':<12}{'calls':>8}{'total(s)':>10}{'avg(ms)':>10}{'share':>8}"]
    for name, seconds in sorted(_total_seconds.items(), key=lambda kv: -kv[1]):
        calls = _call_counts[name]
        lines.append(
            f"{name:<12}{calls:>8}{seconds:>10.2f}{seconds / calls * 1000:>10.2f}"
            f"{seconds / total * 100:>7.1f}%"
        )
    return "\n".join(lines)


def reset() -> None:
    _total_seconds.clear()
    _call_counts.clear()
