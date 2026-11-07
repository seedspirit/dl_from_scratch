"""함수와 코드 구간의 실제 경과 시간을 측정한다.

@profile_time("학습") 또는 with profile_time("빈도 집계 전체") 형태로 사용한다.
병렬 작업의 시간은 서로 겹치므로 워커 시간의 합과 전체 시간은 직접 비교할 수 없다.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import current_process
from time import perf_counter

from rich.table import Table

from storybot.display import console

_records: list[tuple[str, float]] = []


@dataclass
class Timing:
    seconds: float = 0.0


@contextmanager
def profile_time(name: str, *, report: bool = True) -> Iterator[Timing]:
    timing = Timing()
    started = perf_counter()
    try:
        yield timing
    finally:
        timing.seconds = perf_counter() - started
        if report:
            record_timing(name, timing.seconds)


def record_timing(name: str, seconds: float) -> None:
    if current_process().name == "MainProcess":
        _records.append((name, seconds))
    else:
        console.print(f"[profile pid={os.getpid()}] {name}: {seconds:.3f}s", markup=False)


def print_profile_summary() -> None:
    if not _records:
        return
    table = Table(title="프로파일링 · 메인 프로세스 (중첩 구간 포함)")
    table.add_column("측정 구간")
    table.add_column("시간", justify="right")
    for name, seconds in _records:
        table.add_row(name, f"{seconds:.3f}s")
    console.print(table)
    _records.clear()
