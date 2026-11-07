"""작업 함수와 입력 목록을 받아 프로세스 실행을 관리한다."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from multiprocessing import Pool

from tqdm import tqdm

from storybot.args import ExecutionArgs


@contextmanager
def parallel_map[Input, Result](
    worker: Callable[[Input], Result],
    inputs: Sequence[Input],
    args: ExecutionArgs,
) -> Iterator[Iterator[Result]]:
    if not inputs:
        yield iter(())
        return
    with Pool(processes=min(args.num_processes, len(inputs))) as pool:
        results = pool.imap_unordered(worker, inputs)
        with tqdm(results, total=len(inputs), disable=not args.show_progress) as progress:
            yield iter(progress)
