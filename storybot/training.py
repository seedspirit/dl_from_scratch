"""코퍼스 빈도 집계와 캐시를 준비하고 BPE 병합 규칙을 학습한다."""

from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass

from storybot.args import (
    FrequencyCacheArgs,
    TrainingArgs,
)
from storybot.bpe import MergeRules
from storybot.corpus import split_file_chunks
from storybot.execution import parallel_map
from storybot.profiling import profile_time, record_timing
from storybot.storage import FrequencyCache
from storybot.training_steps import PretokenCounter, PretokenCountInput, learn_merges


@dataclass
class BPETrainer:
    counter: PretokenCounter
    cache: FrequencyCache

    @profile_time("BPETrainer.train")
    def train(self, args: TrainingArgs) -> MergeRules:
        cache_args = None
        counts = None
        if args.cache_dir is not None:
            cache_args = FrequencyCacheArgs(
                directory=args.cache_dir,
                source=args.source,
                tokenization=args.tokenization,
            )
            counts = self.cache.load(cache_args)

        if counts is None:
            chunks = split_file_chunks(
                path=args.source,
                end_token=args.tokenization.end_token,
                num_chunks=args.execution.num_chunks,
                read_buffer_bytes=args.execution.read_buffer_bytes,
            )
            tasks: list[PretokenCountInput] = []
            for chunk in chunks:
                data = PretokenCountInput(chunk, args.tokenization)
                tasks.append(data)
            counts = Counter()
            aggregation_seconds = 0.0
            with ExitStack() as stack:
                stack.enter_context(profile_time("빈도 집계 전체 (풀 생성·전송·합산·정리 포함)"))
                results = stack.enter_context(
                    parallel_map(self.counter.count_chunk, tasks, args.execution)
                )
                for chunk_counts in results:
                    with profile_time("결과 합산", report=False) as timing:
                        counts.update(chunk_counts)
                    aggregation_seconds += timing.seconds
            record_timing("결과 합산 누적", aggregation_seconds)

            if cache_args is not None:
                self.cache.save(cache_args, counts)

        return learn_merges(
            counts=counts,
            vocab_size=args.vocab_size,
            show_progress=args.execution.show_progress,
        )
