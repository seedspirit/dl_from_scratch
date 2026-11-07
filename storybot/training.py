"""코퍼스 빈도 집계와 캐시를 준비하고 BPE 병합 규칙을 학습한다."""

from collections import Counter
from dataclasses import dataclass

from storybot.args import (
    FrequencyCacheArgs,
    TrainingArgs,
)
from storybot.bpe import MergeRules
from storybot.corpus import split_file_chunks
from storybot.execution import parallel_map
from storybot.storage import FrequencyCache
from storybot.training_steps import PretokenCounter, PretokenCountInput, learn_merges


@dataclass
class BPETrainer:
    counter: PretokenCounter
    cache: FrequencyCache

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
            with parallel_map(self.counter.count_chunk, tasks, args.execution) as results:
                for chunk_counts in results:
                    counts.update(chunk_counts)

            if cache_args is not None:
                self.cache.save(cache_args, counts)

        return learn_merges(
            counts=counts,
            vocab_size=args.vocab_size,
            show_progress=args.execution.show_progress,
        )
