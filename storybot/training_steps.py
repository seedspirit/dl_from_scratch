"""사전 토큰 빈도 집계와 BPE 병합 규칙 학습."""

import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise

from tqdm import tqdm

from storybot.args import TokenizationArgs
from storybot.bpe import BYTE_VOCAB_SIZE, MergeRule, MergeRules, PreTokenizer
from storybot.corpus import FileChunk


@dataclass(frozen=True)
class PretokenCountInput:
    chunk: FileChunk
    tokenization: TokenizationArgs


@dataclass
class PretokenCounter:
    pretokenizer: PreTokenizer

    def count_chunk(self, data: PretokenCountInput) -> Counter[str]:
        counts: Counter[str] = Counter()
        for text, _ in data.chunk.iter_documents():
            counts.update(self.pretokenizer.split(text, data.tokenization.pattern))
        return counts


def learn_merges(counts: Counter[str], vocab_size: int, show_progress: bool) -> MergeRules:
    num_merges = vocab_size - BYTE_VOCAB_SIZE - 1
    state = _MergeState(counts)
    rules: list[MergeRule] = []
    for step in tqdm(range(num_merges), desc="Training BPE", disable=not show_progress):
        pair = state.most_frequent_pair()
        if pair is None:
            break
        rule = MergeRule(*pair, BYTE_VOCAB_SIZE + step)
        state.apply_merge(rule)
        rules.append(rule)
    return MergeRules(entries=tuple(rules))


class _MergeState:
    """병합에 따라 함께 갱신되는 빈도, 역인덱스, 우선순위 큐."""

    _seq_counts: dict[tuple[int, ...], int]  # 토큰 시퀀스별 등장 횟수
    _pair_counts: Counter[tuple[int, int]]  # 시퀀스 빈도로 가중한 인접 페어 빈도
    # 페어를 포함하는 시퀀스의 역인덱스
    _pair_to_seqs: defaultdict[tuple[int, int], set[tuple[int, ...]]]
    _changed_pairs: set[tuple[int, int]]  # 현재 병합에서 빈도가 바뀐 페어
    _heap: list[tuple[int, int, int]]  # (-빈도, -왼쪽 ID, -오른쪽 ID)

    def __init__(self, counts: Counter[str]) -> None:
        self._seq_counts = {}
        self._pair_counts = Counter()
        self._pair_to_seqs = defaultdict(set)
        self._changed_pairs = set()
        for text, count in counts.items():
            self._add_sequence(tuple(text.encode("utf-8")), count)
        self._rebuild_heap()
        self._changed_pairs.clear()

    def _rebuild_heap(self) -> None:
        self._heap = [(-count, -pair[0], -pair[1]) for pair, count in self._pair_counts.items()]
        heapq.heapify(self._heap)

    def most_frequent_pair(self) -> tuple[int, int] | None:
        while self._heap:
            negative_count, first, second = self._heap[0]
            pair = (-first, -second)
            if self._pair_counts.get(pair, 0) == -negative_count:
                return pair
            heapq.heappop(self._heap)
        return None

    def apply_merge(self, rule: MergeRule) -> None:
        for seq in self._pair_to_seqs.pop(rule.pair, set()):
            count = self._seq_counts[seq]
            self._remove_sequence(seq, count)
            self._add_sequence(rule.apply(seq), count)
        for changed in self._changed_pairs:
            frequency = self._pair_counts.get(changed, 0)
            if frequency:
                heapq.heappush(self._heap, (-frequency, -changed[0], -changed[1]))
        self._changed_pairs.clear()
        if len(self._heap) > max(1024, 4 * len(self._pair_counts)):
            self._rebuild_heap()

    def _add_sequence(self, seq: tuple[int, ...], count: int) -> None:
        self._seq_counts[seq] = count
        for pair, occurrences in Counter(pairwise(seq)).items():
            self._changed_pairs.add(pair)
            self._pair_counts[pair] += occurrences * count
            self._pair_to_seqs[pair].add(seq)

    def _remove_sequence(self, seq: tuple[int, ...], count: int) -> None:
        del self._seq_counts[seq]
        for pair, occurrences in Counter(pairwise(seq)).items():
            self._changed_pairs.add(pair)
            self._pair_counts[pair] -= occurrences * count
            if self._pair_counts[pair] <= 0:
                del self._pair_counts[pair]
            seqs = self._pair_to_seqs.get(pair)
            if seqs:
                seqs.discard(seq)
