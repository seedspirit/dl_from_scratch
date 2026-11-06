"""Byte-level BPE 토크나이저.

- `train_bpe`: 텍스트 파일에서 병합 규칙(merge rules)을 학습한다.
- `BPETokenizer`: 학습된 병합 규칙으로 텍스트를 인코딩/디코딩한다.
"""

import math
import os
import pickle
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from multiprocessing import Pool
from pathlib import Path
from typing import BinaryIO, Self

import numpy as np
import regex as re
from tqdm import tqdm

type Pair = tuple[int, int]
type TokenIds = tuple[int, ...]
type MergeRules = dict[Pair, int]

BYTE_VOCAB_SIZE = 256
DEFAULT_END_TOKEN = "<|endoftext|>"

# GPT-2와 같은 사전 토큰화 규칙 (축약형, 단어, 숫자, 기호, 공백을 각각 분리)
PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# encode_file 결과를 저장하는 dtype. vocab_size가 이 범위를 넘으면 안 된다.
TOKEN_DTYPE = np.uint16
MAX_VOCAB_SIZE = int(np.iinfo(TOKEN_DTYPE).max) + 1

_BOUNDARY_SEARCH_BUFFER_SIZE = 4096


def pretokenize(text: str) -> Iterator[str]:
    for match in PRETOKEN_PATTERN.finditer(text):
        yield match.group()


def count_pairs(ids: Sequence[int]) -> Counter[Pair]:
    return Counter(pairwise(ids))


def merge(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """ids에서 pair가 나타나는 위치를 모두 new_id 하나로 치환한다."""
    merged = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


# --- 파일 청크 분할 ---


@dataclass(frozen=True)
class FileChunk:
    path: str
    start: int
    end: int

    def read_text(self) -> str:
        with open(self.path, "rb") as f:
            f.seek(self.start)
            return f.read(self.end - self.start).decode("utf-8", errors="ignore")


def find_chunk_boundaries(file_path: str, num_chunks: int, end_token: str) -> list[int]:
    """파일을 약 num_chunks개로 나누되, 각 경계를 end_token의 시작 위치에 맞춘다.

    0과 파일 크기를 포함한 정렬된 바이트 오프셋 목록을 반환한다.
    겹치는 경계는 합쳐지므로 실제 청크 수는 num_chunks보다 적을 수 있다.
    """
    token_bytes = end_token.encode("utf-8")
    file_size = os.path.getsize(file_path)
    chunk_size = file_size // num_chunks
    boundaries = [i * chunk_size for i in range(num_chunks)] + [file_size]

    with open(file_path, "rb") as f:
        for i in range(1, num_chunks):
            boundaries[i] = _find_next(f, token_bytes, start=boundaries[i], default=file_size)
    return sorted(set(boundaries))


def _find_next(f: BinaryIO, needle: bytes, start: int, default: int) -> int:
    """start 이후 needle이 처음 나타나는 오프셋을 반환한다. 없으면 default."""
    position = start
    while True:
        f.seek(position)
        buffer = f.read(_BOUNDARY_SEARCH_BUFFER_SIZE)
        found = buffer.find(needle)
        if found != -1:
            return position + found
        if len(buffer) < _BOUNDARY_SEARCH_BUFFER_SIZE:
            return default  # 파일 끝까지 읽음
        # 버퍼 경계에 걸친 토큰을 놓치지 않도록 len(needle) - 1 바이트를 겹쳐 읽는다
        position += len(buffer) - len(needle) + 1


def split_into_chunks(file_path: str, num_chunks: int, end_token: str) -> list[FileChunk]:
    boundaries = find_chunk_boundaries(file_path, num_chunks, end_token)
    return [FileChunk(file_path, start, end) for start, end in pairwise(boundaries)]


# --- BPE 학습 ---


def _count_pretokens(chunk: FileChunk, end_token: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    # end_token은 문서 구분자이므로 그 경계를 넘어 병합되지 않도록 먼저 분리한다
    for text in chunk.read_text().split(end_token):
        counts.update(pretokenize(text))
    return counts


def count_pretokens_in_file(
    file_path: str, end_token: str, num_processes: int, num_chunks: int
) -> Counter[str]:
    chunks = split_into_chunks(file_path, num_chunks, end_token)
    total: Counter[str] = Counter()
    with Pool(processes=num_processes) as pool:
        chunk_counts = pool.imap(partial(_count_pretokens, end_token=end_token), chunks)
        for counts in tqdm(chunk_counts, total=len(chunks), desc="Pretokenizing"):
            total.update(counts)
    return total


class _MergeState:
    """BPE 학습 중 병합할 때마다 함께 갱신되는 상태.

    - seq_counts: 사전 토큰(ID 시퀀스)별 등장 횟수
    - pair_counts: 인접 페어별 전체 등장 횟수 (seq_counts로 가중)
    - pair_to_seqs: 페어 -> 그 페어를 포함한 시퀀스들.
      병합 시 영향을 받는 시퀀스만 갱신하기 위한 역인덱스.
    """

    def __init__(self, seq_counts: dict[TokenIds, int]) -> None:
        self.seq_counts: dict[TokenIds, int] = {}
        self.pair_counts: Counter[Pair] = Counter()
        self.pair_to_seqs: defaultdict[Pair, set[TokenIds]] = defaultdict(set)
        for seq, count in seq_counts.items():
            self._add_sequence(seq, count)

    def most_frequent_pair(self) -> Pair | None:
        if not self.pair_counts:
            return None
        # 빈도가 같으면 ID가 큰 페어를 골라 결과를 결정적으로 만든다
        return max(self.pair_counts, key=lambda pair: (self.pair_counts[pair], pair))

    def apply_merge(self, pair: Pair, new_id: int) -> None:
        for seq in self.pair_to_seqs.pop(pair, set()):
            count = self.seq_counts[seq]
            self._remove_sequence(seq, count)
            self._add_sequence(tuple(merge(seq, pair, new_id)), count)

    def _add_sequence(self, seq: TokenIds, count: int) -> None:
        self.seq_counts[seq] = count
        for pair, occurrences in count_pairs(seq).items():
            self.pair_counts[pair] += occurrences * count
            self.pair_to_seqs[pair].add(seq)

    def _remove_sequence(self, seq: TokenIds, count: int) -> None:
        del self.seq_counts[seq]
        for pair, occurrences in count_pairs(seq).items():
            self.pair_counts[pair] -= occurrences * count
            if self.pair_counts[pair] <= 0:
                del self.pair_counts[pair]
            if seqs := self.pair_to_seqs.get(pair):
                seqs.discard(seq)


def train_bpe(
    file_path: str,
    vocab_size: int,
    end_token: str = DEFAULT_END_TOKEN,
    num_processes: int = 8,
    num_chunks: int = 8,
) -> MergeRules:
    """병합 규칙을 학습한다. vocab_size는 바이트 256개와 end_token 1개를 포함한 크기다."""
    num_merges = vocab_size - BYTE_VOCAB_SIZE - 1
    if num_merges < 0:
        raise ValueError(f"vocab_size must be at least {BYTE_VOCAB_SIZE + 1}, got {vocab_size}")

    pretoken_counts = count_pretokens_in_file(file_path, end_token, num_processes, num_chunks)
    state = _MergeState(
        {tuple(pretoken.encode("utf-8")): count for pretoken, count in pretoken_counts.items()}
    )

    merge_rules: MergeRules = {}
    for step in tqdm(range(num_merges), desc="Training BPE"):
        best_pair = state.most_frequent_pair()
        if best_pair is None:
            break  # 더 병합할 페어가 없음
        new_id = BYTE_VOCAB_SIZE + step
        state.apply_merge(best_pair, new_id)
        merge_rules[best_pair] = new_id
    return merge_rules


# --- 인코딩/디코딩 ---


@dataclass(frozen=True)
class _EncodeTask:
    chunk: FileChunk
    output_path: Path


class BPETokenizer:
    def __init__(self, merge_rules: MergeRules, end_token: str = DEFAULT_END_TOKEN) -> None:
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = BYTE_VOCAB_SIZE + len(merge_rules)

        self.id_to_bytes = {i: bytes([i]) for i in range(BYTE_VOCAB_SIZE)}
        # 병합 규칙은 학습 순서대로 저장되어 있으므로 id1, id2는 항상 먼저 정의되어 있다
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        self.id_to_bytes[self.end_token_id] = end_token.encode("utf-8")

        self.vocab_size = len(self.id_to_bytes)

    @classmethod
    def load_from(cls, filepath: str) -> Self:
        with open(filepath, "rb") as f:
            merge_rules = pickle.load(f)
        return cls(merge_rules)

    def encode(self, input_text: str, show_progress: bool = False) -> list[int]:
        # 캡처 그룹으로 분리해 end_token 자체도 결과 목록에 남긴다
        segments = re.split(f"({re.escape(self.end_token)})", input_text)
        all_ids: list[int] = []
        for segment in tqdm(segments) if show_progress else segments:
            if segment == self.end_token:
                all_ids.append(self.end_token_id)
                continue
            for pretoken in pretokenize(segment):
                all_ids.extend(self._encode_pretoken(pretoken))
        return all_ids

    def decode(self, ids: Sequence[int]) -> str:
        text_bytes = b"".join(self.id_to_bytes[i] for i in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def encode_file(
        self,
        file_path: str,
        output_file: str,
        num_processes: int = 4,
        num_chunks: int = 64,
        cache_dir: str | None = None,
    ) -> int:
        """파일을 병렬로 인코딩해 TOKEN_DTYPE 배열로 output_file에 저장하고 토큰 수를 반환한다.

        청크별 중간 결과는 cache_dir(없으면 시스템 기본 위치) 아래의 임시 디렉터리에
        저장되며, 완료 후 그 임시 디렉터리만 삭제된다.
        """
        if self.vocab_size > MAX_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size {self.vocab_size} does not fit in {np.dtype(TOKEN_DTYPE).name}"
            )
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

        chunks = split_into_chunks(file_path, num_chunks, self.end_token)
        with tempfile.TemporaryDirectory(dir=cache_dir) as tmp_dir:
            tasks = [
                _EncodeTask(chunk, Path(tmp_dir) / f"chunk_{i:05d}.bin")
                for i, chunk in enumerate(chunks)
            ]
            with Pool(processes=num_processes) as pool:
                token_counts = list(
                    tqdm(
                        pool.imap(self._encode_chunk_to_file, tasks),
                        total=len(tasks),
                        desc="Encoding chunks",
                    )
                )
            total_tokens = sum(token_counts)
            self._concat_chunk_files(tasks, token_counts, output_file)
        return total_tokens

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """학습된 순서(= 우선순위)대로 병합 규칙을 적용한다."""
        ids = list(pretoken.encode("utf-8"))
        while len(ids) > 1:
            best_pair = min(pairwise(ids), key=self._merge_priority)
            if best_pair not in self.merge_rules:
                break  # 적용할 수 있는 병합 규칙이 없음
            ids = merge(ids, best_pair, self.merge_rules[best_pair])
        return ids

    def _merge_priority(self, pair: Pair) -> float:
        # 먼저 학습된 규칙일수록 ID가 작으며, 규칙이 없는 페어는 최저 우선순위
        return self.merge_rules.get(pair, math.inf)

    def _encode_chunk_to_file(self, task: _EncodeTask) -> int:
        ids = np.array(self.encode(task.chunk.read_text()), dtype=TOKEN_DTYPE)
        ids.tofile(task.output_path)
        return len(ids)

    @staticmethod
    def _concat_chunk_files(
        tasks: Sequence[_EncodeTask], token_counts: Sequence[int], output_file: str
    ) -> None:
        output = np.memmap(output_file, dtype=TOKEN_DTYPE, mode="w+", shape=(sum(token_counts),))
        offset = 0
        for task, count in zip(tasks, token_counts, strict=True):
            output[offset : offset + count] = np.fromfile(task.output_path, dtype=TOKEN_DTYPE)
            offset += count
        output.flush()
        del output
