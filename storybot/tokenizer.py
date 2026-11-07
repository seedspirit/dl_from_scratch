"""Byte-level BPE 토크나이저.

- `train_bpe`: 텍스트 파일에서 병합 규칙(merge rules)을 학습한다.
- `BPETokenizer`: 학습된 병합 규칙으로 텍스트를 인코딩/디코딩한다.
"""

import gzip
import hashlib
import heapq
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from itertools import pairwise
from multiprocessing import Pool
from pathlib import Path
from types import MappingProxyType
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
_READ_BUFFER_SIZE = 1024 * 1024
_CACHE_VERSION = 2
_MODEL_VERSION = 1


def _validate_options(end_token: str, num_processes: int, num_chunks: int) -> None:
    if not end_token:
        raise ValueError("end_token must not be empty")
    if num_processes < 1 or num_chunks < 1:
        raise ValueError("num_processes and num_chunks must be positive")


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
            return f.read(self.end - self.start).decode("utf-8")

    def iter_documents(self, end_token: str) -> Iterator[tuple[str, bool]]:
        """문서 경계를 보존하고 구분자의 존재 여부와 함께 반환한다."""
        needle = end_token.encode("utf-8")
        pending = bytearray()
        with open(self.path, "rb") as f:
            f.seek(self.start)
            remaining = self.end - self.start
            while remaining:
                block = f.read(min(_READ_BUFFER_SIZE, remaining))
                if not block:
                    raise OSError("Input file changed while reading")
                remaining -= len(block)
                pending.extend(block)
                start = 0
                while (index := pending.find(needle, start)) != -1:
                    yield pending[start:index].decode("utf-8"), True
                    start = index + len(needle)
                if start:
                    del pending[:start]
            if pending:
                yield pending.decode("utf-8"), False


def find_chunk_boundaries(file_path: str, num_chunks: int, end_token: str) -> list[int]:
    """파일을 약 num_chunks개로 나누되, 각 경계를 end_token의 시작 위치에 맞춘다.

    0과 파일 크기를 포함한 정렬된 바이트 오프셋 목록을 반환한다.
    겹치는 경계는 합쳐지므로 실제 청크 수는 num_chunks보다 적을 수 있다.
    """
    _validate_options(end_token, 1, num_chunks)
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
    buffer_size = max(_BOUNDARY_SEARCH_BUFFER_SIZE, len(needle) * 2)
    while True:
        f.seek(position)
        buffer = f.read(buffer_size)
        found = buffer.find(needle)
        if found != -1:
            return position + found
        if len(buffer) < buffer_size:
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
    for text, _ in chunk.iter_documents(end_token):
        counts.update(pretokenize(text))
    return counts


def count_pretokens_in_file(
    file_path: str,
    end_token: str,
    num_processes: int,
    num_chunks: int,
    cache_dir: str | None = None,
) -> Counter[str]:
    _validate_options(end_token, num_processes, num_chunks)
    cache_path = None
    if cache_dir is not None:
        path = Path(file_path).resolve()
        stat = path.stat()
        identity = repr(
            (
                _CACHE_VERSION,
                str(path),
                stat.st_size,
                stat.st_mtime_ns,
                end_token,
                PRETOKEN_PATTERN.pattern,
            )
        ).encode()
        cache_path = Path(cache_dir) / (hashlib.sha256(identity).hexdigest() + ".json.gz")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            with gzip.open(cache_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            if (
                not isinstance(data, dict)
                or type(data.get("version")) is not int
                or data["version"] != _CACHE_VERSION
                or not isinstance(data.get("counts"), dict)
            ):
                raise ValueError("Invalid pretoken cache format or version")
            counts = data["counts"]
            if any(type(count) is not int or count <= 0 for count in counts.values()):
                raise ValueError("Pretoken counts must be positive integers")
            return Counter(counts)
    chunks = split_into_chunks(file_path, num_chunks, end_token)
    total: Counter[str] = Counter()
    with Pool(processes=num_processes) as pool:
        chunk_counts = pool.imap_unordered(partial(_count_pretokens, end_token=end_token), chunks)
        for counts in tqdm(chunk_counts, total=len(chunks), desc="Pretokenizing"):
            total.update(counts)
    if cache_path is not None:
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, delete=False) as f:
            temporary = Path(f.name)
            try:
                f.close()
                with gzip.open(temporary, "wt", encoding="utf-8") as compressed:
                    json.dump(
                        {"version": _CACHE_VERSION, "counts": total},
                        compressed,
                        ensure_ascii=False,
                    )
                os.replace(temporary, cache_path)
            finally:
                temporary.unlink(missing_ok=True)
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
        self._changed_pairs: set[Pair] = set()
        for seq, count in seq_counts.items():
            self._add_sequence(seq, count)
        self._rebuild_heap()
        self._changed_pairs.clear()

    def _rebuild_heap(self) -> None:
        self._heap = [(-count, -pair[0], -pair[1]) for pair, count in self.pair_counts.items()]
        heapq.heapify(self._heap)

    def most_frequent_pair(self) -> Pair | None:
        while self._heap:
            negative_count, first, second = self._heap[0]
            pair = (-first, -second)
            if self.pair_counts.get(pair, 0) == -negative_count:
                return pair
            heapq.heappop(self._heap)
        return None

    def apply_merge(self, pair: Pair, new_id: int) -> None:
        for seq in self.pair_to_seqs.pop(pair, set()):
            count = self.seq_counts[seq]
            self._remove_sequence(seq, count)
            self._add_sequence(tuple(merge(seq, pair, new_id)), count)
        for changed in self._changed_pairs:
            if frequency := self.pair_counts.get(changed, 0):
                heapq.heappush(self._heap, (-frequency, -changed[0], -changed[1]))
        self._changed_pairs.clear()
        if len(self._heap) > max(1024, 4 * len(self.pair_counts)):
            self._rebuild_heap()

    def _add_sequence(self, seq: TokenIds, count: int) -> None:
        self.seq_counts[seq] = count
        for pair, occurrences in count_pairs(seq).items():
            self._changed_pairs.add(pair)
            self.pair_counts[pair] += occurrences * count
            self.pair_to_seqs[pair].add(seq)

    def _remove_sequence(self, seq: TokenIds, count: int) -> None:
        del self.seq_counts[seq]
        for pair, occurrences in count_pairs(seq).items():
            self._changed_pairs.add(pair)
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
    num_chunks: int = 64,
    cache_dir: str | None = None,
) -> MergeRules:
    """병합 규칙을 학습한다. vocab_size는 바이트 256개와 end_token 1개를 포함한 크기다."""
    num_merges = vocab_size - BYTE_VOCAB_SIZE - 1
    if num_merges < 0:
        raise ValueError(f"vocab_size must be at least {BYTE_VOCAB_SIZE + 1}, got {vocab_size}")

    pretoken_counts = count_pretokens_in_file(
        file_path, end_token, num_processes, num_chunks, cache_dir
    )
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
        _validate_options(end_token, 1, 1)
        self.merge_rules = MappingProxyType(dict(merge_rules))
        self._cached_encode = lru_cache(maxsize=32768)(self._encode_pretoken)
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
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data["version"] != _MODEL_VERSION
        ):
            raise ValueError("Invalid tokenizer format or unsupported version")
        if data.get("pretoken_pattern") != PRETOKEN_PATTERN.pattern:
            raise ValueError("Tokenizer pretoken pattern does not match this implementation")
        end_token = data.get("end_token")
        if not isinstance(end_token, str) or not end_token:
            raise ValueError("Tokenizer end_token must be a nonempty string")
        merges = data.get("merges")
        if not isinstance(merges, list):
            raise ValueError("Tokenizer merges must be a list")
        merge_rules: MergeRules = {}
        for expected_id, entry in enumerate(merges, start=BYTE_VOCAB_SIZE):
            if (
                not isinstance(entry, list)
                or len(entry) != 3
                or any(type(value) is not int for value in entry)
            ):
                raise ValueError("Each merge must contain three integer token IDs")
            left, right, new_id = entry
            if new_id != expected_id or not (0 <= left < new_id and 0 <= right < new_id):
                raise ValueError("Merge IDs must be sequential and reference earlier tokens")
            if (left, right) in merge_rules:
                raise ValueError("Duplicate merge pair")
            merge_rules[left, right] = new_id
        return cls(merge_rules, end_token)

    def save_to(self, filepath: str) -> None:
        destination = Path(filepath).resolve()
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
            temporary = Path(f.name)
        try:
            with temporary.open("w", encoding="utf-8") as output:
                json.dump(
                    {
                        "version": _MODEL_VERSION,
                        "pretoken_pattern": PRETOKEN_PATTERN.pattern,
                        "end_token": self.end_token,
                        "merges": [
                            [left, right, new_id]
                            for (left, right), new_id in self.merge_rules.items()
                        ],
                    },
                    output,
                    ensure_ascii=False,
                    indent=2,
                )
                output.write("\n")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def encode(self, input_text: str, show_progress: bool = False) -> list[int]:
        # 캡처 그룹으로 분리해 end_token 자체도 결과 목록에 남긴다
        segments = re.split(f"({re.escape(self.end_token)})", input_text)
        all_ids: list[int] = []
        for segment in tqdm(segments) if show_progress else segments:
            if segment == self.end_token:
                all_ids.append(self.end_token_id)
                continue
            for pretoken in pretokenize(segment):
                all_ids.extend(self._cached_encode(pretoken))
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
        _validate_options(self.end_token, num_processes, num_chunks)
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
            with Pool(
                processes=num_processes,
                initializer=_initialize_encoder,
                initargs=(dict(self.merge_rules), self.end_token),
            ) as pool:
                token_counts = list(
                    tqdm(
                        pool.imap(_encode_chunk, tasks),
                        total=len(tasks),
                        desc="Encoding chunks",
                    )
                )
            total_tokens = sum(token_counts)
            destination = Path(output_file).resolve()
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
                temporary = Path(f.name)
            try:
                self._concat_chunk_files(tasks, token_counts, str(temporary))
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return total_tokens

    def _encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        """학습된 순서(= 우선순위)대로 병합 규칙을 적용한다."""
        ids = list(pretoken.encode("utf-8"))
        while len(ids) > 1:
            best_pair = min(pairwise(ids), key=self._merge_priority)
            if best_pair not in self.merge_rules:
                break  # 적용할 수 있는 병합 규칙이 없음
            ids = merge(ids, best_pair, self.merge_rules[best_pair])
        return tuple(ids)

    def _merge_priority(self, pair: Pair) -> float:
        # 먼저 학습된 규칙일수록 ID가 작으며, 규칙이 없는 페어는 최저 우선순위
        return self.merge_rules.get(pair, math.inf)

    def _encode_chunk_to_file(self, task: _EncodeTask) -> int:
        count = 0
        buffer: list[int] = []
        with task.output_path.open("wb") as output:
            for text, has_separator in task.chunk.iter_documents(self.end_token):
                for pretoken in pretokenize(text):
                    buffer.extend(self._cached_encode(pretoken))
                    if len(buffer) >= 65536:
                        np.asarray(buffer, dtype=TOKEN_DTYPE).tofile(output)
                        count += len(buffer)
                        buffer.clear()
                if has_separator:
                    buffer.append(self.end_token_id)
            if buffer:
                np.asarray(buffer, dtype=TOKEN_DTYPE).tofile(output)
                count += len(buffer)
        return count

    @staticmethod
    def _concat_chunk_files(
        tasks: Sequence[_EncodeTask], token_counts: Sequence[int], output_file: str
    ) -> None:
        if not sum(token_counts):
            Path(output_file).write_bytes(b"")
            return
        output = np.memmap(output_file, dtype=TOKEN_DTYPE, mode="w+", shape=(sum(token_counts),))
        offset = 0
        for task, count in zip(tasks, token_counts, strict=True):
            output[offset : offset + count] = np.fromfile(task.output_path, dtype=TOKEN_DTYPE)
            offset += count
        output.flush()
        del output


_worker_tokenizer: BPETokenizer | None = None


def _initialize_encoder(merge_rules: MergeRules, end_token: str) -> None:
    global _worker_tokenizer
    _worker_tokenizer = BPETokenizer(merge_rules, end_token)


def _encode_chunk(task: _EncodeTask) -> int:
    if _worker_tokenizer is None:
        raise RuntimeError("Encoder worker was not initialized")
    return _worker_tokenizer._encode_chunk_to_file(task)
