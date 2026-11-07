"""문서 구분자를 보존하는 파일 분할과 스트리밍."""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class FileChunk:
    path: str
    start: int
    end: int
    end_token: str
    read_buffer_bytes: int

    def iter_documents(self) -> Iterator[tuple[str, bool]]:
        """문서 경계를 보존하고 구분자의 존재 여부와 함께 반환한다."""
        needle = self.end_token.encode("utf-8")
        pending = bytearray()
        with open(self.path, "rb") as f:
            f.seek(self.start)
            remaining = self.end - self.start
            while remaining:
                block = f.read(min(self.read_buffer_bytes, remaining))
                if not block:
                    raise OSError("Input file changed while reading")
                remaining -= len(block)
                pending.extend(block)
                start = 0
                while True:
                    index = pending.find(needle, start)
                    if index == -1:
                        break
                    yield pending[start:index].decode("utf-8"), True
                    start = index + len(needle)
                if start:
                    del pending[:start]
            if pending:
                yield pending.decode("utf-8"), False


def split_file_chunks(
    path: Path,
    *,
    end_token: str,
    num_chunks: int,
    read_buffer_bytes: int,
) -> list[FileChunk]:
    file_size = path.stat().st_size
    chunk_size = file_size // num_chunks
    boundaries = [0, file_size]
    needle = end_token.encode("utf-8")
    with path.open("rb") as source:
        for index in range(1, num_chunks):
            boundary = _find_document_boundary(
                source, needle, index * chunk_size, file_size, read_buffer_bytes
            )
            boundaries.append(boundary)

    chunks: list[FileChunk] = []
    sorted_boundaries = sorted(set(boundaries))
    for start, end in pairwise(sorted_boundaries):
        chunk = FileChunk(str(path), start, end, end_token, read_buffer_bytes)
        chunks.append(chunk)
    return chunks


def _find_document_boundary(
    source: BinaryIO, needle: bytes, start: int, default: int, read_buffer_bytes: int
) -> int:
    buffer_size = max(read_buffer_bytes, len(needle) * 2)
    position = start
    while True:
        source.seek(position)
        buffer = source.read(buffer_size)
        found = buffer.find(needle)
        if found != -1:
            return position + found
        if len(buffer) < buffer_size:
            return default
        position += len(buffer) - len(needle) + 1
