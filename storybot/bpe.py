"""BPE 규칙, 사전 토큰화, 어휘를 표현하는 객체."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import pairwise
from typing import Annotated, Self

import regex
from pydantic import Field, model_validator

from storybot.args import InputModel

BYTE_VOCAB_SIZE = 256


class PreTokenizer:
    """호출별 패턴으로 사전 토큰화한다. 컴파일 결과는 regex의 패턴 캐시를 사용한다."""

    def split(self, text: str, pattern: str) -> Iterator[str]:
        for match in regex.finditer(pattern, text):
            yield match.group()


@dataclass(frozen=True)
class MergeRule:
    left: int
    right: int
    token_id: int

    @property
    def pair(self) -> tuple[int, int]:
        return self.left, self.right

    def apply(self, ids: Sequence[int]) -> tuple[int, ...]:
        merged: list[int] = []
        index = 0
        while index < len(ids):
            if index + 1 < len(ids) and ids[index] == self.left and ids[index + 1] == self.right:
                merged.append(self.token_id)
                index += 2
            else:
                merged.append(ids[index])
                index += 1
        return tuple(merged)


class MergeRules(InputModel):
    """학습 순서의 규칙과 내부 조회 인덱스를 함께 소유한다."""

    entries: Annotated[tuple[MergeRule, ...], Field(max_length=65536 - 257)]

    @cached_property
    def _by_pair(self) -> dict[tuple[int, int], MergeRule]:
        return {rule.pair: rule for rule in self.entries}

    @model_validator(mode="after")
    def ordered_rules(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for expected_id, rule in enumerate(self.entries, start=BYTE_VOCAB_SIZE):
            if rule.token_id != expected_id or not (
                0 <= rule.left < expected_id and 0 <= rule.right < expected_id
            ):
                raise ValueError("Merge rules must reference earlier IDs in training order")
            if rule.pair in seen:
                raise ValueError("Duplicate merge pair")
            seen.add(rule.pair)
        return self

    @classmethod
    def from_entries(cls, entries: Iterable[Sequence[int]]) -> "MergeRules":
        rules = []
        for left, right, token_id in entries:
            rules.append(MergeRule(left, right, token_id))
        return cls(entries=tuple(rules))

    def next_match(self, ids: Sequence[int]) -> MergeRule | None:
        best = None
        for pair in pairwise(ids):
            rule = self._by_pair.get(pair)
            if rule is not None and (best is None or rule.token_id < best.token_id):
                best = rule
        return best


class Vocabulary:
    """연속된 토큰 ID를 바이트 배열의 인덱스로 표현한다."""

    end_token_id: int
    _tokens: tuple[bytes, ...]

    def __init__(self, rules: MergeRules, end_token: str) -> None:
        tokens = [bytes([token_id]) for token_id in range(BYTE_VOCAB_SIZE)]
        for rule in rules.entries:
            tokens.append(tokens[rule.left] + tokens[rule.right])
        self.end_token_id = len(tokens)
        tokens.append(end_token.encode("utf-8"))
        self._tokens = tuple(tokens)

    @property
    def size(self) -> int:
        return len(self._tokens)

    def decode(self, ids: Sequence[int]) -> str:
        token_bytes = [self._tokens[token_id] for token_id in ids]
        text_bytes = b"".join(token_bytes)
        return text_bytes.decode("utf-8", errors="replace")
