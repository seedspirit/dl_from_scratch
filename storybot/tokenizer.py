"""BPE 객체를 조합해 텍스트와 토큰 ID를 변환한다."""

from collections.abc import Callable, Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Self

import regex
from rich.progress import track

from storybot.args import TextEncodingArgs, TokenizerArgs, TokenizerLoadArgs
from storybot.bpe import MergeRules, PreTokenizer, Vocabulary
from storybot.storage import TokenizerFile


class BPETokenizer:
    """검증된 규칙과 설정으로 구성하는 텍스트 인코더."""

    # 입력과 재사용하는 의존성
    rules: MergeRules
    args: TokenizerArgs
    pretokenizer: PreTokenizer
    vocabulary: Vocabulary

    # 생성 시 준비하는 인코딩 캐시와 구분자 패턴
    _cached_encode: Callable[[str], tuple[int, ...]]
    _separator: regex.Pattern

    def __init__(self, rules: MergeRules, args: TokenizerArgs, pretokenizer: PreTokenizer) -> None:
        self.rules = rules
        self.args = args
        self.pretokenizer = pretokenizer
        self.vocabulary = Vocabulary(rules, args.tokenization.end_token)
        self._cached_encode = lru_cache(maxsize=args.cache_size)(self._encode_pretoken)
        self._separator = regex.compile(f"({regex.escape(args.tokenization.end_token)})")

    @property
    def end_token(self) -> str:
        return self.args.tokenization.end_token

    @property
    def end_token_id(self) -> int:
        return self.vocabulary.end_token_id

    @property
    def vocab_size(self) -> int:
        return self.vocabulary.size

    @classmethod
    def load_from(cls, args: TokenizerLoadArgs, pretokenizer: PreTokenizer) -> Self:
        model = TokenizerFile.load(args.path)
        return cls(
            model.rules,
            TokenizerArgs(
                tokenization=model.tokenization,
                cache_size=args.cache_size,
            ),
            pretokenizer,
        )

    def save_to(self, filepath: Path) -> None:
        TokenizerFile.from_rules(self.rules, self.args.tokenization).save(filepath)

    def encode(self, args: TextEncodingArgs) -> list[int]:
        segments = self._separator.split(args.text)
        all_ids: list[int] = []
        progress = track(segments, description="텍스트 인코딩", disable=not args.show_progress)
        for segment in progress:
            if segment == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                for ids in self.encode_pretokens(segment):
                    all_ids.extend(ids)
        return all_ids

    def encode_pretokens(self, text: str) -> Iterator[tuple[int, ...]]:
        """특수 토큰이 없는 문서를 pretoken 단위로 인코딩한다."""
        for pretoken in self.pretokenizer.split(text, self.args.tokenization.pattern):
            yield self._cached_encode(pretoken)

    def decode(self, ids: Sequence[int]) -> str:
        return self.vocabulary.decode(ids)

    def _encode_pretoken(self, text: str) -> tuple[int, ...]:
        ids = tuple(text.encode("utf-8"))
        while True:
            rule = self.rules.next_match(ids)
            if rule is None:
                break
            ids = rule.apply(ids)
        return ids
