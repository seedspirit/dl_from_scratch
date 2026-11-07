"""토크나이저 JSON과 빈도 캐시 저장. 파일 교체는 한 곳에서 처리한다."""

import gzip
import hashlib
import os
import tempfile
from collections import Counter
from pathlib import Path
from types import TracebackType
from typing import Annotated, ClassVar, Literal

from pydantic import Field, NonNegativeInt, PositiveInt

from storybot.args import FrequencyCacheArgs, InputModel, NonEmptyText, TokenizationArgs
from storybot.bpe import MergeRules


class AtomicOutput:
    """임시 출력 파일의 생성, 성공 시 교체, 실패 시 정리를 소유한다."""

    destination: Path
    temporary: Path  # __enter__에서 생성하며 with 블록 동안 사용한다.

    def __init__(self, destination: Path) -> None:
        self.destination = destination

    def __enter__(self) -> Path:
        with tempfile.NamedTemporaryFile(dir=self.destination.parent, delete=False) as file:
            self.temporary = Path(file.name)
        return self.temporary

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if error_type is None:
                os.replace(self.temporary, self.destination)
        finally:
            self.temporary.unlink(missing_ok=True)


class TokenizerFile(InputModel):
    end_token: NonEmptyText
    pretoken_pattern: NonEmptyText
    merges: Annotated[
        list[tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]], Field(max_length=65536 - 257)
    ]
    version: Literal[1]

    @classmethod
    def from_rules(cls, rules: MergeRules, args: TokenizationArgs) -> "TokenizerFile":
        return cls(
            end_token=args.end_token,
            pretoken_pattern=args.pattern,
            merges=[(rule.left, rule.right, rule.token_id) for rule in rules.entries],
            version=1,
        )

    @property
    def tokenization(self) -> TokenizationArgs:
        return TokenizationArgs(end_token=self.end_token, pattern=self.pretoken_pattern)

    @property
    def rules(self) -> MergeRules:
        return MergeRules.from_entries(self.merges)

    @classmethod
    def load(cls, path: Path) -> "TokenizerFile":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        with AtomicOutput(path) as temporary:
            with temporary.open("w", encoding="utf-8") as file:
                file.write(self.model_dump_json(indent=2))
                file.write("\n")


class _FrequencyFile(InputModel):
    # JSON의 문자열 키 객체는 저장 경계에서만 사용한다.
    counts: dict[str, PositiveInt]
    version: Literal[2]


class FrequencyCache:
    VERSION: ClassVar[int] = 2

    def load(self, args: FrequencyCacheArgs) -> Counter[str] | None:
        path = self._path(args)
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as file:
            payload = _FrequencyFile.model_validate_json(file.read())
        return Counter(payload.counts)

    def save(self, args: FrequencyCacheArgs, counts: Counter[str]) -> None:
        path = self._path(args)
        path.parent.mkdir(parents=True, exist_ok=True)
        with AtomicOutput(path) as temporary:
            with gzip.open(temporary, "wt", encoding="utf-8") as file:
                file.write(_FrequencyFile(counts=dict(counts), version=2).model_dump_json())

    def _path(self, args: FrequencyCacheArgs) -> Path:
        path = args.source.resolve()
        stat = path.stat()
        identity = repr(
            (
                self.VERSION,
                str(path),
                stat.st_size,
                stat.st_mtime_ns,
                args.tokenization.end_token,
                args.tokenization.pattern,
            )
        ).encode()
        return args.directory / (hashlib.sha256(identity).hexdigest() + ".json.gz")
