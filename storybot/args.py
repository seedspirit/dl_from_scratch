"""호출자가 명시하는 입력 계약. 실행 전에 Pydantic으로 검증한다."""

from pathlib import Path
from typing import Annotated

import regex
from pydantic import BaseModel, ConfigDict, Field, FilePath, field_validator

type PositiveCount = Annotated[int, Field(gt=0, strict=True)]
type CacheSize = Annotated[int, Field(ge=0, strict=True)]
type VocabularySize = Annotated[int, Field(ge=257, le=65536, strict=True)]
type NonEmptyText = Annotated[str, Field(min_length=1)]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TokenizationArgs(InputModel):
    end_token: NonEmptyText
    pattern: NonEmptyText

    @field_validator("pattern")
    @classmethod
    def valid_pattern(cls, value: str) -> str:
        try:
            regex.compile(value)
        except regex.error as error:
            raise ValueError(str(error)) from error
        return value


class TokenizerArgs(InputModel):
    tokenization: TokenizationArgs
    cache_size: CacheSize


class ExecutionArgs(InputModel):
    num_processes: PositiveCount
    num_chunks: PositiveCount
    read_buffer_bytes: PositiveCount
    show_progress: bool


class TrainingArgs(InputModel):
    source: FilePath
    vocab_size: VocabularySize
    tokenization: TokenizationArgs
    execution: ExecutionArgs
    cache_dir: Path | None


class EncodingArgs(InputModel):
    source: FilePath
    destination: Path
    execution: ExecutionArgs
    buffer_tokens: PositiveCount
    temporary_dir: Path | None


class TextEncodingArgs(InputModel):
    text: str
    show_progress: bool


class TokenizerLoadArgs(InputModel):
    path: FilePath
    cache_size: CacheSize


class FrequencyCacheArgs(InputModel):
    directory: Path
    source: FilePath
    tokenization: TokenizationArgs


class TokenizerTrainingArgs(InputModel):
    training: TrainingArgs
    cache_size: CacheSize
