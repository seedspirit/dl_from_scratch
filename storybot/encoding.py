"""파일 분할부터 병렬 인코딩, 토큰 출력까지 처리한다."""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from storybot.args import EncodingArgs, TokenizerArgs
from storybot.bpe import MergeRules, PreTokenizer
from storybot.corpus import FileChunk, split_file_chunks
from storybot.execution import parallel_map
from storybot.storage import AtomicOutput
from storybot.tokenizer import BPETokenizer


@dataclass(frozen=True)
class EncodingTask:
    chunk: FileChunk
    output_path: Path


@dataclass(frozen=True)
class EncodingInput:
    rules: MergeRules
    args: TokenizerArgs
    tasks: tuple[EncodingTask, ...]
    buffer_tokens: int


@dataclass
class FileEncoder:
    pretokenizer: PreTokenizer

    def encode_file(self, tokenizer: BPETokenizer, args: EncodingArgs) -> int:
        chunks = split_file_chunks(
            path=args.source,
            end_token=tokenizer.end_token,
            num_chunks=args.execution.num_chunks,
            read_buffer_bytes=args.execution.read_buffer_bytes,
        )
        if args.temporary_dir is not None:
            args.temporary_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.temporary_dir) as directory:
            tasks: list[EncodingTask] = []
            for index, chunk in enumerate(chunks):
                output_path = Path(directory) / f"chunk_{index:05d}.bin"
                tasks.append(EncodingTask(chunk, output_path))

            jobs: list[EncodingInput] = []
            batch_count = min(args.execution.num_processes, len(tasks))
            for index in range(batch_count):
                batch = tuple(tasks[index::batch_count])
                data = EncodingInput(
                    rules=tokenizer.rules,
                    args=tokenizer.args,
                    tasks=batch,
                    buffer_tokens=args.buffer_tokens,
                )
                jobs.append(data)

            total_tokens = 0
            with parallel_map(self.encode_batch, jobs, args.execution) as results:
                for token_count in results:
                    total_tokens += token_count

            with AtomicOutput(args.destination) as temporary:
                with temporary.open("wb") as output:
                    for task in tasks:
                        with task.output_path.open("rb") as source:
                            shutil.copyfileobj(source, output)
        return total_tokens

    def encode_batch(self, data: EncodingInput) -> int:
        # 프로세스마다 캐시를 구성하고 묶음 안의 모든 청크에서 재사용한다.
        tokenizer = BPETokenizer(data.rules, data.args, self.pretokenizer)
        total_tokens = 0
        for task in data.tasks:
            buffer: list[int] = []
            with task.output_path.open("wb") as output:
                for text, has_separator in task.chunk.iter_documents():
                    for ids in tokenizer.encode_pretokens(text):
                        buffer.extend(ids)
                        if len(buffer) >= data.buffer_tokens:
                            tokens = np.asarray(buffer, dtype=np.uint16)
                            tokens.tofile(output)
                            total_tokens += len(buffer)
                            buffer.clear()
                    if has_separator:
                        buffer.append(tokenizer.end_token_id)
                        if len(buffer) >= data.buffer_tokens:
                            tokens = np.asarray(buffer, dtype=np.uint16)
                            tokens.tofile(output)
                            total_tokens += len(buffer)
                            buffer.clear()
                if buffer:
                    tokens = np.asarray(buffer, dtype=np.uint16)
                    tokens.tofile(output)
                    total_tokens += len(buffer)
        return total_tokens
