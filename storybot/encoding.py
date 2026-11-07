"""파일 분할부터 병렬 인코딩, 토큰 출력까지 처리한다."""

import multiprocessing
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from multiprocessing import Manager, Pool
from pathlib import Path
from queue import Empty, Queue
from time import monotonic

import numpy as np

from storybot.args import EncodingArgs, TokenizerArgs
from storybot.bpe import MergeRules, PreTokenizer
from storybot.corpus import FileChunk, split_file_chunks
from storybot.display import byte_progress
from storybot.profiling import profile_time
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
    progress_queue: Queue[int]


@dataclass
class FileEncoder:
    pretokenizer: PreTokenizer

    @profile_time("FileEncoder.encode_file")
    def encode_file(self, tokenizer: BPETokenizer, args: EncodingArgs) -> int:
        chunks = split_file_chunks(
            path=args.source,
            end_token=tokenizer.end_token,
            num_chunks=args.execution.num_chunks,
            read_buffer_bytes=args.execution.read_buffer_bytes,
        )
        if args.temporary_dir is not None:
            args.temporary_dir.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory(dir=args.temporary_dir))
            manager = stack.enter_context(Manager())
            progress_queue = manager.Queue()
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
                    progress_queue=progress_queue,
                )
                jobs.append(data)

            total_tokens = 0
            with ExitStack() as encoding_stack:
                progress = encoding_stack.enter_context(
                    byte_progress(enabled=args.execution.show_progress)
                )
                encoding_task = progress.add_task(
                    "인코딩", total=sum(chunk.end - chunk.start for chunk in chunks)
                )
                if jobs:
                    pool = encoding_stack.enter_context(Pool(processes=batch_count))
                    results = pool.imap_unordered(self.encode_batch, jobs)
                    completed = 0
                    while completed < len(jobs):
                        while True:
                            try:
                                progress.advance(encoding_task, progress_queue.get_nowait())
                            except Empty:
                                break
                        try:
                            token_count = results.next(timeout=0.1)
                        except multiprocessing.TimeoutError:
                            continue
                        total_tokens += token_count
                        completed += 1
                    while True:
                        try:
                            progress.advance(encoding_task, progress_queue.get_nowait())
                        except Empty:
                            break

                    pool.close()
                    pool.join()

            output_bytes = total_tokens * np.dtype(np.uint16).itemsize
            with ExitStack() as merging_stack:
                progress = merging_stack.enter_context(
                    byte_progress(enabled=args.execution.show_progress)
                )
                merging_task = progress.add_task("출력 파일 병합", total=output_bytes)
                temporary = merging_stack.enter_context(AtomicOutput(args.destination))
                output = merging_stack.enter_context(temporary.open("wb"))
                for task in tasks:
                    with task.output_path.open("rb") as source:
                        while True:
                            block = source.read(1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                            progress.advance(merging_task, len(block))
        return total_tokens

    @profile_time("FileEncoder.encode_batch")
    def encode_batch(self, data: EncodingInput) -> int:
        # 프로세스마다 캐시를 구성하고 묶음 안의 모든 청크에서 재사용한다.
        tokenizer = BPETokenizer(data.rules, data.args, self.pretokenizer)
        total_tokens = 0
        pending_bytes = 0
        last_report = monotonic()
        separator_bytes = len(data.args.tokenization.end_token.encode("utf-8"))
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
                    pending_bytes += len(text.encode("utf-8"))
                    if has_separator:
                        pending_bytes += separator_bytes
                    now = monotonic()
                    if pending_bytes >= 4 * 1024 * 1024 or now - last_report >= 0.5:
                        data.progress_queue.put(pending_bytes)
                        pending_bytes = 0
                        last_report = now
                if buffer:
                    tokens = np.asarray(buffer, dtype=np.uint16)
                    tokens.tofile(output)
                    total_tokens += len(buffer)
        if pending_bytes:
            data.progress_queue.put(pending_bytes)
        return total_tokens
