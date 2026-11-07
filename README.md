# Deep Learning from Scratch

## Layout

- `base/`: shared model code (GPT, attention, FFN, norms, tokenizer, generation)
- `pretrain/`, `sft/`, `grpo/`: one package per training stage, each with a
  `dataset.py` and a `train.py` entry point

## Running

Run every command from the repository root so the package imports and the
relative `codebot/` data paths resolve:

```sh
uv run python -m pretrain.train   # pretraining
uv run python -m sft.train        # supervised fine-tuning
uv run python -m grpo.train       # GRPO
uv run python -m base.generate    # sample from the pretrained model
uv run python chat.py             # chat with the SFT model
```

## Storybot tokenizer

```python
from storybot.tokenizer import BPETokenizer, train_bpe

rules = train_bpe(
    "storybot/tiny_stories_train.txt", vocab_size=8192,
    num_processes=4, num_chunks=64, cache_dir=".cache/storybot",
)
tokenizer = BPETokenizer(rules)
tokenizer.save_to(".cache/storybot/tokenizer.json")
tokenizer.encode_file(
    "storybot/tiny_stories_train.txt", ".cache/storybot/train.bin",
    num_processes=4, num_chunks=64,
)
```

Training caches pretoken frequencies when `cache_dir` is supplied. The cache
key includes the resolved input path, size, modification time, pretoken pattern,
and document separator. Frequencies are stored as versioned gzip-compressed JSON
(`.json.gz`) with positive integer counts.
If file contents change while size and modification time are preserved, clear
the cache manually. Input files must remain unchanged during processing.

File encoding streams documents into buffered uint16 output. Each worker owns
a bounded 32,768-entry pretoken LRU cache and initializes the tokenizer once.
Memory still depends on the longest document/pretoken and each worker's cache;
a file without document separators cannot be split across workers safely.
`encode_file(cache_dir=...)` controls temporary-file placement, not persistent
encoding reuse. Outputs replace the destination only after successful encoding.
UTF-8 decoding errors are reported rather than silently discarded.

Saved tokenizers use UTF-8 JSON (`.json`) containing a format version, pretoken
pattern, document separator, and ordered `[left, right, new_id]` merge entries.
Loading validates the version, pattern, token IDs, and merge order. Models are
saved atomically. Merge rules are read-only after construction.

Pickle tokenizer files are no longer supported; retrain and save as JSON. Old
pickle frequency caches are ignored and rebuilt in the new format. Encoded
training data continues to use the existing uint16 binary format.

Run regression tests with `uv run python -m unittest discover -s tests`.
