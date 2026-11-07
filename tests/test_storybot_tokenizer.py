import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from storybot import tokenizer as bpe


class TokenizerTests(unittest.TestCase):
    def test_heap_matches_full_scan_including_ties(self):
        state = bpe._MergeState(
            {tuple(b"banana"): 9, tuple(b"bandana"): 4, tuple(b"aaaaaa"): 9, tuple(b"cabana"): 4}
        )
        for step in range(30):
            expected = max(
                state.pair_counts, key=lambda pair: (state.pair_counts[pair], pair), default=None
            )
            self.assertEqual(state.most_frequent_pair(), expected)
            if expected is None:
                break
            state.apply_merge(expected, 256 + step)

    def test_streamed_encoding_matches_whole_text(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.txt"
            output = Path(directory) / "output.bin"
            texts = [
                "",
                "hello 안녕 🌟",
                "<END>",
                "<END>hi<END><END>끝",
                " space\n\n\t words " * 100,
            ]
            for text in texts:
                source.write_text(text, encoding="utf-8")
                tokenizer = bpe.BPETokenizer({(104, 101): 256}, end_token="<END>")
                with patch.object(bpe, "_READ_BUFFER_SIZE", 3):
                    count = tokenizer.encode_file(str(source), str(output), 2, 5)
                expected = tokenizer.encode(text)
                self.assertEqual(np.fromfile(output, dtype=bpe.TOKEN_DTYPE).tolist(), expected)
                self.assertEqual(count, len(expected))
                self.assertEqual(tokenizer.decode(expected), text)

    def test_frequency_cache_and_custom_token_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.txt"
            source.write_text("hello hello<END>안녕 🌟", encoding="utf-8")
            cache = str(Path(directory) / "cache")
            first = bpe.count_pretokens_in_file(str(source), "<END>", 1, 3, cache)
            cache_file = next(Path(cache).glob("*.json.gz"))
            with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                self.assertEqual(json.load(f)["counts"], dict(first))
            with patch.object(bpe, "Pool", side_effect=AssertionError("cache missed")):
                self.assertEqual(
                    first, bpe.count_pretokens_in_file(str(source), "<END>", 1, 3, cache)
                )
            source.write_text("different text<END>", encoding="utf-8")
            self.assertNotEqual(
                first, bpe.count_pretokens_in_file(str(source), "<END>", 1, 3, cache)
            )
            tokenizer = bpe.BPETokenizer({(104, 101): 256, (256, 108): 257}, "<끝>")
            model = str(Path(directory) / "tokenizer.json")
            tokenizer.save_to(model)
            restored = bpe.BPETokenizer.load_from(model)
            self.assertEqual(restored.end_token, "<끝>")
            self.assertEqual(dict(restored.merge_rules), dict(tokenizer.merge_rules))
            self.assertEqual(restored.encode("hello 안녕<끝>"), tokenizer.encode("hello 안녕<끝>"))

    def test_invalid_json_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "tokenizer.json"
            bpe.BPETokenizer({}).save_to(str(model))
            valid = json.loads(model.read_text(encoding="utf-8"))
            invalid_fields = [
                {"version": 999},
                {"version": True},
                {"end_token": ""},
                {"end_token": 7},
                {"pretoken_pattern": "different"},
                {"merges": {}},
                {"merges": [[1, 2]]},
                {"merges": [[True, 2, 256]]},
                {"merges": [[1, 2, 257]]},
                {"merges": [[256, 2, 256]]},
                {"merges": [[-1, 2, 256]]},
                {"merges": [[1, 2, 256], [1, 2, 257]]},
            ]
            for fields in invalid_fields:
                with self.subTest(fields=fields):
                    model.write_text(json.dumps(valid | fields), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        bpe.BPETokenizer.load_from(str(model))
            model.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                bpe.BPETokenizer.load_from(str(model))

    def test_invalid_cache_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.txt"
            source.write_text("hello", encoding="utf-8")
            cache = str(Path(directory) / "cache")
            bpe.count_pretokens_in_file(str(source), "<END>", 1, 1, cache)
            cache_file = next(Path(cache).glob("*.json.gz"))
            with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                payload = json.load(f)
            for count in [-1, 0, True, 1.5, "2"]:
                with self.subTest(count=count):
                    payload["counts"] = {"hello": count}
                    with gzip.open(cache_file, "wt", encoding="utf-8") as f:
                        json.dump(payload, f)
                    with self.assertRaises(ValueError):
                        bpe.count_pretokens_in_file(str(source), "<END>", 1, 1, cache)

    def test_invalid_input_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.txt"
            output = Path(directory) / "out.bin"
            source.write_bytes(b"\xff")
            output.write_bytes(b"existing")
            tokenizer = bpe.BPETokenizer({})
            with self.assertRaises(UnicodeDecodeError):
                tokenizer.encode_file(str(source), str(output), 1, 1)
            self.assertEqual(output.read_bytes(), b"existing")
            with self.assertRaises(ValueError):
                tokenizer.encode_file(str(source), str(output), 0, 1)
            with self.assertRaises(ValueError):
                bpe.find_chunk_boundaries(str(source), 0, "<END>")
            with self.assertRaises(ValueError):
                bpe.BPETokenizer({}, "")


if __name__ == "__main__":
    unittest.main()
