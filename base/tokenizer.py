import pickle


def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:  # ❶
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids


class BPETokenizer:
    def __init__(self, merge_rules, end_token="<|endoftext|>") -> None:
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = 256 + len(merge_rules)
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
        self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")
        self.vocab_size = len(self.id_to_bytes)

    def _encode_text(self, text):
        ids = list(text.encode("utf-8"))
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)
        return ids

    def encode(self, input_text):
        import re

        pattern = "(" + re.escape(self.end_token) + ")"
        texts = re.split(pattern, input_text)
        all_ids = []
        for text in texts:
            if text == self.end_token:  # ❶
                all_ids.append(self.end_token_id)
            else:  # ❷
                ids = self._encode_text(text)
                all_ids.extend(ids)
        return all_ids

    def decode(self, ids: list[int]) -> str:
        byte_lists = [self.id_to_bytes[i] for i in ids]
        text_bytes = b"".join(byte_lists)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    @staticmethod
    def load_from(filepath: str) -> "BPETokenizer":
        with open(filepath, "rb") as f:
            merge_rules = pickle.load(f)
        return BPETokenizer(merge_rules, end_token="<|endoftext|>")
