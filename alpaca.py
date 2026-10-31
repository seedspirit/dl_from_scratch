from base.tokenizer import BPETokenizer
import json

# 토크나이저 불러오기
tokenizer = BPETokenizer.load_from("codebot/merge_rules.pkl")
# JSON 데이터 읽기
with open("codebot/tiny_codes_sft.json") as f:
    data = json.load(f)
# 첫 번째 샘플 꺼내기
item = data[0]
print(item)
text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}<|endoftext|>"
print(text)
ids = tokenizer.encode(text)
print(ids)
