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
