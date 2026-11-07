"""JSON 설정으로 토크나이저를 학습하고 파일을 인코딩하는 Click 명령."""

from pathlib import Path

import click

from storybot.args import (
    EncodingArgs,
    InputModel,
    TokenizerArgs,
    TokenizerLoadArgs,
    TokenizerTrainingArgs,
)
from storybot.bpe import PreTokenizer
from storybot.encoding import FileEncoder
from storybot.profiling import print_profile_summary
from storybot.storage import FrequencyCache
from storybot.tokenizer import BPETokenizer
from storybot.training import BPETrainer
from storybot.training_steps import PretokenCounter


class TrainCommandArgs(TokenizerTrainingArgs):
    model_path: Path


class EncodeCommandArgs(InputModel):
    tokenizer: TokenizerLoadArgs
    encoding: EncodingArgs


@click.group()
def main() -> None:
    """Train a tokenizer or encode a text file."""


@main.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON configuration path",
)
def train(config: Path) -> None:
    """Train and save a tokenizer."""
    pretokenizer = PreTokenizer()
    try:
        train_args = TrainCommandArgs.model_validate_json(config.read_text(encoding="utf-8"))
        trainer = BPETrainer(
            counter=PretokenCounter(pretokenizer),
            cache=FrequencyCache(),
        )
        rules = trainer.train(train_args.training)
        tokenizer_args = TokenizerArgs(
            tokenization=train_args.training.tokenization,
            cache_size=train_args.cache_size,
        )
        tokenizer = BPETokenizer(rules, tokenizer_args, pretokenizer)
        train_args.model_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save_to(train_args.model_path)
        click.echo(f"Saved tokenizer: {train_args.model_path} (vocabulary: {tokenizer.vocab_size})")
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    finally:
        print_profile_summary()


@main.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON configuration path",
)
def encode(config: Path) -> None:
    """Load a tokenizer and encode a text file."""
    pretokenizer = PreTokenizer()
    try:
        encode_args = EncodeCommandArgs.model_validate_json(config.read_text(encoding="utf-8"))
        tokenizer = BPETokenizer.load_from(encode_args.tokenizer, pretokenizer)
        encoding = encode_args.encoding
        encoding.destination.parent.mkdir(parents=True, exist_ok=True)
        encoder = FileEncoder(pretokenizer=pretokenizer)
        count = encoder.encode_file(tokenizer, encoding)
        click.echo(f"Encoded file: {encoding.destination} (tokens: {count})")
    except (OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    finally:
        print_profile_summary()
