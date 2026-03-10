from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import sentencepiece as spm

from src.utils import assert_exists, ensure_parent_dir, resolve_path


REQUIRED_USER_DEFINED_SYMBOLS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|json|>",
    "</json>",
]


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path):
        self.model_path = assert_exists(model_path, "Tokenizer model")
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self._verify_required_tokens()

    def _verify_required_tokens(self) -> None:
        required = ["<pad>", "<bos>", "<eos>", "<unk>"] + REQUIRED_USER_DEFINED_SYMBOLS
        missing = [tok for tok in required if self.sp.piece_to_id(tok) < 0]
        if missing:
            raise ValueError(
                f"Tokenizer is missing required special tokens: {missing}. "
                f"Model file: {self.model_path}"
            )

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    @property
    def pad_id(self) -> int:
        return int(self.sp.pad_id())

    @property
    def bos_id(self) -> int:
        return int(self.sp.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self.sp.eos_id())

    @property
    def unk_id(self) -> int:
        return int(self.sp.unk_id())

    def token_to_id(self, token: str) -> int:
        token_id = int(self.sp.piece_to_id(token))
        if token_id < 0:
            raise ValueError(f"Token not found in tokenizer vocabulary: {token}")
        return token_id

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"encode() expected a string, got {type(text).__name__}")
        return list(self.sp.encode(text, out_type=int, add_bos=add_bos, add_eos=add_eos))

    def decode(self, token_ids: Sequence[int], skip_basic_special_tokens: bool = True) -> str:
        ids = [int(t) for t in token_ids if int(t) >= 0]
        if skip_basic_special_tokens:
            basic = {self.pad_id, self.bos_id, self.eos_id}
            ids = [t for t in ids if t not in basic]
        return self.sp.decode(ids)

    def get_special_token_map(self) -> dict[str, int]:
        mapping = {
            "<pad>": self.pad_id,
            "<bos>": self.bos_id,
            "<eos>": self.eos_id,
            "<unk>": self.unk_id,
        }
        for tok in REQUIRED_USER_DEFINED_SYMBOLS:
            mapping[tok] = self.token_to_id(tok)
        return mapping


def train_sentencepiece_tokenizer(
    input_text_path: str | Path,
    output_prefix: str | Path,
    vocab_size: int = 16000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    normalization_rule_name: str = "identity",
    user_defined_symbols: Iterable[str] | None = None,
    unk_id: int = 0,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 3,
) -> tuple[Path, Path]:
    input_text_path = assert_exists(input_text_path, "Tokenizer training corpus")
    output_prefix = resolve_path(output_prefix)
    ensure_parent_dir(output_prefix)

    user_defined_symbols = list(user_defined_symbols or REQUIRED_USER_DEFINED_SYMBOLS)
    if vocab_size <= len(user_defined_symbols) + 16:
        raise ValueError(
            "vocab_size is too small for the required special tokens and normal subword pieces."
        )

    cmd_parts = {
        "input": str(input_text_path),
        "model_prefix": str(output_prefix),
        "model_type": model_type,
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "normalization_rule_name": normalization_rule_name,
        "unk_id": unk_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "unk_piece": "<unk>",
        "bos_piece": "<bos>",
        "eos_piece": "<eos>",
        "pad_piece": "<pad>",
        "user_defined_symbols": ",".join(user_defined_symbols),
        "hard_vocab_limit": "false",
        "split_digits": "false",
        "byte_fallback": "false",
        "max_sentence_length": 16384,
    }

    command = " ".join(f"--{key}={value}" for key, value in cmd_parts.items())
    spm.SentencePieceTrainer.train(command)

    model_path = output_prefix.with_suffix(".model")
    vocab_path = output_prefix.with_suffix(".vocab")

    if not model_path.exists() or not vocab_path.exists():
        raise RuntimeError(
            f"SentencePiece training did not create expected files:\n"
            f"  {model_path}\n"
            f"  {vocab_path}"
        )

    return model_path, vocab_path