from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.tokenizer_utils import SentencePieceTokenizer, train_sentencepiece_tokenizer


SAMPLE_DOCS = [
    "# Tokenizers\n\nA tiny tokenizer should still preserve special chat tags.\n",
    "# Validation\n\nTrain and validation splits should remain deterministic.\n",
    "# Chat\n\nSystem, user, and assistant turns help shape SFT prompts.\n",
]


def write_markdown_docs(root: Path, docs: Iterable[str] | None = None) -> Path:
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for idx, text in enumerate(docs or SAMPLE_DOCS, start=1):
        (raw_dir / f"{idx:03d}.md").write_text(text, encoding="utf-8")
    return raw_dir


def train_test_tokenizer(root: Path, vocab_size: int = 48) -> SentencePieceTokenizer:
    corpus_path = root / "corpus.txt"
    corpus_path.write_text("\n".join(SAMPLE_DOCS), encoding="utf-8")
    model_prefix = root / "tokenizer"
    model_path, _ = train_sentencepiece_tokenizer(
        input_text_path=corpus_path,
        output_prefix=model_prefix,
        vocab_size=vocab_size,
    )
    return SentencePieceTokenizer(model_path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_eval_prompts(path: Path) -> Path:
    return write_jsonl(
        path,
        [
            {
                "name": "plain",
                "system": "You explain things clearly.",
                "user": "What is tokenization?",
                "json_mode": False,
            },
            {
                "name": "json",
                "system": "Return JSON when asked.",
                "user": "Return a JSON object with a key named topic.",
                "json_mode": True,
            },
        ],
    )


def write_personas(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "name": "technical",
                    "system_prompt": "You explain technical topics clearly.",
                    "description": "General technical helper",
                    "json_mode": False,
                },
                {
                    "name": "json_bot",
                    "system_prompt": "Return JSON when the user asks for structure.",
                    "description": "Structured output helper",
                    "json_mode": True,
                },
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
