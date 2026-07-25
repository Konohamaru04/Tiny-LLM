from __future__ import annotations

import hashlib
import json
import re
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from src.tokenizer_utils import SentencePieceTokenizer
from src.utils import PROJECT_ROOT, assert_exists, read_json, resolve_path, write_json


def collect_markdown_files(raw_dir: str | Path) -> List[Path]:
    raw_path = assert_exists(raw_dir, "Raw data directory")
    if not raw_path.is_dir():
        raise NotADirectoryError(f"Raw data path is not a directory: {raw_path}")

    files = sorted(p for p in raw_path.rglob("*.md") if p.is_file())
    if not files:
        raise FileNotFoundError(
            f"No Markdown files were found under: {raw_path}\n"
            f"Place one or more .md files under data/raw/ and try again."
        )
    return files


def deduplicate_markdown_files(paths: Sequence[Path]) -> tuple[List[Path], List[Path]]:
    """Keep the first deterministic path for each normalized document body."""
    unique: List[Path] = []
    duplicates: List[Path] = []
    seen_hashes: set[str] = set()
    for path in sorted(paths):
        text = read_markdown_file(path)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            duplicates.append(path)
            continue
        seen_hashes.add(content_hash)
        unique.append(path)
    return unique, duplicates


def normalize_markdown_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_markdown_file(path: str | Path) -> str:
    path = assert_exists(path, "Markdown file")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return normalize_markdown_text(f.read())


def deterministic_train_val_split(
    paths: Sequence[Path],
    val_fraction: float,
    seed: int,
) -> tuple[List[Path], List[Path]]:
    if len(paths) < 2:
        raise ValueError(
            "At least 2 Markdown files are required for a document-level train/validation split."
        )
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")

    items = list(paths)
    rnd = random.Random(seed)
    rnd.shuffle(items)

    val_count = max(1, int(round(len(items) * val_fraction)))
    if val_count >= len(items):
        val_count = len(items) - 1
    if val_count <= 0:
        raise ValueError("Validation split produced 0 documents. Increase dataset size.")

    val_files = sorted(items[:val_count])
    train_files = sorted(items[val_count:])

    if not train_files:
        raise ValueError("Train split is empty after splitting. Add more Markdown files.")
    if not val_files:
        raise ValueError("Validation split is empty after splitting. Add more Markdown files.")

    return train_files, val_files


def _path_to_manifest_entry(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.resolve().as_posix()


def save_manifest(paths: Sequence[Path], out_path: str | Path) -> None:
    data = {"files": [_path_to_manifest_entry(p) for p in paths]}
    write_json(data, out_path)


def load_manifest(manifest_path: str | Path) -> List[Path]:
    data = read_json(manifest_path)
    if not isinstance(data, dict) or "files" not in data or not isinstance(data["files"], list):
        raise ValueError(f"Invalid manifest format: {resolve_path(manifest_path)}")

    resolved_paths: List[Path] = []
    for item in data["files"]:
        if not isinstance(item, str):
            raise ValueError(f"Manifest contains a non-string file entry: {item!r}")
        p = resolve_path(item)
        if not p.exists():
            raise FileNotFoundError(
                f"Manifest references a file that no longer exists: {p}\n"
                f"Manifest: {resolve_path(manifest_path)}"
            )
        resolved_paths.append(p)
    return resolved_paths


def write_combined_corpus(paths: Iterable[Path], out_path: str | Path) -> int:
    docs: List[str] = []
    for p in paths:
        text = read_markdown_file(p)
        if text:
            docs.append(text)

    if not docs:
        raise ValueError(
            f"All documents were empty after normalization. Cannot build corpus: {resolve_path(out_path)}"
        )

    out_path = resolve_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined = "\n\n".join(docs) + "\n"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(combined)
    return len(docs)


def _sft_text_segments(row: dict) -> Iterable[str]:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            for key in ("content", "reasoning_content", "reasoning"):
                value = message.get(key)
                if isinstance(value, str) and value.strip():
                    yield value.strip()
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                yield json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))
    else:
        for key in ("system", "user", "assistant"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                yield value.strip()

    tools = row.get("tools")
    if isinstance(tools, list) and tools:
        yield json.dumps(tools, ensure_ascii=False, separators=(",", ":"))


def append_sft_jsonl_to_corpus(
    jsonl_paths: Sequence[str | Path],
    corpus_path: str | Path,
) -> dict:
    """Append chat/tool text to a tokenizer corpus without leaking JSONL validation data."""
    corpus = resolve_path(corpus_path)
    if not corpus.exists():
        raise FileNotFoundError(f"Tokenizer corpus must exist before appending SFT text: {corpus}")

    records = 0
    segments = 0
    with corpus.open("a", encoding="utf-8") as output:
        for jsonl_path in jsonl_paths:
            source = assert_exists(jsonl_path, "Additional tokenizer JSONL corpus")
            with source.open("r", encoding="utf-8") as input_file:
                for line_no, line in enumerate(input_file, start=1):
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Malformed JSONL in tokenizer corpus {source}:{line_no}: {exc}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"Tokenizer corpus row must be an object: {source}:{line_no}"
                        )
                    records += 1
                    for segment in _sft_text_segments(row):
                        normalized = normalize_markdown_text(segment)
                        if not normalized:
                            continue
                        # SentencePiece skips lines over max_sentence_length, so split
                        # long tool trajectories into bounded training lines.
                        for start in range(0, len(normalized), 12000):
                            output.write(normalized[start : start + 12000])
                            output.write("\n")
                            segments += 1

    return {"records": records, "segments": segments}


def encode_documents(
    paths: Sequence[Path],
    tokenizer: SentencePieceTokenizer,
    add_bos: bool = True,
    add_eos: bool = True,
) -> List[int]:
    token_stream: List[int] = []
    kept_docs = 0
    for p in paths:
        text = read_markdown_file(p)
        if not text:
            continue
        token_stream.extend(tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos))
        kept_docs += 1

    if kept_docs == 0:
        raise ValueError("No non-empty documents were available to encode.")
    return token_stream


def pack_token_stream(token_ids: Sequence[int], block_size: int) -> tuple[np.ndarray, dict]:
    if block_size <= 1:
        raise ValueError("block_size must be greater than 1")
    if not token_ids:
        raise ValueError("Token stream is empty.")

    seq_len = block_size + 1
    usable_tokens = (len(token_ids) // seq_len) * seq_len
    if usable_tokens < seq_len:
        raise ValueError(
            f"Not enough tokens to form a single training block of length {seq_len}. "
            f"Need at least {seq_len} tokens, got {len(token_ids)}."
        )

    packed = np.asarray(token_ids[:usable_tokens], dtype=np.int32).reshape(-1, seq_len)
    meta = {
        "total_input_tokens": int(len(token_ids)),
        "usable_tokens": int(usable_tokens),
        "dropped_tail_tokens": int(len(token_ids) - usable_tokens),
        "num_sequences": int(packed.shape[0]),
        "sequence_length": int(seq_len),
        "block_size": int(block_size),
    }
    return packed, meta
