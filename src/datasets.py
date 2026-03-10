from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.tokenizer_utils import SentencePieceTokenizer
from src.utils import assert_exists


class PretrainNpyDataset(Dataset):
    def __init__(self, npy_path: str | Path, block_size: int):
        self.npy_path = assert_exists(npy_path, "Packed training array")
        self.block_size = int(block_size)
        self.data = np.load(self.npy_path, mmap_mode="r")

        if self.data.ndim != 2:
            raise ValueError(f"Expected a 2D array in {self.npy_path}, got shape {self.data.shape}")
        expected = self.block_size + 1
        if self.data.shape[1] != expected:
            raise ValueError(
                f"Packed array second dimension must be block_size + 1 ({expected}), "
                f"but got {self.data.shape[1]} in {self.npy_path}"
            )
        if len(self.data) == 0:
            raise ValueError(f"Packed array is empty: {self.npy_path}")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = np.asarray(self.data[index], dtype=np.int64)
        x = torch.from_numpy(seq[:-1].copy())
        y = torch.from_numpy(seq[1:].copy())
        return x, y


class SFTJsonlDataset(Dataset):
    REQUIRED_KEYS = ("system", "user", "assistant")

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: SentencePieceTokenizer,
        block_size: int,
    ):
        self.jsonl_path = assert_exists(jsonl_path, "SFT JSONL file")
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.max_seq_len = self.block_size + 1
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSONL row in {self.jsonl_path} at line {line_no}: {exc}"
                    ) from exc

                self.examples.append(self._build_example(row, line_no))

        if not self.examples:
            raise ValueError(f"No usable SFT examples found in {self.jsonl_path}")

    def _normalize_text(self, value: Any, key: str, line_no: int) -> str:
        if not isinstance(value, str):
            raise ValueError(
                f"SFT row {line_no} in {self.jsonl_path} has non-string field '{key}': {type(value).__name__}"
            )
        return value.replace("\r\n", "\n").replace("\r", "\n").strip()

    def _validate_row(self, row: Dict[str, Any], line_no: int) -> None:
        if not isinstance(row, dict):
            raise ValueError(f"SFT row {line_no} in {self.jsonl_path} must be a JSON object.")
        missing = [k for k in self.REQUIRED_KEYS if k not in row]
        if missing:
            raise ValueError(
                f"SFT row {line_no} in {self.jsonl_path} is missing required fields: {missing}"
            )

    def _encode_role_text(self, role_token: str, text: str) -> List[int]:
        chunk = f"{role_token}\n{text}\n"
        return self.tokenizer.encode(chunk, add_bos=False, add_eos=False)

    def _build_example(self, row: Dict[str, Any], line_no: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_row(row, line_no)

        system_text = self._normalize_text(row["system"], "system", line_no)
        user_text = self._normalize_text(row["user"], "user", line_no)
        assistant_text = self._normalize_text(row["assistant"], "assistant", line_no)

        bos = [self.tokenizer.bos_id]
        prefix_tokens = bos[:]

        if system_text:
            prefix_tokens.extend(self._encode_role_text("<|system|>", system_text))

        prefix_tokens.extend(self._encode_role_text("<|user|>", user_text))
        prefix_tokens.extend(self.tokenizer.encode("<|assistant|>\n", add_bos=False, add_eos=False))

        assistant_tokens = self.tokenizer.encode(assistant_text, add_bos=False, add_eos=False)
        assistant_tokens.append(self.tokenizer.eos_id)

        if len(prefix_tokens) + len(assistant_tokens) <= self.max_seq_len:
            full_seq = prefix_tokens + assistant_tokens
            assistant_start = len(prefix_tokens)
        else:
            if len(assistant_tokens) >= self.max_seq_len:
                full_seq = assistant_tokens[-self.max_seq_len :]
                assistant_start = 0
            else:
                keep_prefix = self.max_seq_len - len(assistant_tokens)
                full_seq = prefix_tokens[-keep_prefix:] + assistant_tokens
                assistant_start = len(full_seq) - len(assistant_tokens)

        if len(full_seq) < self.max_seq_len:
            pad_len = self.max_seq_len - len(full_seq)
            full_seq.extend([self.tokenizer.pad_id] * pad_len)

        input_ids = full_seq[:-1]
        labels: List[int] = []

        for i in range(len(full_seq) - 1):
            target_index = i + 1
            target_token = full_seq[target_index]

            if target_token == self.tokenizer.pad_id or target_index < assistant_start:
                labels.append(-100)
            else:
                labels.append(target_token)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]