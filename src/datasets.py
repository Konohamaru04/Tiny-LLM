from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.tokenizer_utils import SentencePieceTokenizer
from src.chat_format import encode_conversation
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

    def _validate_row(self, row: Dict[str, Any], line_no: int) -> None:
        if not isinstance(row, dict):
            raise ValueError(f"SFT row {line_no} in {self.jsonl_path} must be a JSON object.")
        has_messages = isinstance(row.get("messages"), list) and bool(row["messages"])
        has_legacy = all(key in row for key in ("system", "user", "assistant"))
        if not has_messages and not has_legacy:
            raise ValueError(
                f"SFT row {line_no} in {self.jsonl_path} must contain a non-empty 'messages' "
                "list or legacy system/user/assistant fields."
            )

    def _messages_from_row(self, row: Mapping[str, Any], line_no: int) -> List[Dict[str, Any]]:
        raw_messages = row.get("messages")
        if isinstance(raw_messages, list) and raw_messages:
            messages: List[Dict[str, Any]] = []
            for message_index, message in enumerate(raw_messages):
                if not isinstance(message, dict):
                    raise ValueError(
                        f"SFT row {line_no} message {message_index} in {self.jsonl_path} "
                        "must be an object."
                    )
                messages.append(dict(message))
            return messages

        values: Dict[str, str] = {}
        for key in ("system", "user", "assistant"):
            value = row.get(key)
            if not isinstance(value, str):
                raise ValueError(
                    f"SFT row {line_no} in {self.jsonl_path} has non-string field "
                    f"'{key}': {type(value).__name__}"
                )
            values[key] = value
        messages = []
        if values["system"].strip():
            messages.append({"role": "system", "content": values["system"]})
        messages.append({"role": "user", "content": values["user"]})
        messages.append({"role": "assistant", "content": values["assistant"]})
        return messages

    def _build_example(self, row: Dict[str, Any], line_no: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_row(row, line_no)
        messages = self._messages_from_row(row, line_no)
        raw_tools = row.get("tools")
        if raw_tools in (None, ""):
            tools = None
        elif isinstance(raw_tools, list):
            tools = raw_tools
        else:
            raise ValueError(f"SFT row {line_no} in {self.jsonl_path} has non-list 'tools'.")

        full_seq, full_loss_mask = encode_conversation(
            self.tokenizer,
            messages,
            tools=tools,
        )

        if len(full_seq) > self.max_seq_len:
            supervised_positions = [
                index for index, supervised in enumerate(full_loss_mask) if supervised
            ]
            if not supervised_positions:
                raise ValueError(
                    f"SFT row {line_no} in {self.jsonl_path} has no assistant tokens to supervise."
                )
            # End the window at the last assistant target. Long agent traces often
            # finish with a large tool response; blindly taking the tail can leave
            # an almost entirely masked training example.
            window_end = supervised_positions[-1] + 1
            window_start = max(0, window_end - self.max_seq_len)
            full_seq = full_seq[window_start:window_end]
            full_loss_mask = full_loss_mask[window_start:window_end]
            full_seq[0] = self.tokenizer.bos_id
            full_loss_mask[0] = False

        if len(full_seq) < self.max_seq_len:
            pad_len = self.max_seq_len - len(full_seq)
            full_seq.extend([self.tokenizer.pad_id] * pad_len)
            full_loss_mask.extend([False] * pad_len)

        if not any(full_loss_mask):
            raise ValueError(
                f"SFT row {line_no} in {self.jsonl_path} has no assistant tokens to supervise."
            )

        input_ids = full_seq[:-1]
        labels: List[int] = []

        for i in range(len(full_seq) - 1):
            target_index = i + 1
            target_token = full_seq[target_index]

            if target_token == self.tokenizer.pad_id or not full_loss_mask[target_index]:
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
