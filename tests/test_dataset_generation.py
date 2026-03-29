from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.dataset_generation import (
    compute_stats,
    deduplicate_records,
    extract_highest_train_index,
    make_seed_records,
    read_jsonl_records,
    write_jsonl_records,
)


class DatasetGenerationTests(unittest.TestCase):
    def test_seed_records_start_from_requested_index(self) -> None:
        records = make_seed_records(start_index=7, count=3)

        self.assertEqual([record["id"] for record in records], ["train_7", "train_8", "train_9"])
        self.assertTrue(all(record["user"] for record in records))
        self.assertTrue(all(record["assistant"] for record in records))

    def test_read_dedupe_and_stats_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_train.jsonl"
            records = make_seed_records(start_index=1, count=3)
            duplicate = dict(records[1])
            duplicate["id"] = "train_99"
            write_jsonl_records(path, records + [duplicate])

            valid_records, failures = read_jsonl_records(path)
            unique_records, removals = deduplicate_records(valid_records)
            stats = compute_stats(unique_records)

            self.assertEqual(len(failures), 0)
            self.assertEqual(len(removals), 1)
            self.assertEqual(extract_highest_train_index(unique_records), 3)
            self.assertEqual(stats["total_records"], 3)
