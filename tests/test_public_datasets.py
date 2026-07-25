from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.download_public_datasets import (
    download,
    parse_python_function_calls,
    transform_openr1,
)


class PublicDatasetTransformTests(unittest.TestCase):
    def test_python_style_function_calls_are_normalized(self) -> None:
        calls = parse_python_function_calls(
            'weather.forecast(q="Paris", days=5)\n'
            'route-plan(origin="A", destination="B")'
        )

        self.assertEqual([call["function"]["name"] for call in calls], [
            "weather.forecast",
            "route-plan",
        ])
        self.assertIn('"days":5', calls[0]["function"]["arguments"])

    def test_openr1_reasoning_is_split_from_final_answer(self) -> None:
        record = transform_openr1(
            {
                "problem": "What is 2 + 2?",
                "uuid": "example",
                "generations": ["<think>Add the integers.</think>\n\nThe answer is 4."],
                "is_reasoning_complete": [True],
                "correctness_math_verify": [True],
            },
            {
                "dataset": "open-r1/example",
                "revision": "abc",
                "license": "Apache-2.0",
            },
        )

        self.assertIsNotNone(record)
        assert record is not None
        assistant = record["messages"][-1]
        self.assertEqual(assistant["reasoning_content"], "Add the integers.")
        self.assertEqual(assistant["content"], "The answer is 4.")

    def test_stage_filter_skips_network_for_excluded_sources_and_uses_portable_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "configs" / "public_datasets.yaml"
            sft_dir = root / "data" / "public_sft"
            pretrain_dir = root / "data" / "raw" / "public"
            config_path.parent.mkdir(parents=True)
            sft_dir.mkdir(parents=True)
            (sft_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
            (sft_dir / "validation.jsonl").write_text("{}\n", encoding="utf-8")
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "request_timeout_seconds": 30,
                        "request_delay_seconds": 0,
                        "seed": 7,
                        "validation_fraction": 0.1,
                        "sft_output_dir": str(sft_dir),
                        "pretrain_output_dir": str(pretrain_dir),
                        "sources": [
                            {
                                "name": "included_pretrain",
                                "dataset": "public/included",
                                "config": "default",
                                "stage": "pretrain",
                                "split": "train",
                                "license": "test",
                                "limit": 1,
                                "scan_limit": 1,
                                "text_field": "text",
                                "min_chars": 1,
                            },
                            {
                                "name": "excluded_sft",
                                "dataset": "public/must-not-be-contacted",
                                "config": "default",
                                "stage": "sft",
                                "split": "train",
                                "license": "test",
                                "limit": 1,
                                "transform": "does_not_matter",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            metadata = {
                "dataset": "public/included",
                "sha": "abc123",
                "last_modified": "2026-07-25T00:00:00Z",
                "private": False,
                "gated": False,
            }
            with (
                patch("scripts.download_public_datasets.ROOT", root),
                patch(
                    "scripts.download_public_datasets.fetch_dataset_metadata",
                    return_value=metadata,
                ) as fetch_metadata,
                patch(
                    "scripts.download_public_datasets.iter_viewer_rows",
                    return_value=[(0, {"id": "doc-1", "text": "public text"})],
                ),
            ):
                manifest = download(config_path, stage_filter="pretrain")

            fetch_metadata.assert_called_once_with("public/included", 30)
            self.assertEqual(manifest["config"], "configs/public_datasets.yaml")
            self.assertEqual(
                manifest["sft_train_path"],
                "data/public_sft/train.jsonl",
            )
            self.assertEqual(
                manifest["sft_validation_path"],
                "data/public_sft/validation.jsonl",
            )
