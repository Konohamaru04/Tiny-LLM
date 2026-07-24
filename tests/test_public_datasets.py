from __future__ import annotations

import unittest

from scripts.download_public_datasets import (
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
