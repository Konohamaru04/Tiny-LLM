from __future__ import annotations

import json
import inspect
import sys
import tempfile
import time
import unittest
from pathlib import Path

from scripts.training_dashboard import (
    ROOT,
    Stage,
    TrainingController,
    _blocks_options,
    _launch_options,
    _private_event_options,
    build_app,
    read_metrics_file,
    training_command,
)


class TrainingDashboardTests(unittest.TestCase):
    def test_training_command_resumes_latest_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            latest = output_dir / "latest.safetensors"
            latest.write_bytes(b"checkpoint")
            stage = Stage(
                key="tiny",
                label="Tiny",
                command=(sys.executable, "train.py", "--config", "tiny.yaml"),
                kind="training",
                output_dir=output_dir,
                max_steps=10,
            )

            command = training_command(stage, auto_resume=True)

            self.assertEqual(command[-2], "--resume")
            self.assertEqual(Path(command[-1]), latest.resolve())
            self.assertNotIn("--resume", training_command(stage, auto_resume=False))

    def test_metrics_reader_ignores_partial_and_invalid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.jsonl"
            valid = {"step": 10, "event": "train", "train_loss": 2.5}
            metrics_path.write_text(
                json.dumps(valid)
                + "\n"
                + '{"step": 20, "event": "train"'
                + "\n"
                + json.dumps(["not", "a", "mapping"])
                + "\n",
                encoding="utf-8",
            )

            rows = read_metrics_file(metrics_path, "pretrain_2k", "2K")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], 10)
            self.assertEqual(rows[0]["_stage"], "pretrain_2k")

    def test_controller_streams_subprocess_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = TrainingController(Path(tmp))
            stage = Stage(
                key="smoke",
                label="Smoke stage",
                command=(
                    sys.executable,
                    "-u",
                    "-c",
                    "print('dashboard-stream-ok')",
                ),
            )

            started, _ = controller.start_stages([stage])
            self.assertTrue(started)
            deadline = time.time() + 5.0
            while controller.snapshot()["running"] and time.time() < deadline:
                time.sleep(0.02)

            snapshot = controller.snapshot()
            self.assertEqual(snapshot["status"], "completed")
            self.assertIn("dashboard-stream-ok", snapshot["logs"])

    def test_gradio_app_builds_without_launching_training(self) -> None:
        controller = TrainingController(ROOT)

        app = build_app(controller)
        config = app.get_config_file()

        self.assertEqual(config["mode"], "blocks")
        self.assertGreater(len(config["components"]), 20)
        self.assertEqual(controller.snapshot()["status"], "idle")

    def test_gradio_compatibility_options_match_installed_api(self) -> None:
        import gradio as gr

        app = build_app(TrainingController(ROOT))
        block_options = _blocks_options()
        launch_options = _launch_options(app)
        event_options = _private_event_options(gr.Button.click)
        event_parameters = inspect.signature(gr.Button.click).parameters

        self.assertIn("css", block_options.keys() | launch_options.keys())
        self.assertNotIn("show_api", event_options)
        if "api_visibility" in event_parameters:
            self.assertEqual(event_options, {"api_visibility": "private"})
        else:
            self.assertEqual(event_options, {"api_name": False})


if __name__ == "__main__":
    unittest.main()
