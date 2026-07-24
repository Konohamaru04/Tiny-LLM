from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import gradio as gr
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FULL_CURRICULUM = "Full curriculum · 2K → 4K → 16K → SFT"
PRETRAIN_CURRICULUM = "Pretrain only · 2K → 4K → 16K"
SFT_ONLY = "SFT only · use existing 16K checkpoint"

PRESETS = {
    FULL_CURRICULUM: ("pretrain_2k", "pretrain_4k", "pretrain_16k", "sft_16k"),
    PRETRAIN_CURRICULUM: ("pretrain_2k", "pretrain_4k", "pretrain_16k"),
    SFT_ONLY: ("sft_16k",),
}

TRAINING_CONFIGS = {
    "pretrain_2k": (
        "2K MoE pretraining",
        "scripts/train_pretrain.py",
        "configs/pretrain_tiny.yaml",
    ),
    "pretrain_4k": (
        "4K context continuation",
        "scripts/train_pretrain.py",
        "configs/pretrain_long_context_4k.yaml",
    ),
    "pretrain_16k": (
        "16K context continuation",
        "scripts/train_pretrain.py",
        "configs/pretrain_long_context_16k.yaml",
    ),
    "sft_16k": (
        "16K reasoning + tool SFT",
        "scripts/train_sft.py",
        "configs/sft_long_context_16k.yaml",
    ),
}

STAGE_COLORS = {
    "pretrain_2k": "#7c83ff",
    "pretrain_4k": "#22d3ee",
    "pretrain_16k": "#34d399",
    "sft_16k": "#fb7185",
}


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    command: tuple[str, ...]
    kind: str = "setup"
    metrics_path: Path | None = None
    output_dir: Path | None = None
    max_steps: int = 0
    block_size: int = 0


def _repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_yaml(root: Path, relative_path: str) -> dict[str, Any]:
    path = _repo_path(root, relative_path)
    with path.open("r", encoding="utf-8") as config_file:
        payload = yaml.safe_load(config_file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def load_training_stages(root: Path = ROOT) -> dict[str, Stage]:
    root = root.resolve()
    stages: dict[str, Stage] = {}
    for key, (label, script, config_path) in TRAINING_CONFIGS.items():
        payload = _load_yaml(root, config_path)
        model = payload.get("model", {})
        training = payload.get("training", {})
        if not isinstance(model, dict) or not isinstance(training, dict):
            raise ValueError(f"Missing model/training mappings in {config_path}")
        output_dir = _repo_path(root, str(training["output_dir"]))
        metrics_path = _repo_path(
            root,
            str(training.get("metrics_jsonl_path") or output_dir / "metrics.jsonl"),
        )
        stages[key] = Stage(
            key=key,
            label=label,
            command=(
                sys.executable,
                "-u",
                str(_repo_path(root, script)),
                "--config",
                str(_repo_path(root, config_path)),
            ),
            kind="training",
            metrics_path=metrics_path,
            output_dir=output_dir,
            max_steps=int(training["max_steps"]),
            block_size=int(model["block_size"]),
        )
    return stages


def training_command(stage: Stage, auto_resume: bool = True) -> tuple[str, ...]:
    command = list(stage.command)
    if stage.kind == "training" and auto_resume and stage.output_dir is not None:
        latest = stage.output_dir / "latest.safetensors"
        if latest.exists():
            command.extend(("--resume", str(latest.resolve())))
    return tuple(command)


def _tokenizer_ready(root: Path) -> bool:
    required = (
        "data/processed/tokenizer.model",
        "data/processed/tokenizer.vocab",
        "data/processed/tokenizer_meta.json",
        "data/processed/train_manifest.json",
        "data/processed/val_manifest.json",
    )
    return all(_repo_path(root, path).exists() for path in required)


def _public_sft_ready(root: Path) -> bool:
    return all(
        _repo_path(root, path).exists()
        for path in (
            "data/public_sft/train.jsonl",
            "data/public_sft/validation.jsonl",
        )
    )


def _prepared_data_ready(root: Path, config_path: str) -> bool:
    payload = _load_yaml(root, config_path)
    data = payload.get("data", {})
    if not isinstance(data, dict):
        return False
    required_keys = ("train_array_path", "val_array_path", "meta_path")
    return all(
        key in data and _repo_path(root, str(data[key])).exists()
        for key in required_keys
    )


def build_pipeline(
    preset: str,
    refresh_public_data: bool,
    rebuild_tokenizer_and_data: bool,
    auto_resume: bool,
    root: Path = ROOT,
) -> list[Stage]:
    root = root.resolve()
    if preset not in PRESETS:
        raise ValueError(f"Unknown training preset: {preset}")

    selected_keys = PRESETS[preset]
    training_stages = load_training_stages(root)
    pipeline: list[Stage] = []

    public_data_missing = not _public_sft_ready(root)
    if refresh_public_data or public_data_missing:
        pipeline.append(
            Stage(
                key="download_data",
                label="Refresh pinned public datasets",
                command=(
                    sys.executable,
                    "-u",
                    str(_repo_path(root, "scripts/download_public_datasets.py")),
                    "--config",
                    str(_repo_path(root, "configs/public_datasets.yaml")),
                    "--stage",
                    "all",
                ),
            )
        )

    tokenizer_missing = not _tokenizer_ready(root)
    pretraining_selected = any(key.startswith("pretrain_") for key in selected_keys)
    if preset == SFT_ONLY and (rebuild_tokenizer_and_data or tokenizer_missing):
        raise ValueError(
            "SFT-only mode needs the existing tokenizer that matches the 16K "
            "pretrained checkpoint. Choose the full curriculum to rebuild it safely."
        )

    rebuild_inputs = rebuild_tokenizer_and_data or (
        (refresh_public_data or public_data_missing) and pretraining_selected
    )
    if pretraining_selected and (rebuild_inputs or tokenizer_missing):
        pipeline.append(
            Stage(
                key="tokenizer",
                label="Build 8K tool-aware tokenizer",
                command=(
                    sys.executable,
                    "-u",
                    str(_repo_path(root, "scripts/train_tokenizer.py")),
                    "--config",
                    str(_repo_path(root, "configs/tokenizer.yaml")),
                ),
            )
        )

    for key in selected_keys:
        if not key.startswith("pretrain_"):
            continue
        _, _, config_path = TRAINING_CONFIGS[key]
        if rebuild_inputs or tokenizer_missing or not _prepared_data_ready(root, config_path):
            pipeline.append(
                Stage(
                    key=f"prepare_{key}",
                    label=f"Pack {training_stages[key].block_size // 1024}K token arrays",
                    command=(
                        sys.executable,
                        "-u",
                        str(_repo_path(root, "scripts/prepare_data.py")),
                        "--config",
                        str(_repo_path(root, config_path)),
                    ),
                )
            )

    for key in selected_keys:
        stage = training_stages[key]
        pipeline.append(
            Stage(
                key=stage.key,
                label=stage.label,
                command=training_command(stage, auto_resume=auto_resume),
                kind=stage.kind,
                metrics_path=stage.metrics_path,
                output_dir=stage.output_dir,
                max_steps=stage.max_steps,
                block_size=stage.block_size,
            )
        )
    return pipeline


class TrainingController:
    def __init__(self, root: Path = ROOT, max_log_lines: int = 2500) -> None:
        self.root = root.resolve()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._pipeline: list[Stage] = []
        self._current_index = -1
        self._completed_count = 0
        self._status = "idle"
        self._message = "Ready for launch."
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._logs: deque[str] = deque(maxlen=max_log_lines)

    def start(
        self,
        preset: str,
        refresh_public_data: bool,
        rebuild_tokenizer_and_data: bool,
        auto_resume: bool,
    ) -> tuple[bool, str]:
        try:
            pipeline = build_pipeline(
                preset,
                bool(refresh_public_data),
                bool(rebuild_tokenizer_and_data),
                bool(auto_resume),
                root=self.root,
            )
        except Exception as exc:
            return False, str(exc)
        return self.start_stages(pipeline)

    def start_stages(self, pipeline: Sequence[Stage]) -> tuple[bool, str]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False, "A training pipeline is already running."
            if not pipeline:
                return False, "The selected pipeline has no stages."

            self._pipeline = list(pipeline)
            self._current_index = -1
            self._completed_count = 0
            self._status = "running"
            self._message = "Starting pipeline…"
            self._started_at = time.time()
            self._ended_at = None
            self._logs.clear()
            self._stop_event.clear()
            self._append_log(
                f"[dashboard] queued {len(self._pipeline)} stage(s): "
                + " → ".join(stage.label for stage in self._pipeline)
            )
            self._thread = threading.Thread(
                target=self._run_pipeline,
                name="tiny-llm-training-pipeline",
                daemon=True,
            )
            self._thread.start()
        return True, f"Started {len(pipeline)} stage(s)."

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                return False, "No training pipeline is running."
            self._stop_event.set()
            self._status = "stopping"
            self._message = "Stopping after the active process exits…"
            process = self._process
            self._append_log("[dashboard] stop requested")

        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        return True, "Stop requested. The current process is being terminated safely."

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self._ended_at or time.time()
            elapsed = 0.0 if self._started_at is None else max(0.0, now - self._started_at)
            current_stage = (
                self._pipeline[self._current_index]
                if 0 <= self._current_index < len(self._pipeline)
                else None
            )
            return {
                "status": self._status,
                "message": self._message,
                "pipeline": list(self._pipeline),
                "current_index": self._current_index,
                "completed_count": self._completed_count,
                "current_stage": current_stage,
                "elapsed_seconds": elapsed,
                "logs": "\n".join(self._logs),
                "running": self._thread is not None and self._thread.is_alive(),
            }

    def _append_log(self, line: str) -> None:
        clean = line.rstrip("\r\n")
        if clean:
            self._logs.append(clean)

    def _run_pipeline(self) -> None:
        try:
            for index, stage in enumerate(self._pipeline):
                if self._stop_event.is_set():
                    break
                with self._lock:
                    self._current_index = index
                    self._message = stage.label
                    self._append_log("")
                    self._append_log(
                        f"[dashboard] stage {index + 1}/{len(self._pipeline)} · {stage.label}"
                    )
                    self._append_log(
                        "[dashboard] command: "
                        + subprocess.list2cmdline(list(stage.command))
                    )

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                process = subprocess.Popen(
                    list(stage.command),
                    cwd=self.root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    shell=False,
                )
                with self._lock:
                    self._process = process

                assert process.stdout is not None
                try:
                    for line in iter(process.stdout.readline, ""):
                        with self._lock:
                            self._append_log(line)
                finally:
                    process.stdout.close()
                return_code = process.wait()
                with self._lock:
                    self._process = None

                if self._stop_event.is_set():
                    break
                if return_code != 0:
                    raise RuntimeError(
                        f"{stage.label} exited with code {return_code}."
                    )
                with self._lock:
                    self._completed_count = index + 1
                    self._append_log(f"[dashboard] completed · {stage.label}")

            with self._lock:
                self._ended_at = time.time()
                if self._stop_event.is_set():
                    self._status = "stopped"
                    self._message = "Pipeline stopped by user."
                    self._append_log("[dashboard] pipeline stopped")
                else:
                    self._status = "completed"
                    self._message = "All pipeline stages completed."
                    self._current_index = len(self._pipeline)
                    self._append_log("[dashboard] pipeline complete")
        except Exception as exc:
            with self._lock:
                self._process = None
                self._ended_at = time.time()
                self._status = "failed"
                self._message = str(exc)
                self._append_log(f"[dashboard] ERROR · {exc}")


def read_metrics_file(path: Path, stage_key: str, stage_label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as metrics_file:
            for line_number, line in enumerate(metrics_file, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or not isinstance(row.get("step"), int):
                    continue
                row["_stage"] = stage_key
                row["_stage_label"] = stage_label
                row["_line"] = line_number
                rows.append(row)
    except OSError:
        return []
    return rows


def read_all_metrics(stages: Iterable[Stage]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in stages:
        if stage.metrics_path is not None:
            rows.extend(read_metrics_file(stage.metrics_path, stage.key, stage.label))
    return rows


def _base_figure(title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=325,
        margin={"l": 52, "r": 25, "t": 58, "b": 44},
        paper_bgcolor="rgba(4, 7, 20, 0)",
        plot_bgcolor="rgba(10, 15, 35, 0.72)",
        font={"color": "#dbe7ff", "family": "Inter, ui-sans-serif, system-ui"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 10},
        },
        hovermode="x unified",
        xaxis_title="Step",
        yaxis_title=y_title,
    )
    figure.update_xaxes(gridcolor="rgba(148, 163, 184, 0.12)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(148, 163, 184, 0.12)", zeroline=False)
    return figure


def _empty_figure(figure: go.Figure, message: str = "Waiting for training metrics") -> go.Figure:
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "#8190b5", "size": 14},
    )
    return figure


def _stage_rows(
    rows: Sequence[dict[str, Any]], stage_key: str, field: str
) -> tuple[list[int], list[float]]:
    selected = [
        row
        for row in rows
        if row.get("_stage") == stage_key and isinstance(row.get(field), (int, float))
    ]
    return (
        [int(row["step"]) for row in selected],
        [float(row[field]) for row in selected],
    )


def build_loss_figure(rows: Sequence[dict[str, Any]], stages: Sequence[Stage]) -> go.Figure:
    figure = _base_figure("Loss trajectory", "Cross-entropy")
    traces = 0
    for stage in stages:
        color = STAGE_COLORS.get(stage.key, "#94a3b8")
        for field, suffix, dash in (
            ("train_loss", "train", "solid"),
            ("val_loss", "validation", "dot"),
        ):
            steps, values = _stage_rows(rows, stage.key, field)
            if not steps:
                continue
            traces += 1
            figure.add_trace(
                go.Scatter(
                    x=steps,
                    y=values,
                    mode="lines+markers" if field == "val_loss" else "lines",
                    name=f"{stage.label} · {suffix}",
                    line={"color": color, "width": 2.2, "dash": dash},
                    marker={"size": 5, "color": color},
                )
            )
    return figure if traces else _empty_figure(figure)


def build_lr_figure(rows: Sequence[dict[str, Any]], stages: Sequence[Stage]) -> go.Figure:
    figure = _base_figure("Learning-rate schedule", "Learning rate")
    traces = 0
    for stage in stages:
        steps, values = _stage_rows(rows, stage.key, "lr")
        if not steps:
            continue
        traces += 1
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode="lines",
                name=stage.label,
                line={"color": STAGE_COLORS.get(stage.key, "#94a3b8"), "width": 2.2},
                fill="tozeroy",
                fillcolor="rgba(124, 131, 255, 0.08)",
            )
        )
    figure.update_yaxes(tickformat=".2e")
    return figure if traces else _empty_figure(figure)


def build_throughput_figure(
    rows: Sequence[dict[str, Any]], stages: Sequence[Stage]
) -> go.Figure:
    figure = _base_figure("Training throughput", "Tokens / second")
    traces = 0
    for stage in stages:
        steps, values = _stage_rows(rows, stage.key, "tokens_per_sec")
        if not steps:
            continue
        traces += 1
        figure.add_trace(
            go.Scatter(
                x=steps,
                y=values,
                mode="lines",
                name=stage.label,
                line={"color": STAGE_COLORS.get(stage.key, "#94a3b8"), "width": 2},
            )
        )
    return figure if traces else _empty_figure(figure)


def build_router_figure(
    rows: Sequence[dict[str, Any]], stages: Sequence[Stage]
) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    traces = 0
    for stage in stages:
        color = STAGE_COLORS.get(stage.key, "#94a3b8")
        lb_steps, lb_values = _stage_rows(rows, stage.key, "router_load_balance_loss")
        z_steps, z_values = _stage_rows(rows, stage.key, "router_z_loss")
        if lb_steps:
            traces += 1
            figure.add_trace(
                go.Scatter(
                    x=lb_steps,
                    y=lb_values,
                    mode="lines",
                    name=f"{stage.label} · balance",
                    line={"color": color, "width": 2},
                ),
                secondary_y=False,
            )
        if z_steps:
            traces += 1
            figure.add_trace(
                go.Scatter(
                    x=z_steps,
                    y=z_values,
                    mode="lines",
                    name=f"{stage.label} · router z",
                    line={"color": color, "width": 1.5, "dash": "dot"},
                    opacity=0.75,
                ),
                secondary_y=True,
            )
    figure.update_layout(
        title={"text": "MoE router health", "x": 0.02, "xanchor": "left"},
        height=325,
        margin={"l": 52, "r": 52, "t": 58, "b": 44},
        paper_bgcolor="rgba(4, 7, 20, 0)",
        plot_bgcolor="rgba(10, 15, 35, 0.72)",
        font={"color": "#dbe7ff", "family": "Inter, ui-sans-serif, system-ui"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 10},
        },
        hovermode="x unified",
    )
    figure.update_xaxes(
        title_text="Step",
        gridcolor="rgba(148, 163, 184, 0.12)",
        zeroline=False,
    )
    figure.update_yaxes(
        title_text="Load balance",
        gridcolor="rgba(148, 163, 184, 0.12)",
        zeroline=False,
        secondary_y=False,
    )
    figure.update_yaxes(title_text="Router z-loss", secondary_y=True)
    return figure if traces else _empty_figure(figure)


def _latest_metric(
    rows: Sequence[dict[str, Any]], current_stage: Stage | None
) -> dict[str, Any] | None:
    if current_stage is not None:
        matching = [row for row in rows if row.get("_stage") == current_stage.key]
        if matching:
            return matching[-1]
    return rows[-1] if rows else None


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _number(value: Any, digits: int = 4, fallback: str = "—") -> str:
    if not isinstance(value, (int, float)):
        return fallback
    return f"{float(value):,.{digits}f}"


def _progress_fraction(
    snapshot: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> tuple[float, float, int, int]:
    pipeline: list[Stage] = snapshot["pipeline"]
    current: Stage | None = snapshot["current_stage"]
    completed = int(snapshot["completed_count"])
    if snapshot["status"] == "completed":
        return 1.0, 1.0, len(pipeline), len(pipeline)

    stage_fraction = 0.0
    current_step = 0
    max_steps = current.max_steps if current is not None else 0
    if current is not None and current.kind == "training" and max_steps > 0:
        matching = [row for row in rows if row.get("_stage") == current.key]
        if matching:
            current_step = max(int(row["step"]) for row in matching)
            stage_fraction = min(1.0, current_step / max_steps)

    overall = 0.0
    if pipeline:
        overall = min(1.0, (completed + stage_fraction) / len(pipeline))
    return overall, stage_fraction, current_step, max_steps


def status_html(snapshot: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    status = str(snapshot["status"])
    current: Stage | None = snapshot["current_stage"]
    latest = _latest_metric(rows, current) or {}
    overall, stage_fraction, current_step, max_steps = _progress_fraction(snapshot, rows)
    stage_name = current.label if current is not None else "No active stage"
    loss = latest.get("train_loss")
    if loss is None:
        loss = latest.get("val_loss")
    throughput = latest.get("tokens_per_sec")
    message = html.escape(str(snapshot["message"]))
    stage_name = html.escape(stage_name)
    status_text = html.escape(status.upper())

    return f"""
    <section class="live-shell">
      <div class="live-topline">
        <span class="status-pill status-{status}"><span class="pulse-dot"></span>{status_text}</span>
        <span class="live-message">{message}</span>
      </div>
      <div class="kpi-grid">
        <div class="kpi-card"><span>Active stage</span><strong>{stage_name}</strong></div>
        <div class="kpi-card"><span>Step</span><strong>{current_step:,} <small>/ {max_steps:,}</small></strong></div>
        <div class="kpi-card"><span>Latest loss</span><strong>{_number(loss)}</strong></div>
        <div class="kpi-card"><span>Throughput</span><strong>{_number(throughput, 0)} <small>tok/s</small></strong></div>
        <div class="kpi-card"><span>Elapsed</span><strong>{_format_duration(float(snapshot["elapsed_seconds"]))}</strong></div>
      </div>
      <div class="progress-copy">
        <span>Overall pipeline</span><b>{overall * 100:.1f}%</b>
      </div>
      <div class="progress-track"><div class="progress-fill" style="width:{overall * 100:.2f}%"></div></div>
      <div class="progress-copy stage-copy">
        <span>Current stage</span><b>{stage_fraction * 100:.1f}%</b>
      </div>
      <div class="progress-track thin"><div class="progress-fill secondary" style="width:{stage_fraction * 100:.2f}%"></div></div>
    </section>
    """


def pipeline_html(snapshot: dict[str, Any]) -> str:
    stages: list[Stage] = snapshot["pipeline"]
    current_index = int(snapshot["current_index"])
    completed = int(snapshot["completed_count"])
    if not stages:
        return '<div class="empty-state">Choose a preset and launch when ready.</div>'
    cards: list[str] = []
    for index, stage in enumerate(stages):
        if index < completed:
            state, icon = "done", "✓"
        elif index == current_index and snapshot["running"]:
            state, icon = "active", "◉"
        elif index == current_index and snapshot["status"] == "failed":
            state, icon = "failed", "!"
        else:
            state, icon = "queued", str(index + 1)
        detail = (
            f"{stage.block_size // 1024}K context · {stage.max_steps:,} steps"
            if stage.kind == "training"
            else "Preparation"
        )
        cards.append(
            f'<div class="stage-card {state}"><i>{icon}</i>'
            f"<div><b>{html.escape(stage.label)}</b>"
            f"<span>{html.escape(detail)}</span></div></div>"
        )
    return '<div class="stage-list">' + "".join(cards) + "</div>"


def checkpoint_rows(stages: Sequence[Stage]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    seen: set[Path] = set()
    for stage in stages:
        if stage.output_dir is None or not stage.output_dir.exists():
            continue
        for path in stage.output_dir.glob("*.safetensors"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                stat = resolved.stat()
            except OSError:
                continue
            try:
                display_path = str(resolved.relative_to(ROOT))
            except ValueError:
                display_path = str(resolved)
            rows.append(
                [
                    stage.label,
                    display_path,
                    round(stat.st_size / (1024 * 1024), 2),
                    datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )
    rows.sort(key=lambda row: row[3], reverse=True)
    return rows[:40]


def dashboard_snapshot(
    controller: TrainingController,
    training_stages: Sequence[Stage],
    preview_pipeline: Sequence[Stage] | None = None,
) -> tuple[
    str,
    str,
    str,
    go.Figure,
    go.Figure,
    go.Figure,
    go.Figure,
    list[list[Any]],
]:
    snapshot = controller.snapshot()
    display_snapshot = snapshot
    if not snapshot["pipeline"] and preview_pipeline:
        display_snapshot = dict(snapshot)
        display_snapshot["pipeline"] = list(preview_pipeline)
    rows = read_all_metrics(training_stages)
    logs = snapshot["logs"] or "[dashboard] Logs will appear here after launch."
    return (
        status_html(display_snapshot, rows),
        pipeline_html(display_snapshot),
        logs,
        build_loss_figure(rows, training_stages),
        build_throughput_figure(rows, training_stages),
        build_lr_figure(rows, training_stages),
        build_router_figure(rows, training_stages),
        checkpoint_rows(training_stages),
    )


DASHBOARD_CSS = """
:root {
  --tl-bg: #050816;
  --tl-panel: rgba(11, 17, 38, .82);
  --tl-stroke: rgba(148, 163, 184, .16);
  --tl-text: #e7edff;
  --tl-muted: #8b9abd;
  --tl-indigo: #7c83ff;
  --tl-cyan: #22d3ee;
  --tl-green: #34d399;
}
body, .gradio-container {
  background:
    radial-gradient(circle at 10% -10%, rgba(99, 102, 241, .26), transparent 32rem),
    radial-gradient(circle at 95% 10%, rgba(6, 182, 212, .16), transparent 28rem),
    var(--tl-bg) !important;
  color: var(--tl-text) !important;
}
.gradio-container { max-width: 1480px !important; padding: 24px !important; }
#hero {
  position: relative; overflow: hidden; padding: 30px 32px; margin-bottom: 18px;
  border: 1px solid rgba(124, 131, 255, .28); border-radius: 24px;
  background: linear-gradient(120deg, rgba(20, 28, 62, .94), rgba(8, 17, 38, .88));
  box-shadow: 0 24px 80px rgba(0, 0, 0, .28);
}
#hero:after {
  content: ""; position: absolute; width: 260px; height: 260px; right: -70px; top: -100px;
  border-radius: 50%; background: rgba(34, 211, 238, .15); filter: blur(8px);
}
#hero .eyebrow { color: #7dd3fc; letter-spacing: .16em; font-size: .72rem; font-weight: 800; }
#hero h1 { margin: 7px 0 5px; font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1; letter-spacing: -.04em; }
#hero h1 span { color: #a5b4fc; }
#hero p { color: #aab8d8; max-width: 760px; font-size: 1rem; margin: 10px 0 0; }
#hero .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 17px; }
#hero .badges span {
  border: 1px solid rgba(165, 180, 252, .2); color: #c7d2fe; background: rgba(49, 46, 129, .24);
  padding: 5px 10px; border-radius: 999px; font-size: .76rem; font-weight: 700;
}
.live-shell, #control-panel, #pipeline-panel, .chart-panel, #log-panel, #checkpoint-panel {
  border: 1px solid var(--tl-stroke) !important; border-radius: 18px !important;
  background: var(--tl-panel) !important; box-shadow: 0 16px 48px rgba(0, 0, 0, .18) !important;
}
.live-shell { padding: 18px; margin: 0 0 12px; }
.live-topline { display: flex; gap: 12px; align-items: center; margin-bottom: 14px; }
.live-message { color: var(--tl-muted); font-size: .9rem; }
.status-pill {
  display: inline-flex; align-items: center; gap: 7px; border-radius: 999px; padding: 6px 10px;
  font-size: .7rem; letter-spacing: .08em; font-weight: 900; color: #cbd5e1;
  background: rgba(100, 116, 139, .16); border: 1px solid rgba(148, 163, 184, .18);
}
.pulse-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.status-running, .status-stopping { color: #67e8f9; background: rgba(8, 145, 178, .14); }
.status-running .pulse-dot { animation: pulse 1.25s infinite; }
.status-completed { color: #6ee7b7; background: rgba(5, 150, 105, .13); }
.status-failed { color: #fda4af; background: rgba(225, 29, 72, .13); }
.status-stopped { color: #fcd34d; background: rgba(217, 119, 6, .13); }
@keyframes pulse { 50% { opacity: .35; transform: scale(.7); } }
.kpi-grid { display: grid; grid-template-columns: 1.7fr repeat(4, 1fr); gap: 10px; }
.kpi-card {
  min-height: 76px; padding: 13px 14px; border-radius: 13px; background: rgba(7, 12, 29, .76);
  border: 1px solid rgba(148, 163, 184, .11); display: flex; flex-direction: column; justify-content: center;
}
.kpi-card span { color: var(--tl-muted); font-size: .71rem; text-transform: uppercase; letter-spacing: .07em; font-weight: 700; }
.kpi-card strong { color: #f1f5ff; font-size: 1.05rem; margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi-card small { color: #7f8ead; font-size: .72rem; font-weight: 600; }
.progress-copy { display: flex; justify-content: space-between; color: #aab8d8; font-size: .76rem; margin: 14px 2px 6px; }
.stage-copy { margin-top: 9px; }
.progress-track { height: 9px; background: #11182d; border-radius: 999px; overflow: hidden; }
.progress-track.thin { height: 5px; }
.progress-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--tl-indigo), var(--tl-cyan)); transition: width .35s ease; }
.progress-fill.secondary { background: linear-gradient(90deg, #34d399, #22d3ee); }
#control-panel, #pipeline-panel, #log-panel, #checkpoint-panel { padding: 14px !important; }
#start-button {
  min-height: 48px; font-weight: 800 !important; letter-spacing: .01em;
  background: linear-gradient(100deg, #6366f1, #0891b2) !important; border: 0 !important;
  box-shadow: 0 10px 28px rgba(79, 70, 229, .28) !important;
}
.stage-list { display: grid; gap: 8px; }
.stage-card {
  display: flex; gap: 11px; align-items: center; padding: 10px 11px; border-radius: 12px;
  background: rgba(6, 11, 28, .68); border: 1px solid rgba(148, 163, 184, .1);
}
.stage-card i {
  display: grid; place-items: center; width: 28px; height: 28px; flex: 0 0 28px;
  border-radius: 9px; font-style: normal; color: #8190b5; background: rgba(51, 65, 85, .34); font-size: .75rem; font-weight: 900;
}
.stage-card div { min-width: 0; display: flex; flex-direction: column; }
.stage-card b { color: #dce6ff; font-size: .83rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.stage-card span { color: #7281a5; font-size: .7rem; margin-top: 2px; }
.stage-card.done i { color: #6ee7b7; background: rgba(5, 150, 105, .16); }
.stage-card.active { border-color: rgba(34, 211, 238, .42); background: rgba(8, 145, 178, .08); }
.stage-card.active i { color: #67e8f9; background: rgba(8, 145, 178, .19); animation: pulse 1.25s infinite; }
.stage-card.failed i { color: #fda4af; background: rgba(225, 29, 72, .16); }
.empty-state { color: #7180a1; text-align: center; padding: 30px 12px; }
.chart-panel { overflow: hidden; }
.chart-panel > div { border: 0 !important; background: transparent !important; }
#log-panel textarea, #log-panel pre {
  background: #030611 !important; color: #9ee7ff !important; font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace !important;
}
#action-notice { color: #a5b4fc; min-height: 24px; }
footer { display: none !important; }
@media (max-width: 980px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .kpi-card:first-child { grid-column: 1 / -1; }
  .gradio-container { padding: 12px !important; }
}
"""


def build_app(controller: TrainingController | None = None) -> gr.Blocks:
    controller = controller or TrainingController(ROOT)
    training_stages = list(load_training_stages(controller.root).values())
    initial_preview = build_pipeline(
        FULL_CURRICULUM,
        refresh_public_data=False,
        rebuild_tokenizer_and_data=False,
        auto_resume=True,
        root=controller.root,
    )
    initial = dashboard_snapshot(controller, training_stages, initial_preview)

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate",
    )
    with gr.Blocks(
        title="Tiny-LLM · MoE Training Control Center",
        theme=theme,
        css=DASHBOARD_CSS,
        fill_width=True,
    ) as demo:
        gr.HTML(
            """
            <section id="hero">
              <div class="eyebrow">TINY-LLM TRAINING CONTROL CENTER</div>
              <h1>Train the sparse MoE.<br><span>Watch every signal.</span></h1>
              <p>One click runs the resumable 2K → 4K → 16K context curriculum and reasoning/tool SFT. Live telemetry is read directly from trainer metrics.</p>
              <div class="badges"><span>TOP-2 / 4 EXPERTS</span><span>16K CONTEXT</span><span>SAFETENSORS</span><span>AUTO-RESUME</span></div>
            </section>
            """
        )

        live_status = gr.HTML(initial[0])

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_id="control-panel"):
                gr.Markdown("### Launch")
                preset = gr.Dropdown(
                    choices=list(PRESETS),
                    value=FULL_CURRICULUM,
                    label="Training preset",
                    info="The full curriculum is the safe default for a fresh or resumed model.",
                    interactive=True,
                )
                auto_resume = gr.Checkbox(
                    value=True,
                    label="Resume latest SafeTensors checkpoints",
                    info="Each stage resumes its own latest checkpoint when available.",
                )
                refresh_public = gr.Checkbox(
                    value=False,
                    label="Refresh pinned public datasets first",
                    info="Downloads only revision-checked public sources.",
                )
                rebuild_inputs = gr.Checkbox(
                    value=False,
                    label="Rebuild tokenizer and packed arrays",
                    info="Use after dataset changes; existing artifacts are reused otherwise.",
                )
                with gr.Row():
                    start_button = gr.Button(
                        "▶  Start 1-click training",
                        variant="primary",
                        elem_id="start-button",
                        scale=3,
                    )
                    stop_button = gr.Button("■  Stop", variant="stop", scale=1)
                action_notice = gr.Markdown(
                    "Ready. No training starts until you click the launch button.",
                    elem_id="action-notice",
                )
            with gr.Column(scale=3, elem_id="pipeline-panel"):
                gr.Markdown("### Pipeline")
                pipeline_view = gr.HTML(initial[1])

        gr.Markdown("## Live telemetry")
        with gr.Row():
            loss_plot = gr.Plot(initial[3], show_label=False, elem_classes="chart-panel")
            throughput_plot = gr.Plot(
                initial[4], show_label=False, elem_classes="chart-panel"
            )
        with gr.Row():
            lr_plot = gr.Plot(initial[5], show_label=False, elem_classes="chart-panel")
            router_plot = gr.Plot(
                initial[6], show_label=False, elem_classes="chart-panel"
            )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_id="log-panel"):
                gr.Markdown("### Streaming process log")
                logs = gr.Code(
                    value=initial[2],
                    language="shell",
                    lines=20,
                    max_lines=28,
                    interactive=False,
                    show_label=False,
                    wrap_lines=True,
                )
            with gr.Column(scale=4, elem_id="checkpoint-panel"):
                gr.Markdown("### SafeTensors checkpoints")
                checkpoints = gr.Dataframe(
                    value=initial[7],
                    headers=("Stage", "Checkpoint", "Size (MiB)", "Modified"),
                    datatype=("str", "str", "number", "str"),
                    type="array",
                    interactive=False,
                    max_height=470,
                    wrap=True,
                    show_label=False,
                )

        def launch_pipeline(
            selected_preset: str,
            should_refresh: bool,
            should_rebuild: bool,
            should_resume: bool,
        ) -> str:
            ok, message = controller.start(
                selected_preset,
                should_refresh,
                should_rebuild,
                should_resume,
            )
            icon = "✅" if ok else "⚠️"
            return f"{icon} {message}"

        def stop_pipeline() -> str:
            ok, message = controller.stop()
            icon = "⏹️" if ok else "ℹ️"
            return f"{icon} {message}"

        def refresh_dashboard(
            selected_preset: str,
            should_refresh: bool,
            should_rebuild: bool,
            should_resume: bool,
        ) -> tuple[
            str,
            str,
            str,
            go.Figure,
            go.Figure,
            go.Figure,
            go.Figure,
            list[list[Any]],
        ]:
            preview: list[Stage] = []
            try:
                preview = build_pipeline(
                    selected_preset,
                    should_refresh,
                    should_rebuild,
                    should_resume,
                    root=controller.root,
                )
            except (OSError, KeyError, TypeError, ValueError):
                pass
            return dashboard_snapshot(controller, training_stages, preview)

        start_button.click(
            fn=launch_pipeline,
            inputs=(preset, refresh_public, rebuild_inputs, auto_resume),
            outputs=action_notice,
            queue=False,
            show_progress="hidden",
            api_name=False,
            show_api=False,
        )
        stop_button.click(
            fn=stop_pipeline,
            outputs=action_notice,
            queue=False,
            show_progress="hidden",
            api_name=False,
            show_api=False,
        )

        timer = gr.Timer(1.0)
        timer.tick(
            fn=refresh_dashboard,
            inputs=(preset, refresh_public, rebuild_inputs, auto_resume),
            outputs=(
                live_status,
                pipeline_view,
                logs,
                loss_plot,
                throughput_plot,
                lr_plot,
                router_plot,
                checkpoints,
            ),
            queue=False,
            show_progress="hidden",
            api_name=False,
            show_api=False,
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Tiny-LLM one-click training and telemetry dashboard."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary Gradio share link. Off by default.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        help="Open the dashboard in the default browser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_app()
    demo.queue(default_concurrency_limit=8)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_api=False,
        show_error=True,
        blocked_paths=[
            str((ROOT / "data").resolve()),
            str((ROOT / "checkpoints").resolve()),
        ],
    )


if __name__ == "__main__":
    main()
