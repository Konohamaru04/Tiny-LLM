from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import resolve_path, sha256_file, write_json


VIEWER_BASE = "https://datasets-server.huggingface.co"
HUB_API_BASE = "https://huggingface.co/api/datasets"


def _read_json_url(url: str, timeout: int, retries: int = 7) -> dict[str, Any]:
    headers = {"User-Agent": "Tiny-LLM-dataset-downloader/1.0"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt + 1 < retries:
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(15 * (attempt + 1), 60)
                print(f"[rate-limit] waiting {delay:.0f}s before retry")
                time.sleep(delay)
                continue
            if attempt + 1 < retries and 500 <= exc.code < 600:
                time.sleep(min(2**attempt, 30))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed to download JSON after {retries} attempts: {url}") from last_error


def fetch_dataset_metadata(dataset: str, timeout: int) -> dict[str, Any]:
    url = f"{HUB_API_BASE}/{urllib.parse.quote(dataset, safe='/')}"
    payload = _read_json_url(url, timeout)
    return {
        "dataset": dataset,
        "sha": str(payload.get("sha", "")),
        "last_modified": str(payload.get("lastModified", "")),
        "private": bool(payload.get("private", False)),
        "gated": payload.get("gated", False),
    }


def iter_viewer_rows(
    dataset: str,
    config: str,
    split: str,
    *,
    start_offset: int,
    scan_limit: int,
    timeout: int,
    request_delay_seconds: float,
) -> Iterable[tuple[int, dict[str, Any]]]:
    offset = start_offset
    end_offset = start_offset + scan_limit
    while offset < end_offset:
        length = min(100, end_offset - offset)
        params = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        payload = _read_json_url(f"{VIEWER_BASE}/rows?{params}", timeout)
        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            break
        for item in rows:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                row_index = int(item.get("row_idx", offset))
                yield row_index, row
        offset += len(rows)
        if len(rows) < length:
            break


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        if isinstance(node, ast.Name):
            return node.id
        return ast.unparse(node)


def parse_python_function_calls(raw: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if not isinstance(raw, str) or not raw.strip():
        return calls
    raw = raw.strip()
    raw = re.sub(r"^<function_calls>\s*", "", raw)
    raw = re.sub(r"\s*</function_calls>$", "", raw)

    decoded = _parse_json_maybe(raw, None)
    if isinstance(decoded, Mapping):
        decoded = [decoded]
    if isinstance(decoded, list):
        for index, item in enumerate(decoded):
            if not isinstance(item, Mapping):
                continue
            function = item.get("function") if isinstance(item.get("function"), Mapping) else item
            name = str(function.get("name", "")).strip()
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = _parse_json_maybe(arguments, {"raw": arguments})
            if name:
                calls.append(
                    {
                        "id": str(item.get("id", "")).strip() or f"call_{index + 1}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                arguments,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                )
        if calls:
            return calls

    for index, line in enumerate(filter(str.strip, raw.splitlines())):
        name = ""
        call: ast.Call | None = None
        try:
            expression = ast.parse(line, mode="eval").body
        except SyntaxError:
            expression = None
        if isinstance(expression, ast.Call):
            call = expression
            if isinstance(call.func, ast.Attribute):
                name = ast.unparse(call.func)
            elif isinstance(call.func, ast.Name):
                name = call.func.id
        if call is None or not name:
            match = re.fullmatch(r"\s*([A-Za-z0-9_.-]+)\((.*)\)\s*", line)
            if not match:
                continue
            name, raw_arguments = match.groups()
            try:
                parsed_call = ast.parse(f"_tool_({raw_arguments})", mode="eval").body
            except SyntaxError:
                continue
            if not isinstance(parsed_call, ast.Call):
                continue
            call = parsed_call
        arguments = {keyword.arg: _literal(keyword.value) for keyword in call.keywords if keyword.arg}
        if call.args:
            arguments["_args"] = [_literal(arg) for arg in call.args]
        calls.append(
            {
                "id": f"call_{index + 1}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        )
    return calls


def _parse_json_maybe(value: Any, fallback: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value if value is not None else fallback


def transform_openr1(row: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any] | None:
    generations = row.get("generations")
    complete = row.get("is_reasoning_complete")
    correct = row.get("correctness_math_verify")
    chosen = ""
    if isinstance(generations, list):
        for index, generation in enumerate(generations):
            is_complete = not isinstance(complete, list) or index >= len(complete) or bool(complete[index])
            is_correct = not isinstance(correct, list) or index >= len(correct) or bool(correct[index])
            if isinstance(generation, str) and generation.strip() and is_complete and is_correct:
                chosen = generation.strip()
                break
    if not chosen:
        solution = row.get("solution")
        if isinstance(solution, str):
            chosen = solution.strip()
    problem = row.get("problem")
    if not isinstance(problem, str) or not problem.strip() or not chosen:
        return None

    reasoning = ""
    answer = chosen
    start = chosen.find("<think>")
    end = chosen.find("</think>")
    if start >= 0 and end > start:
        reasoning = chosen[start + len("<think>") : end].strip()
        answer = chosen[end + len("</think>") :].strip()
    assistant: dict[str, Any] = {"role": "assistant", "content": answer}
    if reasoning:
        assistant["reasoning_content"] = reasoning

    source_id = str(row.get("uuid", "")).strip() or hashlib.sha256(
        problem.encode("utf-8")
    ).hexdigest()[:20]
    return {
        "id": f"openr1_{source_id}",
        "messages": [
            {
                "role": "system",
                "content": "Solve the problem carefully. Show a concise, checkable reasoning process before the final answer.",
            },
            {"role": "user", "content": problem.strip()},
            assistant,
        ],
        "source": {
            "dataset": source["dataset"],
            "revision": source["revision"],
            "license": source["license"],
            "source_id": source_id,
        },
    }


def transform_dolci(row: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return None
    tools: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    pending_ids: list[str] = []
    assistant_signals = 0

    for raw_message in raw_messages:
        if not isinstance(raw_message, Mapping):
            continue
        role = str(raw_message.get("role", "")).lower()
        content = raw_message.get("content")
        functions = _parse_json_maybe(raw_message.get("functions"), [])
        if isinstance(functions, list) and functions and not tools:
            tools = functions

        if role == "assistant":
            calls = parse_python_function_calls(str(raw_message.get("function_calls") or ""))
            pending_ids = [str(call["id"]) for call in calls]
            message: dict[str, Any] = {"role": "assistant", "content": content or ""}
            if calls:
                message["tool_calls"] = calls
            if calls or (isinstance(content, str) and content.strip()):
                assistant_signals += 1
            messages.append(message)
        elif role == "environment":
            call_id = pending_ids.pop(0) if pending_ids else ""
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content or "",
                }
            )
        elif role in {"system", "developer", "user"}:
            messages.append({"role": role, "content": content or ""})

    if assistant_signals == 0:
        return None
    source_id = str(row.get("id", "")).strip() or hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "id": f"dolci_{source_id}",
        "messages": messages,
        "tools": tools,
        "source": {
            "dataset": source["dataset"],
            "revision": source["revision"],
            "license": source["license"],
            "source_id": source_id,
        },
    }


def transform_tmax(row: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return None
    tools = _parse_json_maybe(row.get("tools"), [])
    if not isinstance(tools, list):
        tools = []
    messages: list[dict[str, Any]] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, Mapping):
            continue
        role = str(raw_message.get("role", "")).lower()
        if role == "environment":
            role = "tool"
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            continue
        message = dict(raw_message)
        message["role"] = role
        if role == "tool" and not message.get("tool_call_id"):
            ids = message.get("tool_call_ids")
            if isinstance(ids, list) and ids:
                message["tool_call_id"] = str(ids[0])
        messages.append(message)
    if not any(message.get("role") == "assistant" for message in messages):
        return None
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    source_id = str(metadata.get("run_id", "")).strip() or hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "id": f"tmax_{source_id}",
        "messages": messages,
        "tools": tools,
        "source": {
            "dataset": source["dataset"],
            "revision": source["revision"],
            "license": source["license"],
            "source_id": source_id,
        },
    }


SFT_TRANSFORMS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, Any] | None]] = {
    "openr1": transform_openr1,
    "dolci_tool": transform_dolci,
    "tmax": transform_tmax,
}


def _stable_validation_assignment(record_id: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < fraction


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def _pretrain_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.startswith("<!-- source:"):
        marker_end = text.find("-->")
        if marker_end >= 0:
            return text[marker_end + 3 :].lstrip("\r\n")
    return text


def deduplicate_and_compact_pretrain_files(directory: Path, source_name: str) -> int:
    seen: set[str] = set()
    unique_files: list[Path] = []
    for path in sorted(directory.glob(f"{source_name}_*.md")):
        signature = hashlib.sha256(_pretrain_body(path).encode("utf-8")).hexdigest()
        if signature in seen:
            path.unlink()
            continue
        seen.add(signature)
        unique_files.append(path)

    for index, path in enumerate(unique_files, start=1):
        target = directory / f"{source_name}_{index:06d}.md"
        if path != target:
            path.replace(target)
    return len(unique_files)


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict) or not isinstance(config.get("sources"), list):
        raise ValueError("Dataset config must contain a top-level 'sources' list")
    return config


def _validate_source_revision(source: dict[str, Any], metadata: dict[str, Any]) -> None:
    expected = str(source.get("revision", "")).strip()
    actual = metadata["sha"]
    if expected and expected != actual:
        raise RuntimeError(
            f"Dataset revision changed for {source['dataset']}.\n"
            f"configured={expected}\ncurrent={actual}\n"
            "Review the upstream changes, then update configs/public_datasets.yaml."
        )
    if metadata["private"] or metadata["gated"]:
        raise RuntimeError(f"Dataset is no longer ungated public data: {source['dataset']}")


def _repository_relative_path(path: str | Path) -> str:
    resolved = resolve_path(path)
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Manifest path must be inside the repository: {resolved}"
        ) from exc


def download(
    config_path: str | Path,
    *,
    seed_override: int | None = None,
    stage_filter: str = "all",
) -> dict[str, Any]:
    config = _load_config(config_path)
    timeout = int(config.get("request_timeout_seconds", 60))
    request_delay_seconds = float(config.get("request_delay_seconds", 1.5))
    seed = int(config.get("seed", 42) if seed_override is None else seed_override)
    validation_fraction = float(config.get("validation_fraction", 0.05))
    if not (0.0 < validation_fraction < 0.5):
        raise ValueError("validation_fraction must be between 0 and 0.5")

    sft_dir = resolve_path(config.get("sft_output_dir", "data/public_sft"))
    pretrain_dir = resolve_path(config.get("pretrain_output_dir", "data/raw/public"))
    sft_dir.mkdir(parents=True, exist_ok=True)
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sft_dir / "manifest.json"
    existing_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            loaded_manifest = json.load(manifest_file)
        if isinstance(loaded_manifest, dict):
            existing_manifest = loaded_manifest
    state_path = pretrain_dir / ".download_state.json"
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as state_file:
            download_state = json.load(state_file)
    else:
        download_state = {}
    if not isinstance(download_state, dict):
        download_state = {}

    all_sft: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    source_reports: list[dict[str, Any]] = []
    pretrain_source_names = [
        str(source.get("name"))
        for source in config["sources"]
        if isinstance(source, dict) and source.get("stage") == "pretrain"
    ]
    pretrain_count = sum(
        len(list(pretrain_dir.glob(f"{source_name}_*.md")))
        for source_name in pretrain_source_names
    )
    sft_processed = False

    for raw_source in config["sources"]:
        if not isinstance(raw_source, dict):
            raise ValueError("Each dataset source must be an object")
        source = dict(raw_source)
        stage = str(source["stage"])
        if stage_filter != "all" and stage != stage_filter:
            continue
        dataset = str(source["dataset"])
        metadata = fetch_dataset_metadata(dataset, timeout)
        _validate_source_revision(source, metadata)
        sft_processed = sft_processed or stage == "sft"
        limit = int(source["limit"])
        scan_limit = int(source.get("scan_limit", max(limit * 2, limit)))
        print(
            f"[source] {dataset}@{metadata['sha'][:12]} stage={stage} "
            f"target={limit}"
        )

        accepted = 0
        scanned = 0
        start_offset = int(source.get("start_offset", 0))
        if stage == "pretrain":
            existing_count = deduplicate_and_compact_pretrain_files(
                pretrain_dir,
                str(source["name"]),
            )
            pretrain_count = sum(
                len(list(pretrain_dir.glob(f"{source_name}_*.md")))
                for source_name in pretrain_source_names
            )
            accepted = min(existing_count, limit)
            source_state = download_state.get(str(source["name"]), {})
            saved_offset = (
                int(source_state.get("next_offset", 0))
                if isinstance(source_state, dict)
                else 0
            )
            start_offset = max(start_offset + accepted, saved_offset)
            print(f"[resume] found {accepted} existing documents for {source['name']}")
            if accepted >= limit:
                source_reports.append(
                    {
                        "name": source["name"],
                        "dataset": dataset,
                        "config": source["config"],
                        "split": source.get("split", "train"),
                        "license": source["license"],
                        "revision": metadata["sha"],
                        "last_modified": metadata["last_modified"],
                        "requested": limit,
                        "accepted": accepted,
                        "scanned": 0,
                    }
                )
                continue
        transform_name = str(source.get("transform", ""))
        transform = SFT_TRANSFORMS.get(transform_name)
        last_row_index = start_offset - 1
        for row_index, row in iter_viewer_rows(
            dataset,
            str(source["config"]),
            str(source.get("split", "train")),
            start_offset=start_offset,
            scan_limit=max(0, scan_limit - accepted),
            timeout=timeout,
            request_delay_seconds=request_delay_seconds,
        ):
            scanned += 1
            last_row_index = row_index
            if stage == "pretrain":
                text = row.get(str(source.get("text_field", "text")))
                if not isinstance(text, str):
                    continue
                text = text.replace("\x00", "").strip()
                min_chars = int(source.get("min_chars", 300))
                max_chars = int(source.get("max_chars", 50000))
                if len(text) < min_chars:
                    continue
                text = text[:max_chars]
                doc_id = str(row.get("id", pretrain_count + 1))
                header = (
                    f"<!-- source: {dataset}; revision: {metadata['sha']}; "
                    f"id: {doc_id} -->\n\n"
                )
                output_index = accepted + 1
                output_path = pretrain_dir / f"{source['name']}_{output_index:06d}.md"
                output_path.write_text(header + text + "\n", encoding="utf-8", newline="\n")
                pretrain_count += 1
                accepted += 1
            elif stage == "sft":
                if transform is None:
                    raise ValueError(f"Unknown SFT transform: {transform_name}")
                normalized_source = {
                    **source,
                    "revision": metadata["sha"],
                }
                record = transform(row, normalized_source)
                if record is None:
                    continue
                serialized = json.dumps(record, sort_keys=True, ensure_ascii=False)
                max_chars = int(source.get("max_chars", 250000))
                if len(serialized) > max_chars:
                    continue
                signature = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
                if signature in seen_hashes:
                    continue
                seen_hashes.add(signature)
                all_sft.append(record)
                accepted += 1
            else:
                raise ValueError(f"Unsupported dataset stage: {stage}")

            if accepted >= limit:
                break

            if stage == "pretrain" and scanned % 100 == 0:
                download_state[str(source["name"])] = {
                    "next_offset": last_row_index + 1,
                    "accepted": accepted,
                    "revision": metadata["sha"],
                }
                write_json(download_state, state_path)

        if stage == "pretrain":
            download_state[str(source["name"])] = {
                "next_offset": last_row_index + 1,
                "accepted": accepted,
                "revision": metadata["sha"],
            }
            write_json(download_state, state_path)

        source_reports.append(
            {
                "name": source["name"],
                "dataset": dataset,
                "config": source["config"],
                "split": source.get("split", "train"),
                "license": source["license"],
                "revision": metadata["sha"],
                "last_modified": metadata["last_modified"],
                "requested": limit,
                "accepted": accepted,
                "scanned": scanned,
            }
        )
        print(f"[source] accepted={accepted} scanned={scanned}")

    train_path = sft_dir / "train.jsonl"
    val_path = sft_dir / "validation.jsonl"
    if sft_processed:
        random.Random(seed).shuffle(all_sft)
        train_rows: list[dict[str, Any]] = []
        val_rows: list[dict[str, Any]] = []
        for record in all_sft:
            target = (
                val_rows
                if _stable_validation_assignment(str(record["id"]), seed, validation_fraction)
                else train_rows
            )
            target.append(record)
        if all_sft and not val_rows:
            val_rows.append(train_rows.pop())
        if all_sft and not train_rows:
            train_rows.append(val_rows.pop())
        _write_jsonl(train_path, train_rows)
        _write_jsonl(val_path, val_rows)
        train_count = len(train_rows)
        val_count = len(val_rows)
    else:
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                "Pretrain-only refresh requires existing SFT train/validation files."
            )
        train_count = int(existing_manifest.get("sft_train_records", 0))
        val_count = int(existing_manifest.get("sft_validation_records", 0))

    merged_reports = {
        str(report.get("name")): report
        for report in existing_manifest.get("sources", [])
        if isinstance(report, dict) and report.get("name")
    }
    for report in source_reports:
        merged_reports[str(report["name"])] = report
    manifest = {
        "config": _repository_relative_path(config_path),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "pretrain_documents": pretrain_count,
        "sft_train_records": train_count,
        "sft_validation_records": val_count,
        "sft_train_path": _repository_relative_path(train_path),
        "sft_validation_path": _repository_relative_path(val_path),
        "sft_train_sha256": sha256_file(train_path),
        "sft_validation_sha256": sha256_file(val_path),
        "sources": [
            merged_reports[name]
            for name in sorted(merged_reports)
        ],
    }
    write_json(manifest, manifest_path)
    print(f"[done] pretrain documents: {pretrain_count}")
    print(f"[done] SFT train/validation: {train_count}/{val_count}")
    print(f"[done] manifest: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download revision-checked public pretraining, reasoning, and tool-use data."
    )
    parser.add_argument(
        "--config",
        default="configs/public_datasets.yaml",
        help="Dataset source YAML.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional split seed override.")
    parser.add_argument(
        "--stage",
        choices=("all", "pretrain", "sft"),
        default="all",
        help="Refresh all sources or only one training stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download(args.config, seed_override=args.seed, stage_filter=args.stage)


if __name__ == "__main__":
    main()
