from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib import error, request

from src.utils import ensure_parent_dir, resolve_path


ID_PATTERN = re.compile(r"^train_(\d+)$")

DEFAULT_CATEGORIES = [
    "machine_learning",
    "deep_learning",
    "transformers",
    "tokenization",
    "pytorch",
    "optimization",
    "evaluation",
    "data_quality",
    "local_inference",
    "instruction_tuning",
]

DEFAULT_DIFFICULTIES = ["beginner", "intermediate", "advanced"]

DEFAULT_TAG_POOL = [
    "llm",
    "pytorch",
    "training",
    "sft",
    "tokenizer",
    "evaluation",
    "dataset",
    "inference",
    "transformers",
    "debugging",
]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"Expected a string, got {type(value).__name__}")
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _slugify(value: str) -> str:
    value = _normalize_text(value).lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]", "", value)
    return re.sub(r"_+", "_", value).strip("_")


def load_dataset_config(path: str | Path = "config/dataset_config.json") -> dict[str, Any]:
    cfg_path = resolve_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Dataset config must be a JSON object: {cfg_path}")

    data.setdefault("output_path", "sample_train.jsonl")
    data.setdefault("failure_log_path", "logs/dataset_generation_failures.jsonl")
    data.setdefault("stats_output_path", "logs/dataset_stats.json")
    data.setdefault("deduped_output_path", "sample_train.deduped.jsonl")
    data.setdefault("api_base_url", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
    data.setdefault("model", "glm-4.7-flash")
    data.setdefault("batch_size", 10)
    data.setdefault("target_records", 50)
    data.setdefault("temperature", 0.7)
    data.setdefault("max_tokens", 4096)
    data.setdefault("timeout_seconds", 60)
    data.setdefault("max_retries", 3)
    data.setdefault("rewrite_output_with_clean_records", True)
    data.setdefault("categories", DEFAULT_CATEGORIES)
    data.setdefault("difficulties", DEFAULT_DIFFICULTIES)
    data.setdefault("tag_pool", DEFAULT_TAG_POOL)
    return data


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    env_path = resolve_path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
            if key:
                loaded[key] = os.environ.get(key, value)
    return loaded


def validate_record(raw: Any, require_id: bool = True) -> tuple[bool, dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return False, None, "record must be a JSON object"

    record: dict[str, Any] = {}

    if require_id:
        record_id = _normalize_text(raw.get("id", ""))
        if not ID_PATTERN.fullmatch(record_id):
            return False, None, "id must match the format train_N"
        record["id"] = record_id
    else:
        record["id"] = ""

    system = _normalize_text(raw.get("system", ""))
    user = _normalize_text(raw.get("user", ""))
    assistant = _normalize_text(raw.get("assistant", ""))
    category = _slugify(str(raw.get("category", "")))
    difficulty = _slugify(str(raw.get("difficulty", "")))

    if not user:
        return False, None, "user must be a non-empty string"
    if not assistant:
        return False, None, "assistant must be a non-empty string"
    if not category:
        return False, None, "category must be a non-empty string"
    if not difficulty:
        return False, None, "difficulty must be a non-empty string"

    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not tags:
        return False, None, "tags must be a non-empty list of strings"

    clean_tags: list[str] = []
    seen_tags: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str):
            return False, None, "tags must contain only strings"
        normalized_tag = _slugify(tag)
        if not normalized_tag:
            return False, None, "tags must not contain empty values"
        if normalized_tag not in seen_tags:
            seen_tags.add(normalized_tag)
            clean_tags.append(normalized_tag)

    record.update(
        {
            "system": system,
            "user": user,
            "assistant": assistant,
            "category": category,
            "difficulty": difficulty,
            "tags": clean_tags,
        }
    )
    return True, record, None


def record_signature(record: dict[str, Any]) -> str:
    payload = "\n".join(
        [
            _normalize_text(record.get("system", "")).lower(),
            _normalize_text(record.get("user", "")).lower(),
            _normalize_text(record.get("assistant", "")).lower(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl_records(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    file_path = resolve_path(path)
    if not file_path.exists():
        return [], []

    valid: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                parsed = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                failures.append(
                    {
                        "line_no": line_no,
                        "error": f"invalid_json: {exc}",
                        "raw": raw_line,
                    }
                )
                continue

            is_valid, record, error_message = validate_record(parsed, require_id=True)
            if not is_valid or record is None:
                failures.append(
                    {
                        "line_no": line_no,
                        "error": error_message or "validation_failed",
                        "raw": parsed,
                    }
                )
                continue

            valid.append(record)

    return valid, failures


def write_jsonl_records(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    file_path = resolve_path(path)
    ensure_parent_dir(file_path)
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return file_path


def append_jsonl_records(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    file_path = resolve_path(path)
    ensure_parent_dir(file_path)
    with file_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return file_path


def log_failures(path: str | Path, failures: Iterable[dict[str, Any]]) -> Path:
    file_path = resolve_path(path)
    ensure_parent_dir(file_path)
    with file_path.open("a", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")
    return file_path


def extract_highest_train_index(records: Iterable[dict[str, Any]]) -> int:
    highest = 0
    for record in records:
        match = ID_PATTERN.fullmatch(str(record.get("id", "")))
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


def deduplicate_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique_records: list[dict[str, Any]] = []
    removals: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_signatures: set[str] = set()

    for record in records:
        reasons: list[str] = []
        record_id = record["id"]
        signature = record_signature(record)

        if record_id in seen_ids:
            reasons.append("duplicate_id")
        if signature in seen_signatures:
            reasons.append("duplicate_signature")

        if reasons:
            removals.append(
                {
                    "id": record_id,
                    "reasons": reasons,
                    "user": record["user"],
                }
            )
            continue

        seen_ids.add(record_id)
        seen_signatures.add(signature)
        unique_records.append(record)

    return unique_records, removals


def compute_stats(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    category_counts = Counter(record["category"] for record in records)
    difficulty_counts = Counter(record["difficulty"] for record in records)
    tag_counts = Counter(tag for record in records for tag in record["tags"])
    json_style_answers = sum(1 for record in records if record["assistant"].lstrip().startswith("{"))
    wrapped_json_answers = sum(1 for record in records if "<|json|>" in record["assistant"])

    total_user_chars = sum(len(record["user"]) for record in records)
    total_assistant_chars = sum(len(record["assistant"]) for record in records)

    return {
        "total_records": len(records),
        "highest_train_id": extract_highest_train_index(records),
        "avg_user_chars": round(total_user_chars / len(records), 2) if records else 0.0,
        "avg_assistant_chars": round(total_assistant_chars / len(records), 2) if records else 0.0,
        "json_style_answers": json_style_answers,
        "wrapped_json_answers": wrapped_json_answers,
        "category_counts": dict(sorted(category_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "top_tags": tag_counts.most_common(10),
    }


def make_seed_records(
    start_index: int,
    count: int,
    categories: list[str] | None = None,
    difficulties: list[str] | None = None,
    tag_pool: list[str] | None = None,
) -> list[dict[str, Any]]:
    categories = categories or DEFAULT_CATEGORIES
    difficulties = difficulties or DEFAULT_DIFFICULTIES
    tag_pool = tag_pool or DEFAULT_TAG_POOL

    system_templates = [
        "You are a practical technical assistant who explains machine learning topics clearly.",
        "You are a concise assistant for local LLM development and PyTorch experiments.",
        "You are a careful assistant who gives direct answers before extra detail.",
        "You are a helpful assistant for dataset, training, and inference questions.",
    ]
    user_templates = [
        "Explain the practical role of {category} for a {difficulty} learner.",
        "Give a clear example of how {category} shows up in a tiny LLM workflow.",
        "What usually goes wrong with {category}, and how should a {difficulty} practitioner respond?",
        "How would you teach {category} to someone building a local GPT-style model?",
    ]
    assistant_templates = [
        "{category_title} matters because it affects whether a small language model learns cleanly and behaves predictably. For a {difficulty} learner, the best move is to connect the concept to one concrete pipeline stage, inspect the inputs and outputs, and keep evaluation honest.",
        "A practical way to think about {category_title} is as an engineering lever inside a tiny LLM project. In a {difficulty} setup, it usually helps to start with a simple baseline, measure the effect, and only then add complexity.",
        "Teams run into trouble with {category_title} when they chase bigger experiments before they understand the small ones. A safer {difficulty} approach is to document assumptions, watch validation behavior, and tighten the dataset before tuning the model.",
        "If you are building a local GPT-style assistant, {category_title} becomes useful when you want more reliable training or cleaner responses. For a {difficulty} workflow, use short feedback loops, reproducible configs, and visible failure cases.",
    ]

    records: list[dict[str, Any]] = []
    for offset in range(count):
        idx = start_index + offset
        position = max(0, idx - 1)
        category = categories[position % len(categories)]
        difficulty = difficulties[position % len(difficulties)]
        category_title = category.replace("_", " ")
        system = system_templates[position % len(system_templates)]
        user = user_templates[position % len(user_templates)].format(
            category=category_title,
            difficulty=difficulty,
        )
        assistant = assistant_templates[position % len(assistant_templates)].format(
            category_title=category_title,
            difficulty=difficulty,
        )
        tags = [
            category,
            difficulty,
            tag_pool[position % len(tag_pool)],
        ]
        records.append(
            {
                "id": f"train_{idx}",
                "system": system,
                "user": user,
                "assistant": assistant,
                "category": category,
                "difficulty": difficulty,
                "tags": tags,
            }
        )
    return records


def _build_generation_messages(
    batch_size: int,
    categories: list[str],
    difficulties: list[str],
    recent_user_prompts: list[str],
) -> list[dict[str, str]]:
    schema = {
        "items": [
            {
                "system": "string",
                "user": "string",
                "assistant": "string",
                "category": "one of the allowed categories",
                "difficulty": "one of the allowed difficulties",
                "tags": ["short_tag", "short_tag"],
            }
        ]
    }
    system_message = (
        "You create high-quality supervised fine-tuning records for a local technical assistant. "
        "Return only valid JSON that matches the requested schema. Do not include markdown fences. "
        "Keep answers grounded, concise, and useful for machine learning, PyTorch, tokenization, "
        "training, evaluation, or local inference."
    )
    user_payload = {
        "task": "Generate SFT dataset rows",
        "count": batch_size,
        "allowed_categories": categories,
        "allowed_difficulties": difficulties,
        "schema": schema,
        "constraints": [
            "Each item must be distinct in both user intent and assistant wording.",
            "Keep the assistant helpful and technically accurate.",
            "Do not include IDs; they will be assigned by the caller.",
            "Prefer realistic prompts about tiny LLM pipelines, debugging, evaluation, and training.",
            "Tags must be short lowercase identifiers.",
        ],
        "avoid_reusing_user_prompts": recent_user_prompts[-20:],
    }
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def request_glm_batch(
    config: dict[str, Any],
    batch_size: int,
    recent_user_prompts: list[str] | None = None,
) -> list[dict[str, Any]]:
    api_key = os.environ.get("GLM_API_KEY", "").strip()
    api_base_url = os.environ.get("GLM_BASE_URL", str(config["api_base_url"])).strip()
    model_name = os.environ.get("GLM_MODEL", str(config["model"])).strip()
    if not api_key:
        raise RuntimeError("GLM_API_KEY is not set. Add it to your environment or .env file.")

    payload = {
        "model": model_name,
        "messages": _build_generation_messages(
            batch_size=batch_size,
            categories=list(config["categories"]),
            difficulties=list(config["difficulties"]),
            recent_user_prompts=list(recent_user_prompts or []),
        ),
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "max_tokens": int(config["max_tokens"]),
        "temperature": float(config["temperature"]),
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = request.Request(
        api_base_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=float(config["timeout_seconds"])) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"GLM request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"GLM request failed: {exc}") from exc

    parsed = json.loads(raw)
    content = parsed["choices"][0]["message"]["content"]
    if isinstance(content, str):
        content = json.loads(content)

    if isinstance(content, dict):
        items = content.get("items")
    elif isinstance(content, list):
        items = content
    else:
        items = None

    if not isinstance(items, list):
        raise RuntimeError("GLM response did not contain an `items` list.")
    return items
