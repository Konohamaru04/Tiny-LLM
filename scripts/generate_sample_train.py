from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_generation import (
    append_jsonl_records,
    deduplicate_records,
    extract_highest_train_index,
    load_dataset_config,
    load_env_file,
    log_failures,
    make_seed_records,
    read_jsonl_records,
    record_signature,
    request_glm_batch,
    validate_record,
    write_jsonl_records,
)
from src.utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample_train JSONL data in batches.")
    parser.add_argument("--config", type=str, default="config/dataset_config.json", help="Path to dataset config JSON.")
    parser.add_argument("--env-file", type=str, default=".env", help="Optional .env file to load.")
    parser.add_argument("--count", type=int, default=0, help="Number of new records to generate.")
    parser.add_argument("--output", type=str, default="", help="Optional output JSONL path override.")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Generate deterministic local starter records instead of calling the GLM API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_dataset_config(args.config)
    load_env_file(args.env_file)

    output_path = args.output or cfg["output_path"]
    target_new_records = args.count or int(cfg["target_records"])

    existing_records, existing_failures = read_jsonl_records(output_path)
    clean_existing, removed_existing = deduplicate_records(existing_records)
    seen_signatures = {record_signature(record) for record in clean_existing}
    failure_log: list[dict] = []

    if existing_failures:
        failure_log.extend({"stage": "load_existing", **failure} for failure in existing_failures)
    if removed_existing:
        failure_log.extend({"stage": "dedupe_existing", **failure} for failure in removed_existing)

    if cfg.get("rewrite_output_with_clean_records", True):
        write_jsonl_records(output_path, clean_existing)

    next_index = extract_highest_train_index(clean_existing) + 1
    generated: list[dict] = []
    stalled_rounds = 0

    while len(generated) < target_new_records:
        batch_size = min(int(cfg["batch_size"]), target_new_records - len(generated))

        if args.seed_only:
            candidates = make_seed_records(
                start_index=next_index,
                count=batch_size,
                categories=list(cfg["categories"]),
                difficulties=list(cfg["difficulties"]),
                tag_pool=list(cfg["tag_pool"]),
            )
        else:
            candidates = request_glm_batch(
                cfg,
                batch_size=batch_size,
                recent_user_prompts=[record["user"] for record in clean_existing[-20:]] + [record["user"] for record in generated[-20:]],
            )

        accepted_this_round = 0
        for candidate in candidates:
            candidate_with_id = dict(candidate)
            candidate_with_id["id"] = f"train_{next_index}"
            is_valid, record, error_message = validate_record(candidate_with_id, require_id=True)
            if not is_valid or record is None:
                failure_log.append(
                    {
                        "stage": "validate_new",
                        "error": error_message or "validation_failed",
                        "raw": candidate,
                    }
                )
                continue

            signature = record_signature(record)
            if signature in seen_signatures:
                failure_log.append(
                    {
                        "stage": "dedupe_new",
                        "error": "duplicate_signature",
                        "raw": record,
                    }
                )
                continue

            generated.append(record)
            seen_signatures.add(signature)
            accepted_this_round += 1
            next_index += 1

            if len(generated) >= target_new_records:
                break

        if accepted_this_round == 0:
            stalled_rounds += 1
            if stalled_rounds >= int(cfg["max_retries"]):
                raise RuntimeError(
                    "Generation stalled without producing valid unique records. "
                    "Check the failure log for details."
                )
        else:
            stalled_rounds = 0

    if generated:
        append_jsonl_records(output_path, generated)
    if failure_log:
        log_failures(cfg["failure_log_path"], failure_log)

    print(f"[done] wrote {len(generated)} new records to {resolve_path(output_path)}")
    print(f"[info] next available id: train_{next_index}")
    print(f"[info] failure log: {resolve_path(cfg['failure_log_path'])}")


if __name__ == "__main__":
    main()
