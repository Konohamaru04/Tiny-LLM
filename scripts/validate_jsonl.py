from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_generation import load_dataset_config, log_failures, read_jsonl_records, write_jsonl_records
from src.utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a JSONL SFT dataset.")
    parser.add_argument("--config", type=str, default="config/dataset_config.json", help="Path to dataset config JSON.")
    parser.add_argument("--input", type=str, default="", help="JSONL file to validate.")
    parser.add_argument("--write-clean", type=str, default="", help="Optional path for valid cleaned rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_dataset_config(args.config)
    input_path = args.input or cfg["output_path"]
    clean_path = args.write_clean

    valid_records, failures = read_jsonl_records(input_path)

    print(f"[info] input: {resolve_path(input_path)}")
    print(f"[info] valid rows:   {len(valid_records)}")
    print(f"[info] invalid rows: {len(failures)}")

    if clean_path:
        write_jsonl_records(clean_path, valid_records)
        print(f"[info] clean file: {resolve_path(clean_path)}")

    if failures:
        log_failures(cfg["failure_log_path"], [{"stage": "validate_jsonl", **failure} for failure in failures])
        print(f"[info] failure log: {resolve_path(cfg['failure_log_path'])}")


if __name__ == "__main__":
    main()
