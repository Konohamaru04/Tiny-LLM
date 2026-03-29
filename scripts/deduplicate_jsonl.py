from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_generation import deduplicate_records, load_dataset_config, log_failures, read_jsonl_records, write_jsonl_records
from src.utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove duplicate records from a JSONL SFT dataset.")
    parser.add_argument("--config", type=str, default="config/dataset_config.json", help="Path to dataset config JSON.")
    parser.add_argument("--input", type=str, default="", help="JSONL file to deduplicate.")
    parser.add_argument("--output", type=str, default="", help="Output path for deduplicated JSONL.")
    parser.add_argument("--in-place", action="store_true", help="Rewrite the input file instead of using a separate output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_dataset_config(args.config)
    input_path = args.input or cfg["output_path"]
    output_path = input_path if args.in_place else (args.output or cfg["deduped_output_path"])

    valid_records, failures = read_jsonl_records(input_path)
    unique_records, removals = deduplicate_records(valid_records)
    write_jsonl_records(output_path, unique_records)

    print(f"[info] input:  {resolve_path(input_path)}")
    print(f"[info] output: {resolve_path(output_path)}")
    print(f"[info] kept:   {len(unique_records)}")
    print(f"[info] removed duplicates: {len(removals)}")
    print(f"[info] invalid rows skipped: {len(failures)}")

    logged = [{"stage": "deduplicate_jsonl", **failure} for failure in failures]
    logged.extend({"stage": "deduplicate_jsonl", **removal} for removal in removals)
    if logged:
        log_failures(cfg["failure_log_path"], logged)
        print(f"[info] failure log: {resolve_path(cfg['failure_log_path'])}")


if __name__ == "__main__":
    main()
