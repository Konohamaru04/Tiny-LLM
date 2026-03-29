from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_generation import compute_stats, load_dataset_config, read_jsonl_records
from src.utils import ensure_parent_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute stats for a JSONL SFT dataset.")
    parser.add_argument("--config", type=str, default="config/dataset_config.json", help="Path to dataset config JSON.")
    parser.add_argument("--input", type=str, default="", help="JSONL file to analyze.")
    parser.add_argument("--output", type=str, default="", help="Optional JSON file path for saved stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_dataset_config(args.config)
    input_path = args.input or cfg["output_path"]
    output_path = args.output or cfg["stats_output_path"]

    valid_records, failures = read_jsonl_records(input_path)
    stats = compute_stats(valid_records)
    stats["invalid_rows"] = len(failures)

    print(json.dumps(stats, indent=2, ensure_ascii=False))

    resolved_output = resolve_path(output_path)
    ensure_parent_dir(resolved_output)
    with resolved_output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[info] stats written to {resolved_output}")


if __name__ == "__main__":
    main()
