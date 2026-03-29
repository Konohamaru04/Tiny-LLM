# Dataset Generation Workflow

This workflow hardens supervised fine-tuning data creation before records are
merged into the main training set.

## What it does

- generates JSONL data in batches
- validates every line before append
- resumes from the highest existing `train_N` id
- removes duplicates by both ID and semantic signature
- logs malformed rows and duplicate drops separately
- computes dataset statistics for category, difficulty, tags, and answer shape

## Files

- `scripts/generate_sample_train.py`
- `scripts/validate_jsonl.py`
- `scripts/deduplicate_jsonl.py`
- `scripts/dataset_stats.py`
- `config/dataset_config.json`
- `.env.example`
- `sample_train.jsonl`

## Live GLM generation

1. Copy `.env.example` to `.env`.
2. Add your `GLM_API_KEY`.
3. Run:

```powershell
python scripts/generate_sample_train.py --config config/dataset_config.json --count 50
```

The script uses the official GLM chat completions endpoint and requests strict
JSON output so each batch can be validated before it is appended.

## Local starter batch

If you do not want to call the API yet, generate a deterministic starter batch:

```powershell
python scripts/generate_sample_train.py --config config/dataset_config.json --seed-only --count 50
```

This produces a clean `sample_train.jsonl` with `train_1` through `train_50`.

## Validation and cleanup

Validate the file:

```powershell
python scripts/validate_jsonl.py --config config/dataset_config.json --input sample_train.jsonl
```

Deduplicate the file:

```powershell
python scripts/deduplicate_jsonl.py --config config/dataset_config.json --input sample_train.jsonl --in-place
```

Generate stats:

```powershell
python scripts/dataset_stats.py --config config/dataset_config.json --input sample_train.jsonl
```

## Notes

- The generator assigns IDs locally, so resume behavior stays deterministic even
  if the upstream model returns malformed items.
- The staged `sample_train.jsonl` file is intentionally separate from the main
  `data/sft/` training set. That gives you room to review and curate records
  before folding them into the repo's training corpus.
