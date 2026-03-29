Build a production-ready Python pipeline to generate `sample_train.jsonl` in batches using GLM-4.7-Flash.

Requirements:
- JSONL format only, one valid JSON object per line
- schema:
  id, system, user, assistant, category, difficulty, tags
- support resume from highest existing train_N id
- validate each line before append
- discard invalid JSON
- avoid duplicates
- log failures separately
- include batch generation, validator, dedupe, stats, config, README, .env.example, requirements.txt
- create a small starter batch of 50 records

Files to create:
- scripts/generate_sample_train.py
- scripts/validate_jsonl.py
- scripts/deduplicate_jsonl.py
- scripts/dataset_stats.py
- config/dataset_config.json
- .env.example
- requirements.txt
- README_dataset_generation.md
- sample_train.jsonl

First inspect the workspace, then implement all files.