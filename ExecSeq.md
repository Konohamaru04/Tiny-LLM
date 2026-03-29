# Tiny-LLM execution sequence

## 0. Generate raw Markdown and SFT data

Generates 180 `.md` documents in `data/raw/` **and** writes
`data/sft/sample_train.jsonl` + `data/sft/sample_val.jsonl` in one shot:

```
python scripts/regenerate_datasets.py
```

Custom counts:

```
python scripts/regenerate_datasets.py --raw-count 180 --train-count 1080 --val-count 180 --seed 11
```

## 1. Train the tokenizer

```
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
```

> **Check vocab size:** open `data/processed/tokenizer_meta.json` and read
> `actual_vocab_size`. If it differs from `model.vocab_size` in
> `configs/pretrain_tiny.yaml` and `configs/sft_tiny.yaml`, update both
> files to match before continuing.

## 2. Prepare packed pretraining arrays

```
python scripts/prepare_data.py --config configs/pretrain_tiny.yaml
```

## 3. Run pretraining

```
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml
```

Resume pretraining from the latest checkpoint:

```
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml --resume checkpoints/pretrain_tiny/latest.pt
```

## 4. Run supervised fine-tuning

```
python scripts/train_sft.py --config configs/sft_tiny.yaml
```

Resume SFT from the latest checkpoint:

```
python scripts/train_sft.py --config configs/sft_tiny.yaml --resume checkpoints/sft_tiny/latest.pt
```

## 5. Evaluate the checkpoint

```
python scripts/eval_checkpoint.py --chat-config configs/chat.yaml --sft-config configs/sft_tiny.yaml
```

Optional fuller evaluation:

```
python scripts/eval_checkpoint.py --chat-config configs/chat.yaml --sft-config configs/sft_tiny.yaml --pretrain-config configs/pretrain_tiny.yaml --prompts configs/eval_prompts.jsonl --repetition-penalty 1.05
```

## 6. Run the terminal chat interface

List personas:

```
python scripts/chat.py --config configs/chat.yaml --list-personas
```

Start chat:

```
python scripts/chat.py --config configs/chat.yaml
```

Start chat with a specific persona:

```
python scripts/chat.py --config configs/chat.yaml --persona coach
```

Start chat in JSON mode:

```
python scripts/chat.py --config configs/chat.yaml --json-mode
```

Start chat with a custom session file:

```
python scripts/chat.py --config configs/chat.yaml --session-file logs/my_chat.json
```

Disable streaming:

```
python scripts/chat.py --config configs/chat.yaml --no-stream
```

## 7. Run the web chat UI

```
python scripts/web_chat.py --config configs/chat.yaml
```

Start the web UI with a custom persona and session file:

```
python scripts/web_chat.py --config configs/chat.yaml --persona json_bot --session-file logs/browser_chat.json
```

Default web URL:

```
http://127.0.0.1:8000
```

---

## Optional: API-based SFT record generation

Uses the GLM API to generate extra records. Output goes to `sample_train.jsonl`
**at the repo root** — not inside `data/sft/`. After reviewing and deduplicating,
manually copy records you want into `data/sft/sample_train.jsonl`.

Requires `GLM_API_KEY` set in `.env`.

Generate seed records without the API:

```
python scripts/generate_sample_train.py --config config/dataset_config.json --seed-only --count 50
```

Validate, deduplicate, and inspect:

```
python scripts/validate_jsonl.py --config config/dataset_config.json --input sample_train.jsonl
python scripts/deduplicate_jsonl.py --config config/dataset_config.json --input sample_train.jsonl --in-place
python scripts/dataset_stats.py --config config/dataset_config.json --input sample_train.jsonl
```

Live API generation:

```
python scripts/generate_sample_train.py --config config/dataset_config.json --count 50
```
