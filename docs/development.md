# Development Guide

This repo now has a lightweight development workflow that does not depend on any
locally trained checkpoints.

## Setup

1. Create and activate a virtual environment.
2. Install runtime dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Run smoke tests

Use the standard-library test suite:

```powershell
python -m unittest discover -s tests -v
```

The tests cover:

- config validation
- tokenizer and data preparation helpers
- SFT dataset masking
- model forward and prompt building
- resumable SafeTensors checkpoint save/load with a tiny synthetic training run
- sparse MoE routing, router losses, and long-horizon task resume

## Canonical local workflow

```powershell
python scripts/regenerate_datasets.py
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
python scripts/prepare_data.py --config configs/pretrain_tiny.yaml
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml
python scripts/train_sft.py --config configs/sft_tiny.yaml
python scripts/eval_checkpoint.py --chat-config configs/chat.yaml --sft-config configs/sft_tiny.yaml
python scripts/chat.py --config configs/chat.yaml
```

## Notes

- The smoke tests intentionally build tiny temporary datasets so contributors can
  validate the repo without depending on ignored files under `data/processed/`
  or `checkpoints/`.
- CI runs the same `unittest` command on every push and pull request.
