# Tiny-LLM execution sequence

Run commands from the repository root.

## 1. Download the pinned public dataset snapshot

```powershell
python scripts/download_public_datasets.py --config configs/public_datasets.yaml --stage all
```

This prepares 5,000 FineWeb-Edu documents plus 3,300 reasoning/tool SFT
examples. The 297 additional local Markdown documents under `data/raw/` are
also included once; identical mirrored copies are removed by content hash.

## 2. Train the tokenizer

```powershell
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
```

The tokenizer learns from the pretraining train split and the public SFT
training split, including all chat, reasoning, and tool-control tokens.

## 3. Pack pretraining arrays

```powershell
python scripts/prepare_data.py --config configs/pretrain_tiny.yaml
```

## 4. Pretrain

```powershell
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml
```

Resume:

```powershell
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml --resume checkpoints/pretrain_modern/latest.pt
```

## 5. Supervised fine-tuning

```powershell
python scripts/train_sft.py --config configs/sft_tiny.yaml
```

Resume:

```powershell
python scripts/train_sft.py --config configs/sft_tiny.yaml --resume checkpoints/sft_modern/latest.pt
```

## 6. Evaluate

```powershell
python scripts/eval_checkpoint.py --chat-config configs/chat.yaml --sft-config configs/sft_tiny.yaml --pretrain-config configs/pretrain_tiny.yaml --prompts configs/eval_prompts.jsonl
```

## 7. Chat with thinking and tool use

Terminal:

```powershell
python scripts/chat.py --config configs/chat.yaml
```

Web:

```powershell
python scripts/web_chat.py --config configs/chat.yaml
```

The web UI defaults to `http://127.0.0.1:8000`. Tool mode provides a
restricted calculator and current-time lookup. The model may perform up to four
tool rounds, controlled by `configs/chat.yaml`.

Old checkpoints are architecture-incompatible with the modern GQA model and
8,192-token vocabulary. Complete pretraining and SFT before launching chat.
