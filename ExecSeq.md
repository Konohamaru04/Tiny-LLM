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

## 3. Pack the 2K pretraining arrays

```powershell
python scripts/prepare_data.py --config configs/pretrain_tiny.yaml
```

## 4. Pretrain the 2K MoE stage

```powershell
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml
```

Resume:

```powershell
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml --resume checkpoints/pretrain_moe/latest.safetensors
```

## 5. Continue RoPE context to 4K and then 16K

The context stages model-only warm start from the preceding SafeTensors
checkpoint. Optimizer and scheduler state intentionally restart at each larger
context length.

```powershell
python scripts/prepare_data.py --config configs/pretrain_long_context_4k.yaml
python scripts/train_pretrain.py --config configs/pretrain_long_context_4k.yaml
python scripts/prepare_data.py --config configs/pretrain_long_context_16k.yaml
python scripts/train_pretrain.py --config configs/pretrain_long_context_16k.yaml
```

The warm-start validator permits only a larger RoPE window plus compatible
SDPA/checkpointing execution switches. Expert count, top-k, GQA shape, width,
depth, tokenizer hash, and all learned parameter shapes must match.

## 6. Supervised fine-tuning at 16K

```powershell
python scripts/train_sft.py --config configs/sft_long_context_16k.yaml
```

Resume:

```powershell
python scripts/train_sft.py --config configs/sft_long_context_16k.yaml --resume checkpoints/sft_moe_16k/latest.safetensors
```

Dynamic batch padding means short SFT examples do not pay the full 16K
padding cost.

## 7. Evaluate

```powershell
python scripts/eval_checkpoint.py --chat-config configs/chat_long_horizon.yaml --sft-config configs/sft_long_context_16k.yaml --pretrain-config configs/pretrain_long_context_16k.yaml --prompts configs/eval_prompts.jsonl
```

## 8. Chat with thinking and tool use

Terminal:

```powershell
python scripts/chat.py --config configs/chat_long_horizon.yaml
```

Web:

```powershell
python scripts/web_chat.py --config configs/chat_long_horizon.yaml
```

The web UI defaults to `http://127.0.0.1:8000`. Tool mode provides a
restricted calculator and current-time lookup. The long-horizon chat config
allows up to 32 in-turn tool rounds.

For task-level execution that can pause and resume across processes:

```powershell
python scripts/run_agent_task.py --state logs/task.json --objective "Complete this multi-step task." --steps-per-run 8
python scripts/run_agent_task.py --state logs/task.json --steps-per-run 8
```

Old dense `.pt` checkpoints are architecture-incompatible with the MoE model
and updated tokenizer. Complete pretraining and SFT before launching chat.
