# Evaluation Guide

Phase 3 adds two layers of observability:

- training-time metrics and sample generations written into each checkpoint directory
- a standalone checkpoint evaluation command for ad hoc scoring

## Training-time outputs

Both [pretrain_tiny.yaml](/E:/Tiny-LLM/configs/pretrain_tiny.yaml) and
[sft_tiny.yaml](/E:/Tiny-LLM/configs/sft_tiny.yaml) now point to
[eval_prompts.jsonl](/E:/Tiny-LLM/configs/eval_prompts.jsonl).

During training, the trainer writes:

- `metrics.jsonl`
- `metrics.csv`
- `sample_generations/step_XXXXXXX.json`

These files are created inside the run's checkpoint directory.

## Standalone checkpoint evaluation

Run:

```powershell
python scripts/eval_checkpoint.py --chat-config configs/chat.yaml --sft-config configs/sft_tiny.yaml --max-batches 20
```

Optional additions:

- `--pretrain-config configs/pretrain_tiny.yaml`
- `--checkpoint checkpoints/sft_tiny/best.pt`
- `--prompts configs/eval_prompts.jsonl`
- `--repetition-penalty 1.05`
- `--device cpu`

The command writes a JSON report under `eval_reports/` by default.

## Metrics currently reported

- validation loss
- perplexity
- average response length
- empty response rate
- 3-gram repetition score
- JSON validity rate on prompts that request JSON
