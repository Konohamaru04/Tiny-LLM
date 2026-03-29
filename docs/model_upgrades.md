# Model Upgrades

Phase 4 adds backward-compatible model and training upgrades behind config
flags.

## Model options

These live under the `model:` section in
[pretrain_tiny.yaml](/E:/Tiny-LLM/configs/pretrain_tiny.yaml) and
[sft_tiny.yaml](/E:/Tiny-LLM/configs/sft_tiny.yaml).

- `norm_type`: `layernorm` or `rmsnorm`
- `mlp_type`: `gelu` or `swiglu`
- `positional_embedding`: `learned` or `rope`
- `rope_theta`: RoPE base frequency
- `attention_impl`: `auto`, `manual`, or `sdpa`
- `gradient_checkpointing`: `true` or `false`

Defaults preserve the original architecture, but `attention_impl: auto` will
use PyTorch scaled dot-product attention when available.

## Training options

These live under the `training:` section.

- `compile_model`
- `compile_backend`
- `compile_mode`

`compile_model` is off by default because compile behavior can vary by platform.

## Checkpoint safety

New checkpoints now store:

- a stable hash of `model_config`
- a SHA-256 hash of the tokenizer model file

On resume and other load paths, those fingerprints are verified when present.
That helps catch silent mismatches between checkpoints, configs, and tokenizer
artifacts.

## Sampling

[chat.yaml](/E:/Tiny-LLM/configs/chat.yaml) now supports `repetition_penalty`
for a small amount of repetition control during generation and evaluation.
