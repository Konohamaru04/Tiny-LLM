# Modern model and training stack

The default configuration is now a compact 2026-style decoder-only
Transformer. The implementation remains pure PyTorch and intentionally small
enough to study.

## Architecture

- Pre-normalized RMSNorm with `norm_eps: 1e-6`
- Rotary position embeddings (RoPE) with `rope_theta: 1_000_000`
- Grouped-query attention (GQA): 6 query heads and 2 key/value heads
- Per-head QK normalization
- PyTorch scaled dot-product attention, including native GQA when available
- SwiGLU feed-forward layers with hardware-friendly hidden-size rounding
- Bias-free linear projections
- Tied input/output embeddings
- Depth-scaled residual projection initialization
- 2,048-token context window

The compatibility switches in `ModelConfig` still support learned positions,
LayerNorm, GELU MLPs, equal query/KV head counts, and manual attention for
experiments.

## Training optimizations

- Mixed precision where supported
- Gradient accumulation and clipping
- Activation checkpointing
- Fused AdamW on supported CUDA runtimes
- Cosine learning-rate decay with warmup
- Auxiliary logit z-loss for stability
- Stable model/tokenizer fingerprints in checkpoints
- Optional `torch.compile`

`attention_impl: auto` selects scaled dot-product attention when the installed
PyTorch runtime supports it and otherwise uses the manual implementation.

## Reasoning and tool use

The tokenizer and SFT pipeline use one canonical multi-turn format:

```text
<|system|> ...
<|tools|> [...] </|tools|>
<|user|> ...
<|assistant|>
<|think|> ... </|think|>
<|tool_call|> {...} </|tool_call|>
<|tool_response|> {...} </|tool_response|>
```

SFT loss is applied only to assistant reasoning, tool calls, and answers. Tool
responses remain context, not targets. The runtime can execute multiple tool
rounds and feed results back into generation. Its built-in calculator uses a
restricted expression AST; it never evaluates Python code. The other built-in
tool returns the current date/time for an IANA timezone.

These are trainable capabilities, not a claim that untrained weights already
reason reliably. Retrain pretraining and SFT checkpoints after this upgrade.

## Checkpoint compatibility

The new tokenizer vocabulary, GQA projection shapes, and model defaults are
not compatible with old `pretrain_tiny` or `sft_tiny` weights. New artifacts
are written to:

- `checkpoints/pretrain_modern/`
- `checkpoints/sft_modern/`

Configuration and tokenizer fingerprints are checked during resume and chat
loading to make mismatches fail early.

## Technical references

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
- [OLMo 3 Technical Report](https://arxiv.org/abs/2512.13961)
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
