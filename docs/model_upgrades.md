# Modern model and training stack

The default configuration is now a compact 2026-style sparse
Mixture-of-Experts decoder. It remains pure PyTorch and small enough to study.

## Architecture

- Pre-normalized RMSNorm with `norm_eps: 1e-6`
- Rotary position embeddings (RoPE) with `rope_theta: 1_000_000`
- Grouped-query attention (GQA): 6 query heads and 2 key/value heads
- Per-head QK normalization
- PyTorch scaled dot-product attention, including native GQA when available
- Four routed SwiGLU experts per block, normalized top-2 token-choice routing
- One shared SwiGLU expert per block for common knowledge capacity
- Capacity-free routing: every selected assignment executes; no tokens drop
- Sort-once contiguous expert dispatch rather than a full scan per expert
- Switch-style load balancing plus router log-partition z-loss
- Bias-free linear projections
- Tied input/output embeddings
- Depth-scaled residual projection initialization
- Staged 2,048 -> 4,096 -> 16,384-token RoPE continuation

The shipped configs use MoE in every block. `moe_every_n_layers` remains an
experimental ablation switch.

## Training optimizations

- Mixed precision where supported
- Gradient accumulation and clipping
- Activation checkpointing
- Fused AdamW on supported CUDA runtimes
- Cosine learning-rate decay with warmup
- Auxiliary logit z-loss for stability
- Router jitter during pretraining, disabled during SFT
- Load-balancing and router z-loss regularization
- Stable model/tokenizer fingerprints in checkpoints
- Fully resumable SafeTensors state, including optimizer and scheduler tensors
- Optional `torch.compile`

The 16K configs require `attention_impl: sdpa`; this avoids allocating a dense
Python causal mask and lets current CUDA builds select memory-efficient kernels.

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
<|final|> ...
```

SFT batches are padded only to the longest sequence in the current batch. Loss
is applied only to assistant reasoning, tool calls, and final answers. The
runtime parses hidden reasoning separately, so it cannot leak through the
visible response stream.

Long-horizon tasks checkpoint atomically after each model step. A durable
progress ledger compacts old tool traces while always retaining system and
developer instructions, the objective, tool schemas, and the recent trace.
Runtime fingerprints prevent resuming state with a different checkpoint,
tokenizer, model config, or tool contract.

These are trainable capabilities, not a claim that untrained weights already
reason reliably. Retrain pretraining and SFT checkpoints after this upgrade.

## Checkpoint compatibility

The tokenizer vocabulary, MoE parameters, and GQA projection shapes are not
compatible with older dense weights. New artifacts are written to:

- `checkpoints/pretrain_moe/`
- `checkpoints/pretrain_moe_4k/`
- `checkpoints/pretrain_moe_16k/`
- `checkpoints/sft_moe_16k/`

Checkpoints use the `.safetensors` extension. The state serializer stores model,
optimizer, scheduler, scaler, and metadata without pickle. Configuration and
tokenizer fingerprints are checked during resume and chat loading.

## Technical references

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [OLMo 3 Technical Report](https://arxiv.org/abs/2512.13961)
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
- [SafeTensors documentation](https://huggingface.co/docs/safetensors/)
