# Unified MoE Checkpoint Architecture

This upgrade turns Tiny-LLM into a single-checkpoint model that can be prompted to operate in direct-answer, reasoning, automatic-routing, and tool-use modes.

## Design goals

- One tokenizer, model architecture, and final checkpoint for every capability.
- Sparse Mixture-of-Experts feed-forward layers so tokens can specialize without activating every parameter.
- Explicit mode control rather than separate chat and reasoning model files.
- Structured JSON tool calls with deterministic parsing and validation.
- RoPE and SDPA for longer contexts.
- A resumable agent loop for long-horizon tasks.
- Public, streamed datasets with bounded local sampling.

## Model architecture

The decoder remains a causal Transformer. Every Transformer block contains:

1. RMSNorm
2. causal self-attention using PyTorch SDPA
3. RMSNorm
4. sparse MoE feed-forward layer

The current local-friendly MoE defaults are:

- 4 experts per routed block
- top-2 experts selected per token
- SwiGLU expert MLPs
- a learned token router
- Switch-style auxiliary load-balancing loss
- shared attention and embeddings

Only two experts execute for each token. Total parameters increase, but active feed-forward compute is approximately two experts rather than all four.

The router is intentionally token-level. Capability tokens such as `<|think|>` and `<|tool_call|>` give it a learnable signal, but experts are not hard-coded to specific capabilities.

## Unified control protocol

The tokenizer includes these special tokens:

```text
<|think|> <|end_think|>
<|no_think|> <|auto_think|>
<|tools|> <|end_tools|>
<|tool_call|> <|end_tool_call|>
<|tool_result|> <|end_tool_result|>
<|final|>
```

### Direct mode

```text
<|assistant|>
<|no_think|>
<|final|>
A concise answer.
```

### Thinking mode

```text
<|assistant|>
<|think|>
Private reasoning trace used during capability training.
<|end_think|>
<|final|>
The user-facing answer.
```

### Automatic mode with tools

```text
<|assistant|>
<|auto_think|>
<|tools|>
[{"name":"calculator","description":"...","parameters":{...}}]
<|end_tools|>
<|tool_call|>
{"name":"calculator","arguments":{"expression":"2+2"},"id":"call-1"}
<|end_tool_call|>
```

The runtime validates tool names and requires `arguments` to be a JSON object. Tool results are returned in a matching structured envelope.

## Long-context strategy

The old learned 1K position embedding is replaced with RoPE. The base configuration trains at 4K tokens with a high RoPE base. A continuation configuration extends the same model weights to 16K tokens:

```bash
python scripts/prepare_data.py --config configs/pretrain_tiny.yaml
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml

python scripts/prepare_data.py --config configs/pretrain_long_context_16k.yaml
python scripts/train_pretrain.py --config configs/pretrain_long_context_16k.yaml
```

The 16K stage uses:

- RoPE with `rope_theta: 1000000`
- SDPA so supported CUDA builds can select memory-efficient/Flash kernels
- gradient checkpointing
- batch size 1
- larger gradient accumulation
- a lower continuation learning rate

A model cannot become reliable at long context merely by increasing `block_size`. The continuation corpus should contain sequences with dependencies spread across the full context window, retrieval tasks, multi-document synthesis, and long tool trajectories.

## Long-horizon runtime

`src/agent_runtime.py` implements iterative execution:

1. Build a compact task trace.
2. Ask the unified checkpoint to continue in auto-thinking mode.
3. Parse zero or more structured tool calls.
4. Execute registered handlers.
5. append observations.
6. Save state after every step.
7. Continue until `<|final|>` or `max_steps`.

The state file makes tasks resumable after interruption. Runtime orchestration does not by itself guarantee long-horizon competence; multi-step trajectories must be present during SFT and evaluation.

## Public dataset pipeline

Run:

```bash
python scripts/prepare_public_datasets.py \
  --fineweb-docs 20000 \
  --direct 20000 \
  --reasoning 20000 \
  --tools 10000
```

The script streams and caps each source:

- `HuggingFaceFW/fineweb-edu` for educational pretraining text (ODC-By 1.0; source documents retain their original rights and Common Crawl terms also apply).
- `HuggingFaceH4/ultrachat_200k` for direct assistant examples.
- `open-r1/OpenR1-Math-220k` for verified mathematical reasoning traces (Apache 2.0).
- `Johin/function-calling-dataset` for structured function calling (Apache 2.0).

Always review the current dataset cards, licenses, upstream terms, and generated-data provenance before distributing weights.

## Training order

1. Regenerate the tokenizer with the new control symbols.
2. Prepare the 4K pretraining arrays.
3. Train the sparse MoE base checkpoint.
4. Repack long documents at 16K and continue pretraining.
5. Prepare the unified SFT mixture.
6. Fine-tune the 16K checkpoint on direct, reasoning, tool, and multi-step examples.
7. Evaluate each mode independently and test cross-mode interference.

The final deployable artifact is the best checkpoint from `checkpoints/unified_moe_sft/`, not separate thinking and non-thinking checkpoints.

## Evaluation checklist

- direct-answer quality with `<|no_think|>`
- reasoning accuracy with `<|think|>`
- auto-mode routing accuracy
- valid JSON tool-call rate
- correct tool selection and argument accuracy
- no-tool decision accuracy
- recovery after tool errors
- completion rate across multi-step tasks
- 4K, 8K, and 16K retrieval accuracy
- expert utilization and router balance
- latency and VRAM per active token
