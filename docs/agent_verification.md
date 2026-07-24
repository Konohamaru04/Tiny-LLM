# Agent-mode verification

Verified branch: `agent/unified-checkpoint-thinking-tools`

The branch is continuously checked by two GitHub Actions workflows:

- `.github/workflows/ci.yml` runs source compilation and the complete pytest suite.
- `.github/workflows/agent-verify.yml` additionally validates production configs, public-dataset CLI imports, isolated subsystem tests, and a sparse-MoE forward/backward pass with router gradients.

## Verified behavior

- production tokenizer config exposes distinct IDs for all chat, thinking, tool, result, and final-answer tokens
- 4K base, 16K continuation, and 16K SFT configs retain 4-expert top-2 MoE routing
- exact checkpoints restore full optimizer/scheduler state
- safe RoPE context extension performs a model-only warm start
- unrelated architecture changes are rejected during warm start
- direct, thinking, auto-thinking, tool-call, tool-result, and final-answer parsing
- hidden reasoning is not returned as the visible answer
- resumable long-horizon tool loops persist task state and feed protocol-correct observations back to the model
- long-context prompt compaction preserves system instructions, role markers, mode control, and tool schemas
- SFT uses dynamic per-batch padding instead of padding every short example to 16K
- MoE routing participates in backpropagation and router parameters receive gradients

## Issues found during verification

Agent-mode verification found and fixed several integration issues:

1. MoE YAML fields were absent from `ModelConfig`.
2. Training configs did not explicitly enable MoE, allowing a silent dense fallback.
3. Capability tokens were not required by the tokenizer runtime and could collapse to `<unk>`.
4. Hidden reasoning could leak when a response omitted `<|final|>`.
5. Tool results were not fed back using the trained tool-result protocol.
6. The 4K-to-16K stage attempted an incompatible full resume instead of a guarded model-only warm start.
7. Final SFT still targeted the 4K checkpoint and padded every sample to the entire context window.
8. Prompt truncation could discard system instructions or the task objective.
9. Legacy tests and CI used context/vocabulary fixtures that predated the capability protocol.

## Verification boundary

These checks validate architecture wiring, configuration compatibility, data formatting, inference protocol behavior, checkpoint loading rules, agent orchestration, and CPU-level forward/backward correctness.

They do not prove trained-model quality. A complete verification of reasoning quality, expert specialization, tool accuracy, long-horizon completion, 16K retrieval, VRAM use, throughput, and convergence requires running the documented tokenizer, pretraining, context-extension, and SFT curriculum on real data and GPU hardware.
