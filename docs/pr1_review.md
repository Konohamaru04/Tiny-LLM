# PR #1 architecture review

Reviewed source: [Konohamaru04/Tiny-LLM PR #1](https://github.com/Konohamaru04/Tiny-LLM/pull/1).

The PR had a useful direction and green CI, but it was not merged wholesale.
Its 45 commits also replaced newer GQA/QK-normalized attention with full
multi-head QKV projections and used a per-expert `where` scan. The branch
therefore adopts the ideas while reimplementing the hot paths.

## Adopted and strengthened

- Sparse four-expert, top-2 MoE in every transformer block
- Shared expert capacity and capacity-free routing
- Load-balancing and router z-loss telemetry
- Explicit thinking, tool-call, tool-response, and final-answer tokens
- Exact special-token identity validation
- Dynamic SFT batch padding
- Context-aware prompt compaction
- Safe staged RoPE continuation through 4K and 16K
- Resumable long-horizon task state
- CI coverage for current PyTorch and Python

## Deliberately not copied

- Full-size QKV attention, because the existing GQA path reduces KV memory
- Removal of QK normalization and residual-scaled initialization
- A full token scan with `torch.where` for every expert
- Direct 2K-to-16K training without an intermediate continuation stage
- Pickle `.pt` checkpoints

The resulting architecture follows the PR's MoE intent while preserving the
more efficient attention stack and using fully resumable SafeTensors state.
