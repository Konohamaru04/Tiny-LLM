# Tiny-LLM

<p align="center">
  <img src="Assets/logo.jpg" width="400"/>
</p>

A compact, modern decoder-only language model built directly in PyTorch. The
project exposes the complete local pipeline: public-data preparation,
tokenizer training, pretraining, reasoning/tool supervised fine-tuning,
evaluation, and terminal/web chat.

## What is included

- Sparse top-2/4 MoE with a shared SwiGLU expert and capacity-free routing
- RMSNorm, RoPE, QK normalization, and grouped-query attention
- PyTorch scaled dot-product attention with native GQA where available
- Staged 2K -> 4K -> 16K context training and an 8,192-token vocabulary
- Bias-free projections and depth-scaled residual initialization
- Mixed precision, activation checkpointing, fused AdamW, gradient clipping,
  warmup/cosine decay, logit z-loss, and router regularization
- Single-file, fully resumable `.safetensors` training checkpoints
- Canonical multi-turn messages with system, developer, user, assistant, and
  tool roles
- Assistant-only loss masking for reasoning traces, tool calls, and answers
- Multi-round tool execution with a safe calculator and timezone-aware clock
- Resumable long-horizon tasks with atomic step checkpoints and bounded memory
- Streaming terminal chat, persistent sessions, personas, and local web UI
- Exact dataset revisions, resumable downloads, rate limiting, normalization,
  content deduplication, checksums, and deterministic splits

The implementation is intended for learning and local experimentation. A tiny
model needs actual pretraining and SFT before it can use these capabilities
reliably.

## Data snapshot

The pinned starter snapshot includes:

- 5,000 FineWeb-Edu pretraining documents
- 297 additional unique local Markdown documents
- 1,000 OpenR1 Math reasoning examples
- 2,000 Dolci tool-use examples
- 300 Tülu MAX thinking/tool examples

The 297 Markdown documents exist in two mirrored locations but are included
once by normalized content hash. Public SFT is split into 3,127 training and
173 validation records.

See [public_datasets.md](docs/public_datasets.md) for revisions and terms, and
[dataset_requirements.md](docs/dataset_requirements.md) for the accepted data
contract.

## Requirements

- Python 3.10+
- PyTorch 2.6+ and SafeTensors 0.5+
- SentencePiece 0.2+
- NumPy, PyYAML, tqdm, and aiofiles

CPU execution is supported but training is slow. A CUDA GPU with at least 16 GB
VRAM is recommended for the 2K stage. The 16K stage requires SDPA, activation
checkpointing, batch size 1, and may require more VRAM depending on the CUDA
kernel selected. Install the current PyTorch build appropriate for your OS and CUDA runtime using the
[official selector](https://pytorch.org/get-started/locally/), then install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify the runtime:

```powershell
python -c "import torch; print(torch.__version__); print('cuda=', torch.cuda.is_available()); print('cuda_runtime=', torch.version.cuda)"
```

## End-to-end run

```powershell
python scripts/download_public_datasets.py --config configs/public_datasets.yaml --stage all
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
python scripts/prepare_data.py --config configs/pretrain_tiny.yaml
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml
python scripts/prepare_data.py --config configs/pretrain_long_context_4k.yaml
python scripts/train_pretrain.py --config configs/pretrain_long_context_4k.yaml
python scripts/prepare_data.py --config configs/pretrain_long_context_16k.yaml
python scripts/train_pretrain.py --config configs/pretrain_long_context_16k.yaml
python scripts/train_sft.py --config configs/sft_long_context_16k.yaml
python scripts/eval_checkpoint.py --chat-config configs/chat_long_horizon.yaml --sft-config configs/sft_long_context_16k.yaml --pretrain-config configs/pretrain_long_context_16k.yaml
python scripts/chat.py --config configs/chat_long_horizon.yaml
```

For resume commands and the web UI, see [ExecSeq.md](ExecSeq.md).

## Architecture defaults

| Setting | Value |
|---|---:|
| Layers | 8 |
| Model width | 384 |
| Query heads | 6 |
| Key/value heads | 2 |
| Routed experts | 4, top-2 |
| Shared experts | 1 per block |
| Parameters | 59.4M total |
| Context | 16,384 after staged continuation |
| Vocabulary | 8,192 |
| Expert MLP | SwiGLU, 2.667x rounded to 128 |
| Positions | RoPE, theta 1,000,000 |
| Normalization | RMSNorm + per-head QK norm |

The architecture and optimization rationale is documented in
[model_upgrades.md](docs/model_upgrades.md). The selective integration decision
for upstream PR #1 is recorded in [pr1_review.md](docs/pr1_review.md).

## Thinking and tool protocol

The tokenizer reserves explicit control tokens:

```text
<|tools|> ... </|tools|>
<|think|> ... </|think|>
<|tool_call|> ... </|tool_call|>
<|tool_response|> ... </|tool_response|>
<|final|> ...
```

Tool definitions use JSON Schema. During inference, the assistant can emit one
or more calls, the local runtime executes registered tools, appends their
results as tool messages, and asks the model to continue. The built-in
calculator parses a restricted arithmetic AST and cannot execute arbitrary
Python or shell code.

## Repository map

```text
configs/                    model, training, chat, and dataset pins
data/raw/                   local Markdown corpus
data/raw/public_fineweb/    downloaded FineWeb-Edu subset
data/public_sft/            normalized reasoning/tool SFT split and manifest
docs/                       architecture and dataset documentation
scripts/                    download, prepare, train, evaluate, and chat CLIs
src/chat_format.py          canonical multi-turn serialization and loss masks
src/tools.py                tool schemas, parsing, and safe built-ins
src/moe.py                  sparse routing, experts, and router statistics
src/model.py                MoE decoder with GQA, QK norm, RoPE, and SDPA
src/long_horizon.py          resumable task state and bounded progress ledger
src/checkpoint_compat.py     safe staged RoPE context warm starts
src/trainer.py              pretraining/SFT optimization loop
tests/                      architecture, data, masking, and runtime tests
```

Large downloaded datasets and generated tokenizer/array artifacts are ignored
by Git. The small public-data manifest remains tracked for reproducibility.

## Checkpoint migration

Old dense `.pt` checkpoints are not architecture-compatible with the MoE
weights or updated tokenizer. New training state is safely serialized into:

- `checkpoints/pretrain_moe*/**.safetensors`
- `checkpoints/sft_moe*/**.safetensors`

Checkpoint loading verifies both model-configuration and tokenizer
fingerprints so stale combinations fail early.

## Validation

```powershell
python -m unittest discover -s tests
python -m compileall -q src scripts tests
git diff --check
```

## Scope

Tiny-LLM does not yet provide distributed expert parallelism, retrieval augmentation,
constrained JSON decoding, production serving, or frontier-scale evaluation.
Reasoning traces improve the training signal but are not guaranteed to be
correct or faithful. Review all dataset licenses and upstream terms before
redistributing data, weights, or outputs.
