# Tiny-LLM

A small, local GPT-style language model project built in pure PyTorch as part of my learning journey into language models, tokenization, training pipelines, and practical LLM engineering.

This repository is focused on clarity over complexity. The goal was not to build the biggest model possible, but to understand the full workflow end to end:

* preparing raw Markdown data
* training a SentencePiece tokenizer
* creating packed token datasets
* training a decoder-only Transformer
* running supervised fine-tuning for chat-style responses
* chatting with the model locally on consumer hardware

This project helped me understand how modern language models work under the hood without depending on high-level training frameworks.

## Why I built this

This project is part of my personal learning journey in machine learning and LLM engineering.

I wanted to move beyond using prebuilt APIs and actually understand the mechanics of training a small GPT-style model locally. Building this repo gave me hands-on experience with:

* tokenizer training
* sequence packing for causal language modeling
* Transformer architecture basics
* training loop design in raw PyTorch
* mixed precision and local GPU execution
* checkpointing and resume logic
* supervised fine-tuning for instruction/chat behavior

For students and beginners, I think this kind of project is especially helpful because it makes each stage visible instead of hiding everything behind large frameworks.

## Project overview

This repository implements a tiny decoder-only language model that can be trained locally on a single GPU.

### Main features

* Pure PyTorch implementation
* SentencePiece tokenizer training
* Markdown-based training corpus
* Deterministic train/validation split
* Packed next-token training data
* Decoder-only causal Transformer
* Supervised fine-tuning for chat formatting
* Optional JSON-style response patterning
* Minimal terminal chat interface
* CPU fallback support
* Windows and Linux friendly paths and scripts

## Hardware, software, and tools used

### Hardware

Recommended:

* NVIDIA RTX 4060 Ti 16GB
* At least 16GB system RAM
* SSD storage recommended

Minimum usable setup:

* CPU-only system will run, but training will be very slow
* Smaller GPUs may require reducing batch size, block size, or model size

### Software

* Python 3.10+
* pip
* Git
* NVIDIA GPU driver (if using CUDA)
* CUDA-enabled PyTorch build for GPU training

### Python dependencies

* torch
* sentencepiece
* numpy
* PyYAML
* tqdm

## Repository structure

```text
Tiny-LLM/
├── README.md
├── requirements.txt
├── configs/
│   ├── tokenizer.yaml
│   ├── pretrain_tiny.yaml
│   ├── sft_tiny.yaml
│   └── chat.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── sft/
├── checkpoints/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── tokenizer_utils.py
│   ├── data_utils.py
│   ├── datasets.py
│   ├── model.py
│   ├── trainer.py
│   └── generation.py
└── scripts/
    ├── train_tokenizer.py
    ├── prepare_data.py
    ├── train_pretrain.py
    ├── train_sft.py
    └── chat.py
```

## Environment setup guide

### 1. Clone the repository

```bash
git clone https://github.com/Konohamaru04/Tiny-LLM.git
cd Tiny-LLM
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If you want GPU training on NVIDIA hardware, install a CUDA-enabled PyTorch build first.

Example for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt --no-deps
```

If you are using CPU only, then a standard install may be enough:

```bash
pip install -r requirements.txt
```

### 4. Verify PyTorch sees the GPU

```bash
python -c "import torch; print(torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda); print('device_count=', torch.cuda.device_count())"
```

If `cuda_available=True`, training should use your GPU.

## Step-by-step implementation and execution

This project follows a clear pipeline. Each stage has a practical role.

---

### Step 1: Add raw Markdown training data

You can either manually place Markdown files inside `data/raw/` **or generate a starter dataset automatically** using the helper script:

```bash
python generate_natural_dataset.py
```

This script generates natural-language Markdown documents designed for educational training runs and testing the full pipeline. It produces readable explanations, small tutorials, and conceptual write‑ups that are useful for tiny‑model pretraining.

Typical output:

```
data/raw/generated_doc_001.md
...
data/raw/generated_doc_1000.md
```

This makes it easy for students to bootstrap a dataset without collecting large external corpora.

---

### Step 1: Add raw Markdown training data (manual method)

Put your `.md` files inside:

```text
data/raw/
```

You can use:

* personal notes
* tutorials
* documentation
* writeups
* README files
* technical explainers

#### What is happening here?

The model needs text to learn from. In this project, the raw corpus is plain Markdown because it is easy to collect, readable, and naturally contains useful language patterns such as headings, lists, paragraphs, and code blocks.

#### Common question

**Why Markdown?**

Markdown is simple, widely available, and a nice middle ground between plain text and structured documentation.

---

### Step 2: Train the tokenizer

Run:

```bash
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
```

#### What is happening here?

The tokenizer converts text into token IDs that the model can understand.

This script:

* scans all `.md` files under `data/raw/`
* creates a deterministic train/validation document split
* trains SentencePiece on the training split only
* saves tokenizer artifacts in `data/processed/`

#### Why train the tokenizer only on the training split?

This avoids validation leakage. The validation set should remain unseen during training-related steps as much as practical.

#### Output files

Typical outputs include:

* `data/processed/tokenizer.model`
* `data/processed/tokenizer.vocab`
* `data/processed/tokenizer_meta.json`
* `data/processed/train_manifest.json`
* `data/processed/val_manifest.json`

#### Common question

**Why does the actual vocab size sometimes differ from the requested vocab size?**

If the dataset is too small, SentencePiece may not be able to create the full requested vocabulary. In that case, update the model config to match the actual vocab size recorded in `tokenizer_meta.json`.

---

### Step 3: Prepare tokenized training data

Run:

```bash
python scripts/prepare_data.py --config configs/pretrain_tiny.yaml
```

#### What is happening here?

Now that the tokenizer exists, the raw text is encoded into token IDs.

This script:

* loads the train and validation manifests
* reads the corresponding Markdown files
* tokenizes each document
* concatenates token streams
* packs them into fixed-length blocks for causal language modeling
* saves packed arrays for training

#### Why pack fixed-length blocks?

Neural network training is more efficient when sequences are grouped into fixed shapes. Each block becomes one training sample.

#### Output files

Typical outputs include:

* `data/processed/lm_train.npy`
* `data/processed/lm_val.npy`
* `data/processed/lm_meta.json`

#### Common question

**What does block size mean?**

`block_size` is the maximum context length the model sees during training. Larger values let the model look farther back, but they increase memory use.

---

### Step 4: Run pretraining

Run:

```bash
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml
```

#### What is happening here?

This is the language modeling stage.

The model is trained to predict the next token given previous tokens. This is the core learning task behind GPT-style models.

The training loop includes:

* causal language modeling loss
* AdamW optimizer
* learning rate warmup
* cosine decay
* gradient accumulation
* gradient clipping
* mixed precision when supported
* validation checks
* checkpoint saving
* early stopping based on validation behavior

#### Why pretrain before chat fine-tuning?

Pretraining teaches the model general language patterns. Fine-tuning afterward teaches the model how to answer in a more assistant-like format.

#### Output files

Typical outputs include:

* `checkpoints/pretrain_tiny/best.pt`
* `checkpoints/pretrain_tiny/latest.pt`
* periodic `step_*.pt` snapshots

#### Common question

**Why did training stop early?**

If validation loss stops improving for several evaluations, early stopping triggers. This is usually a good thing on small datasets because it reduces overfitting.

---

### Step 5: Run supervised fine-tuning (SFT)

Run:

```bash
python scripts/train_sft.py --config configs/sft_tiny.yaml
```

#### What is happening here?

In SFT, the model is trained on structured chat examples like:

```json
{"system": "...", "user": "...", "assistant": "..."}
```

These are converted into a chat-style token sequence using special tokens such as:

* `<|system|>`
* `<|user|>`
* `<|assistant|>`

The training code masks the loss so the model learns mainly from the assistant response rather than the prompt portion.

#### Why is this useful?

Pretraining teaches language. SFT teaches behavior.

Without SFT, the model may continue text but not necessarily respond like a helpful assistant.

#### Output files

Typical outputs include:

* `checkpoints/sft_tiny/best.pt`
* `checkpoints/sft_tiny/latest.pt`

#### Common question

**Why does the model still make mistakes after SFT?**

Because this is still a small model trained on a relatively small dataset. SFT improves formatting and response style, but it does not magically turn a tiny model into a frontier model.

---

### Step 6: Launch the local chat interface

Run:

```bash
python scripts/chat.py --config configs/chat.yaml
```

#### What is happening here?

The chat script:

* loads the tokenizer
* loads the trained checkpoint
* builds a prompt from system prompt, history, and latest user message
* generates new tokens autoregressively
* decodes them back into text

#### Features

* short conversation history
* optional JSON mode
* temperature sampling
* top-k sampling
* max token control

#### Common question

**Why are my answers weak or repetitive?**

This usually means one or more of the following:

* the training corpus is too small
* the SFT dataset is too repetitive
* the model is too small for the task
* training stopped before learning enough
* the tokenizer/model config mismatch caused problems earlier

## Exact execution flow

For a full end-to-end run:

```bash
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
python scripts/prepare_data.py --config configs/pretrain_tiny.yaml
python scripts/train_pretrain.py --config configs/pretrain_tiny.yaml
python scripts/train_sft.py --config configs/sft_tiny.yaml
python scripts/chat.py --config configs/chat.yaml
```

## Practical learning notes for students

If you are a student or beginner, here are a few useful lessons from this kind of project.

### 1. Small models are excellent for learning

You do not need billions of parameters to understand how LLM pipelines work. A tiny model is much easier to debug and reason about.

### 2. Data quality matters a lot

Even a clean model implementation will behave poorly if the dataset is too repetitive, too small, or too synthetic.

### 3. Tokenizers matter more than beginners expect

Poor tokenization choices can affect vocabulary coverage, context efficiency, and general performance.

### 4. Training loops are where engineering discipline shows up

Many learning resources stop at the model definition, but real projects need:

* validation
* checkpointing
* mixed precision
* resume logic
* gradient accumulation
* error handling

### 5. Evaluation matters more than optimism

Training loss going down is nice, but validation behavior is what tells you whether the model is actually learning something useful.

## Troubleshooting

### CUDA not found

Make sure you installed a CUDA-enabled PyTorch build, not a CPU-only one.

Check with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Tokenizer vocab mismatch

If the tokenizer produced a smaller actual vocabulary than requested, update `model.vocab_size` in both pretraining and SFT configs.

### Validation data too small

If packing fails, your validation split may not have enough tokens for the chosen block size. Lower the block size or increase the dataset size.

### Chat responses are repetitive

Improve the naturalness and diversity of your training corpus and SFT examples.

## Limitations

This project is educational and intentionally lightweight.

It does not include:

* distributed training
* large-scale datasets
* constrained decoding
* retrieval augmentation
* huge model sizes
* advanced serving infrastructure

That is by design. The point is to make the core ideas visible and understandable.

## Credits

This project was built as part of my hands-on learning journey in:

* PyTorch
* tokenization
* Transformer language models
* supervised fine-tuning
* local LLM experimentation

Special credit goes to the open-source ecosystem that makes projects like this possible, especially:

* PyTorch
* SentencePiece
* NumPy
* YAML tooling
* the broader ML education and open-source community

## Outro

If you are a student, beginner, or engineer exploring LLMs for the first time, I hope this repository helps make the process feel less mysterious.

This project is not about chasing hype. It is about understanding the pipeline, learning by building, and turning abstract ideas into something you can run, inspect, and improve locally.

If this repository helps you learn even one stage of the workflow more clearly, then it has already done its job.

If you found it useful, feel free to fork it, experiment with it, and adapt it to your own learning journey.
