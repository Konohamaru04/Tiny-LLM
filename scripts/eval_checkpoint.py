from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, load_chat_config, load_pretrain_config, load_sft_config
from src.datasets import PretrainNpyDataset, SFTJsonlDataset
from src.evaluation import evaluate_loss_on_loader, load_eval_prompts, run_generation_eval
from src.model import GPT
from src.tokenizer_utils import SentencePieceTokenizer
from src.utils import (
    ensure_parent_dir,
    get_amp_settings,
    get_device,
    load_torch_checkpoint,
    verify_checkpoint_fingerprints,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Tiny LLM checkpoint.")
    parser.add_argument("--chat-config", type=str, default="configs/chat.yaml", help="Path to chat YAML config.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path override.")
    parser.add_argument("--tokenizer", type=str, default="", help="Optional tokenizer path override.")
    parser.add_argument("--prompts", type=str, default="configs/eval_prompts.jsonl", help="Evaluation prompt JSONL.")
    parser.add_argument("--pretrain-config", type=str, default="", help="Optional pretraining config for LM val loss.")
    parser.add_argument("--sft-config", type=str, default="", help="Optional SFT config for validation loss.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for dataset-based evaluation.")
    parser.add_argument("--max-batches", type=int, default=20, help="Maximum batches to score from the validation loader.")
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature override.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling override.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Generation length override.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty override.")
    parser.add_argument("--device", type=str, default="auto", help="Device string, e.g. auto/cpu/cuda.")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chat_cfg = load_chat_config(args.chat_config)
    checkpoint_path = args.checkpoint or chat_cfg.checkpoint_path
    tokenizer_path = args.tokenizer or chat_cfg.tokenizer_model_path
    prompts_path = args.prompts
    temperature = chat_cfg.temperature if args.temperature is None else args.temperature
    top_k = chat_cfg.top_k if args.top_k is None else args.top_k
    max_new_tokens = chat_cfg.max_new_tokens if args.max_new_tokens is None else args.max_new_tokens
    repetition_penalty = chat_cfg.repetition_penalty if args.repetition_penalty is None else args.repetition_penalty

    device = get_device(args.device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    state = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    model_cfg = ModelConfig(**state["model_config"])
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match checkpoint vocab size "
            f"({model_cfg.vocab_size})."
        )
    verify_checkpoint_fingerprints(
        state,
        model_config=model_cfg,
        tokenizer_model_path=tokenizer_path,
        context="evaluation checkpoint",
    )
    model = GPT(model_cfg)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    prompts = load_eval_prompts(prompts_path)
    generation_summary, samples = run_generation_eval(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        default_system_prompt=chat_cfg.system_prompt,
        max_history_turns=chat_cfg.max_history_turns,
        default_json_mode=chat_cfg.json_mode,
        repetition_penalty=repetition_penalty,
    )

    amp = get_amp_settings(device, enabled=False)
    dataset_results: dict[str, dict[str, float | int]] = {}

    if args.pretrain_config:
        pretrain_cfg = load_pretrain_config(args.pretrain_config)
        dataset = PretrainNpyDataset(pretrain_cfg.data.val_array_path, block_size=pretrain_cfg.model.block_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
        dataset_results["pretrain_val"] = evaluate_loss_on_loader(
            model=model,
            loader=loader,
            device=device,
            max_batches=args.max_batches,
            amp_enabled=amp["enabled"],
            amp_dtype=amp["dtype"],
        )

    if args.sft_config:
        sft_cfg = load_sft_config(args.sft_config)
        dataset = SFTJsonlDataset(
            sft_cfg.data.val_jsonl_path,
            tokenizer=tokenizer,
            block_size=sft_cfg.model.block_size,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
        dataset_results["sft_val"] = evaluate_loss_on_loader(
            model=model,
            loader=loader,
            device=device,
            max_batches=args.max_batches,
            amp_enabled=amp["enabled"],
            amp_dtype=amp["dtype"],
        )

    report = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "tokenizer_path": str(Path(tokenizer_path).resolve()),
        "device": str(device),
        "generation": {
            "summary": generation_summary,
            "samples": samples,
        },
        "datasets": dataset_results,
    }

    output_path = Path(args.output) if args.output else Path("eval_reports") / f"{Path(checkpoint_path).stem}_eval.json"
    ensure_parent_dir(output_path)
    write_json(report, output_path)

    print(json.dumps({"generation_summary": generation_summary, "datasets": dataset_results}, indent=2, ensure_ascii=False))
    print(f"[info] report written to {Path(output_path).resolve()}")


if __name__ == "__main__":
    main()
