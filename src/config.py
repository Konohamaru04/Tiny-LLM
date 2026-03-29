from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.utils import resolve_path


@dataclass
class TokenizerConfig:
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    train_manifest_path: str = "data/processed/train_manifest.json"
    val_manifest_path: str = "data/processed/val_manifest.json"
    train_corpus_path: str = "data/processed/train_corpus.txt"
    val_corpus_path: str = "data/processed/val_corpus.txt"
    tokenizer_prefix: str = "data/processed/tokenizer"
    vocab_size: int = 16000
    model_type: str = "bpe"
    character_coverage: float = 1.0
    seed: int = 42
    val_fraction: float = 0.10
    normalization_rule_name: str = "identity"
    unk_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 3
    user_defined_symbols: List[str] = field(
        default_factory=lambda: [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|json|>",
            "</json>",
        ]
    )

    @property
    def tokenizer_model_path(self) -> str:
        return f"{self.tokenizer_prefix}.model"

    @property
    def tokenizer_vocab_path(self) -> str:
        return f"{self.tokenizer_prefix}.vocab"

    @property
    def tokenizer_meta_path(self) -> str:
        return str(Path(self.tokenizer_prefix).with_name(Path(self.tokenizer_prefix).name + "_meta.json"))


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 384
    mlp_ratio: int = 4
    dropout: float = 0.10
    tie_weights: bool = True
    norm_type: str = "layernorm"
    mlp_type: str = "gelu"
    positional_embedding: str = "learned"
    rope_theta: float = 10000.0
    attention_impl: str = "auto"
    gradient_checkpointing: bool = False


@dataclass
class PretrainDataConfig:
    tokenizer_model_path: str = "data/processed/tokenizer.model"
    train_manifest_path: str = "data/processed/train_manifest.json"
    val_manifest_path: str = "data/processed/val_manifest.json"
    train_array_path: str = "data/processed/lm_train.npy"
    val_array_path: str = "data/processed/lm_val.npy"
    meta_path: str = "data/processed/lm_meta.json"


@dataclass
class TrainConfig:
    output_dir: str = "checkpoints/pretrain_tiny"
    seed: int = 42
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 2000
    num_workers: int = 0
    weight_decay: float = 0.10
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_steps: int = 50
    save_interval: int = 200
    log_interval: int = 10
    patience: int = 5
    use_amp: bool = True
    resume_from: str = ""
    metrics_jsonl_path: str = ""
    metrics_csv_path: str = ""
    sample_prompts_path: str = ""
    sample_output_dir: str = ""
    sample_system_prompt: str = "You are a concise and helpful assistant."
    sample_max_history_turns: int = 0
    sample_temperature: float = 0.4
    sample_top_k: int = 50
    sample_max_new_tokens: int = 128
    sample_repetition_penalty: float = 1.0
    sample_json_mode: bool = False
    compile_model: bool = False
    compile_backend: str = ""
    compile_mode: str = ""


@dataclass
class PretrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: PretrainDataConfig = field(default_factory=PretrainDataConfig)
    training: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class SFTDataConfig:
    tokenizer_model_path: str = "data/processed/tokenizer.model"
    train_jsonl_path: str = "data/sft/sample_train.jsonl"
    val_jsonl_path: str = "data/sft/sample_val.jsonl"


@dataclass
class SFTTrainConfig:
    output_dir: str = "checkpoints/sft_tiny"
    pretrained_checkpoint: str = "checkpoints/pretrain_tiny/best.pt"
    seed: int = 42
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 20
    max_steps: int = 300
    num_workers: int = 0
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_steps: int = 20
    save_interval: int = 50
    log_interval: int = 10
    patience: int = 5
    use_amp: bool = True
    resume_from: str = ""
    metrics_jsonl_path: str = ""
    metrics_csv_path: str = ""
    sample_prompts_path: str = ""
    sample_output_dir: str = ""
    sample_system_prompt: str = "You are a concise and helpful assistant."
    sample_max_history_turns: int = 0
    sample_temperature: float = 0.4
    sample_top_k: int = 50
    sample_max_new_tokens: int = 128
    sample_repetition_penalty: float = 1.0
    sample_json_mode: bool = False
    compile_model: bool = False
    compile_backend: str = ""
    compile_mode: str = ""


@dataclass
class SFTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: SFTDataConfig = field(default_factory=SFTDataConfig)
    training: SFTTrainConfig = field(default_factory=SFTTrainConfig)


@dataclass
class ChatConfig:
    checkpoint_path: str = "checkpoints/sft_tiny/best.pt"
    tokenizer_model_path: str = "data/processed/tokenizer.model"
    system_prompt: str = "You are a concise and helpful assistant."
    personas_path: str = "configs/personas.json"
    default_persona: str = "technical"
    session_file: str = ""
    max_history_turns: int = 4
    temperature: float = 0.8
    top_k: int = 50
    max_new_tokens: int = 128
    repetition_penalty: float = 1.0
    stream: bool = True
    json_mode: bool = False
    device: str = "auto"


def _load_yaml_dict(path: str | Path) -> Dict[str, Any]:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping at the top level: {path}")
    return data


def _merge_dataclass(instance: Any, updates: Dict[str, Any] | None) -> Any:
    base = asdict(instance)
    if updates:
        base.update(updates)
    return type(instance)(**base)


def _validate_model_config(cfg: ModelConfig) -> None:
    if cfg.vocab_size <= 0:
        raise ValueError("model.vocab_size must be > 0")
    if cfg.block_size < 8:
        raise ValueError("model.block_size must be >= 8")
    if cfg.n_layer <= 0:
        raise ValueError("model.n_layer must be > 0")
    if cfg.n_head <= 0:
        raise ValueError("model.n_head must be > 0")
    if cfg.n_embd <= 0:
        raise ValueError("model.n_embd must be > 0")
    if cfg.n_embd % cfg.n_head != 0:
        raise ValueError(
            f"model.n_embd ({cfg.n_embd}) must be divisible by model.n_head ({cfg.n_head})"
        )
    if cfg.mlp_ratio < 1:
        raise ValueError("model.mlp_ratio must be >= 1")
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError("model.dropout must be in [0.0, 1.0)")
    if cfg.norm_type not in {"layernorm", "rmsnorm"}:
        raise ValueError("model.norm_type must be 'layernorm' or 'rmsnorm'")
    if cfg.mlp_type not in {"gelu", "swiglu"}:
        raise ValueError("model.mlp_type must be 'gelu' or 'swiglu'")
    if cfg.positional_embedding not in {"learned", "rope"}:
        raise ValueError("model.positional_embedding must be 'learned' or 'rope'")
    if cfg.rope_theta <= 0.0:
        raise ValueError("model.rope_theta must be > 0")
    if cfg.attention_impl not in {"auto", "manual", "sdpa"}:
        raise ValueError("model.attention_impl must be 'auto', 'manual', or 'sdpa'")
    if cfg.positional_embedding == "rope" and (cfg.n_embd // cfg.n_head) % 2 != 0:
        raise ValueError("RoPE requires head_dim to be even.")


def _validate_train_config(cfg: TrainConfig | SFTTrainConfig) -> None:
    if cfg.batch_size <= 0:
        raise ValueError("training.batch_size must be > 0")
    if cfg.gradient_accumulation_steps <= 0:
        raise ValueError("training.gradient_accumulation_steps must be > 0")
    if cfg.learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be > 0")
    if cfg.min_lr <= 0.0:
        raise ValueError("training.min_lr must be > 0")
    if cfg.min_lr > cfg.learning_rate:
        raise ValueError("training.min_lr must be <= training.learning_rate")
    if cfg.max_steps <= 0:
        raise ValueError("training.max_steps must be > 0")
    if cfg.warmup_steps < 0:
        raise ValueError("training.warmup_steps must be >= 0")
    if cfg.eval_interval <= 0:
        raise ValueError("training.eval_interval must be > 0")
    if cfg.save_interval <= 0:
        raise ValueError("training.save_interval must be > 0")
    if cfg.log_interval <= 0:
        raise ValueError("training.log_interval must be > 0")
    if cfg.eval_steps <= 0:
        raise ValueError("training.eval_steps must be > 0")
    if cfg.grad_clip < 0.0:
        raise ValueError("training.grad_clip must be >= 0")
    if cfg.num_workers < 0:
        raise ValueError("training.num_workers must be >= 0")
    if cfg.patience < 0:
        raise ValueError("training.patience must be >= 0")
    if cfg.sample_max_history_turns < 0:
        raise ValueError("training.sample_max_history_turns must be >= 0")
    if cfg.sample_temperature < 0.0:
        raise ValueError("training.sample_temperature must be >= 0.0")
    if cfg.sample_top_k < 0:
        raise ValueError("training.sample_top_k must be >= 0")
    if cfg.sample_max_new_tokens <= 0:
        raise ValueError("training.sample_max_new_tokens must be > 0")
    if cfg.sample_repetition_penalty < 1.0:
        raise ValueError("training.sample_repetition_penalty must be >= 1.0")


def load_tokenizer_config(path: str | Path) -> TokenizerConfig:
    raw = _load_yaml_dict(path)
    cfg = _merge_dataclass(TokenizerConfig(), raw)
    if cfg.vocab_size <= 32:
        raise ValueError("vocab_size must be greater than 32")
    if cfg.model_type not in {"bpe", "unigram"}:
        raise ValueError("model_type must be either 'bpe' or 'unigram'")
    if not (0.0 < cfg.val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")
    if not cfg.user_defined_symbols:
        raise ValueError("user_defined_symbols must not be empty")
    return cfg


def load_pretrain_config(path: str | Path) -> PretrainConfig:
    raw = _load_yaml_dict(path)
    cfg = PretrainConfig(
        model=_merge_dataclass(ModelConfig(), raw.get("model")),
        data=_merge_dataclass(PretrainDataConfig(), raw.get("data")),
        training=_merge_dataclass(TrainConfig(), raw.get("training")),
    )
    _validate_model_config(cfg.model)
    _validate_train_config(cfg.training)
    return cfg


def load_sft_config(path: str | Path) -> SFTConfig:
    raw = _load_yaml_dict(path)
    cfg = SFTConfig(
        model=_merge_dataclass(ModelConfig(), raw.get("model")),
        data=_merge_dataclass(SFTDataConfig(), raw.get("data")),
        training=_merge_dataclass(SFTTrainConfig(), raw.get("training")),
    )
    _validate_model_config(cfg.model)
    _validate_train_config(cfg.training)
    return cfg


def load_chat_config(path: str | Path) -> ChatConfig:
    raw = _load_yaml_dict(path)
    cfg = _merge_dataclass(ChatConfig(), raw)
    if cfg.max_history_turns < 0:
        raise ValueError("max_history_turns must be >= 0")
    if cfg.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if cfg.top_k < 0:
        raise ValueError("top_k must be >= 0")
    if cfg.temperature < 0.0:
        raise ValueError("temperature must be >= 0.0")
    if cfg.repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be >= 1.0")
    return cfg
