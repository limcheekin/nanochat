#!/usr/bin/env python3
"""
Optimized Base Model Pre-training with Unsloth
This script trains an architecturally-identical model to karpathy-nanochat,
with improvements for flexibility and reusability.
"""

import os
import torch
import argparse
import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class TrainingConfig:
    """Configuration for NanoChat pre-training. Can be overridden by command-line arguments."""
    depth: int = 20
    max_seq_len: int = 2048
    device_batch_size: int = 32
    total_batch_size: int = 524288
    base_lr: float = 3e-4
    embedding_lr_scale: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_ratio: float = 0.02
    target_param_data_ratio: int = 20
    save_every: int = 1000
    use_bf16: bool = True
    output_dir_template: str = "./nanochat_unsloth_d{depth}"
    compile_model: bool = False
    report_to: str = "none"
    wandb_project: str = "nanochat_unsloth"
    wandb_run_name_template: str = "d{depth}_{timestamp}"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_split: str = "train"


def get_args_parser():
    parser = argparse.ArgumentParser(description="Optimized NanoChat Pre-training with Unsloth")
    
    config_defaults = TrainingConfig()
    parser.add_argument("--depth", type=int, default=config_defaults.depth, help="Number of layers in the model.")
    parser.add_argument("--max-seq-len", type=int, default=config_defaults.max_seq_len, help="Maximum sequence length.")
    parser.add_argument("--device-batch-size", type=int, default=config_defaults.device_batch_size, help="Batch size per GPU.")
    parser.add_argument("--total-batch-size", type=int, default=config_defaults.total_batch_size, help="Target total batch size across all GPUs.")
    parser.add_argument("--base-lr", type=float, default=config_defaults.base_lr, help="Base learning rate.")
    parser.add_argument("--embedding-lr-scale", type=float, default=config_defaults.embedding_lr_scale, help="Scaling factor for embedding learning rate.")
    parser.add_argument("--weight-decay", type=float, default=config_defaults.weight_decay, help="Weight decay for the optimizer.")
    parser.add_argument("--grad-clip", type=float, default=config_defaults.grad_clip, help="Gradient clipping value.")
    parser.add_argument("--warmup-ratio", type=float, default=config_defaults.warmup_ratio, help="Warmup ratio for the learning rate scheduler.")
    parser.add_argument("--target-param-data-ratio", type=int, default=config_defaults.target_param_data_ratio, help="Target ratio of tokens to parameters for training duration.")
    parser.add_argument("--save-every", type=int, default=config_defaults.save_every, help="Save a checkpoint every N steps.")
    parser.add_argument("--output-dir-template", type=str, default=config_defaults.output_dir_template, help="Template for the output directory.")
    parser.add_argument("--wandb-project", type=str, default=config_defaults.wandb_project, help="WandB project name.")
    parser.add_argument("--wandb-run-name-template", type=str, default=config_defaults.wandb_run_name_template, help="Template for the WandB run name.")
    parser.add_argument("--report-to", type=str, default=config_defaults.report_to, help="Reporting destination (e.g., 'wandb', 'none').")
    parser.add_argument("--dataset-name", type=str, default=config_defaults.dataset_name, help="Hugging Face dataset name.")
    parser.add_argument("--dataset-config", type=str, default=config_defaults.dataset_config, help="Hugging Face dataset config name.")
    parser.add_argument("--dataset-split", type=str, default=config_defaults.dataset_split, help="Hugging Face dataset split.")

    parser.add_argument('--use-bf16', action='store_true', default=True, help="Enable bfloat16 training.")
    parser.add_argument('--no-bf16', action='store_false', dest='use_bf16', help="Disable bfloat16 training.")
    parser.add_argument('--compile-model', action='store_true', default=False, help="Enable torch.compile for the model.")

    return parser


def create_model(config: TrainingConfig, vocab_size: int, is_main_process: bool):
    if is_main_process:
        logging.info(f"Instantiating model with depth={config.depth}")
    
    model_config = GPTConfig(
        sequence_len=config.max_seq_len,
        vocab_size=vocab_size,
        n_layer=config.depth,
        n_embd=config.depth * 64,
        n_head=max(1, ((config.depth * 64) + 127) // 128),
        n_kv_head=max(1, ((config.depth * 64) + 127) // 128)
    )
    
    with torch.device("meta"):
        base_model = GPT(model_config)
    model = base_model.to_empty(device="cuda")
    model.init_weights()
    
    if is_main_process:
        logging.info(f"Model: {model_config.n_layer}L / {model_config.n_embd}D / {model_config.n_head}H")

    if config.compile_model:
        if is_main_process:
            logging.info("Compiling the model with torch.compile()...")
        model = torch.compile(model)

    if is_main_process:
        logging.info("Wrapping the custom model with Unsloth...")
    model = FastLanguageModel(model)
    return model, model_config


def prepare_dataset(config: TrainingConfig, tokenizer, is_main_process: bool):
    if is_main_process:
        logging.info(f"Loading dataset '{config.dataset_name}' in streaming mode...")
    dataset = load_dataset(config.dataset_name, name=config.dataset_config, split=config.dataset_split, streaming=True)
    
    def tokenize(examples):
        num_threads = os.cpu_count() or 4
        return {"input_ids": tokenizer.encode(examples["text"], num_threads=num_threads)}

    train_dataset = dataset.map(tokenize, batched=True)
    return train_dataset


def configure_trainer(config: TrainingConfig, model: FastLanguageModel, model_config: GPTConfig, train_dataset, tokenizer, is_main_process: bool):
    if is_main_process:
        logging.info("Configuring Unsloth trainer...")
    
    num_params = sum(p.numel() for p in model.parameters())
    num_steps = (config.target_param_data_ratio * num_params) // config.total_batch_size
    tokens_per_device_step = config.device_batch_size * config.max_seq_len
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = config.total_batch_size // (tokens_per_device_step * world_size)
    
    if is_main_process:
        logging.info(f"Total parameters: {num_params:,}")
        logging.info(f"Calculated training steps: {num_steps:,} with {grad_accum}x accumulation")
    
    dmodel_scale = (model_config.n_embd / 768) ** -0.5
    base_lr = config.base_lr * dmodel_scale
    emb_lr = base_lr * config.embedding_lr_scale
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = config.output_dir_template.format(depth=config.depth)
    
    training_args_dict = {
        "output_dir": output_dir,
        "max_steps": num_steps,
        "per_device_train_batch_size": config.device_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": base_lr,
        "embedding_learning_rate": emb_lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": config.warmup_ratio,
        "optim": "adamw_8bit",
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.grad_clip,
        "bf16": config.use_bf16,
        "logging_steps": 10,
        "save_steps": config.save_every,
        "save_total_limit": 3,
        "dataloader_num_workers": os.cpu_count() or 4,
        "seed": 42,
        "report_to": config.report_to,
    }

    if config.report_to == "wandb":
        wandb_run_name = config.wandb_run_name_template.format(depth=config.depth, timestamp=timestamp)
        training_args_dict["run_name"] = wandb_run_name
        os.environ["WANDB_PROJECT"] = config.wandb_project

    training_args = UnslothTrainingArguments(**training_args_dict)

    tokenizer.enc.pad_token_id = tokenizer.get_bos_token_id()
    
    trainer = UnslothTrainer(
        model=model,
        tokenizer=None,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer.enc, mlm=False),
    )
    return trainer


def main():
    is_main_process = os.environ.get("RANK", "0") == "0"
    parser = get_args_parser()
    args = parser.parse_args()
    config = TrainingConfig(**{k: v for k, v in vars(args).items() if hasattr(TrainingConfig, k)})

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    if is_main_process:
        logging.info("🚀 Starting Optimized NanoChat Pre-training with Unsloth")
    
    model, model_config = create_model(config, vocab_size, is_main_process)
    train_dataset = prepare_dataset(config, tokenizer, is_main_process)
    trainer = configure_trainer(config, model, model_config, train_dataset, tokenizer, is_main_process)
    
    output_dir = trainer.args.output_dir
    if trainer.is_world_process_zero():
        logging.info(f"Output directory: {output_dir}")
        # Save the final configuration for reproducibility
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=4)
        logging.info(f"Saved final training configuration to {config_path}")

    logging.info("🏋️ Starting training...")
    trainer.train()
    trainer.save_model()
    if trainer.is_world_process_zero():
        logging.info(f"✅ Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
