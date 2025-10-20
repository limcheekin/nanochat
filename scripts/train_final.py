#!/usr/bin/env python3
"""
Fully Customizable and Optimized Base Model Pre-training with Unsloth.
This definitive version (v7) patches the model config to be fully compatible.
"""

import os
import torch
import argparse
from dataclasses import dataclass

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

class UnslothCompatibleGPT(GPT):
    """
    An extended version of nanochat's GPT class that includes all
    methods required by the Hugging Face/Unsloth Trainer.
    """
    def get_input_embeddings(self):
        module = self.transformer.wte
        module.dtype = module.weight.dtype
        return module

    def get_output_embeddings(self):
        return self.lm_head

class StreamDatasetWrapper:
    """
    A wrapper to make a Hugging Face IterableDataset compatible with
    the Unsloth/SFT Trainer by explicitly providing a `column_names` attribute.
    """
    def __init__(self, dataset, columns):
        self.dataset = dataset
        self.column_names = columns

    def __iter__(self):
        return iter(self.dataset)

@dataclass
class TrainingConfig:
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
    output_dir: str = "./output"

def main():
    parser = argparse.ArgumentParser(description="Customizable Unsloth NanoChat Trainer")
    parser.add_argument("--depth", type=int, default=20, help="Depth (number of layers) of the Transformer model.")
    parser.add_argument("--device_batch_size", type=int, default=32, help="Per-device batch size (adjust to fit VRAM).")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length for the model.")
    args = parser.parse_args()

    config = TrainingConfig(
        depth=args.depth, max_seq_len=args.max_seq_len, device_batch_size=args.device_batch_size,
    )
    config.output_dir = f"./nanochat_unsloth_d{config.depth}_len{config.max_seq_len}"

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    print(f"🚀 Fully Customizable NanoChat Pre-training with Unsloth (v7)")
    print(f"   Model Depth: {config.depth}, Max Seq Len: {config.max_seq_len}, Batch Size: {config.device_batch_size}")
    
    model_config = GPTConfig(
        sequence_len=config.max_seq_len, vocab_size=vocab_size, n_layer=config.depth,
        n_embd=config.depth * 64, n_head=max(1, ((config.depth * 64) + 127) // 128),
        n_kv_head=max(1, ((config.depth * 64) + 127) // 128)
    )
    
    # ========================== THE FINAL CHANGE IS HERE ==========================
    # We patch the config object to include the attribute the trainer is looking for.
    # We can set it to a dummy name; its value doesn't matter, it just needs to exist.
    model_config._name_or_path = "Custom/nanochat-gpt"
    # ============================================================================
    
    with torch.device("meta"):
        model = UnslothCompatibleGPT(model_config)
    model.to_empty(device="cuda")
    model.init_weights()
    
    print(f"   Model Arch: {model.config.n_layer}L / {model.config.n_embd}D / {model.config.n_head}H")
    
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    def tokenize(examples):
        return {"input_ids": tokenizer.encode(examples["text"], num_threads=8)}

    tokenized_dataset = dataset.map(
        tokenize, batched=True, remove_columns=["text"]
    )
    
    train_dataset = StreamDatasetWrapper(tokenized_dataset, columns=["input_ids"])
    
    num_params = sum(p.numel() for p in model.parameters())
    num_steps = (config.target_param_data_ratio * num_params) // config.total_batch_size
    tokens_per_device_step = config.device_batch_size * config.max_seq_len
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = config.total_batch_size // (tokens_per_device_step * world_size)
    
    print(f"📊 Training: {num_params:,} params, {num_steps:,} steps, {grad_accum}x accumulation")
    
    dmodel_scale = (model_config.n_embd / 768) ** -0.5
    base_lr = config.base_lr * dmodel_scale
    emb_lr = base_lr * config.embedding_lr_scale
    
    training_args = UnslothTrainingArguments(
        output_dir=config.output_dir, max_steps=num_steps,
        per_device_train_batch_size=config.device_batch_size, gradient_accumulation_steps=grad_accum,
        learning_rate=base_lr, embedding_learning_rate=emb_lr, lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio, optim="adamw_8bit", weight_decay=config.weight_decay,
        max_grad_norm=config.grad_clip, bf16=config.use_bf16, logging_steps=10,
        save_steps=config.save_every, save_total_limit=3, dataloader_num_workers=4,
        report_to="wandb", seed=42,
    )
    
    tokenizer.enc.pad_token_id = tokenizer.get_bos_token_id()
    
    trainer = UnslothTrainer(
        model=model,
        tokenizer=None,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer.enc, mlm=False),
    )
    
    print("\n🏋️ Starting training...")
    trainer.train()
    trainer.save_model()
    print(f"✅ Training complete! Model saved to {config.output_dir}")

if __name__ == "__main__":
    main()
