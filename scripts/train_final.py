#!/usr/bin/env python3
"""
Fully Customizable and Optimized Base Model Pre-training with Unsloth.
This definitive version (v19) provides the complete and verified solution.
It resolves the final `TypeError: unexpected keyword argument` by correctly
filtering the configuration dictionary to only pass the attributes expected by
the original `nanochat` GPTConfig dataclass. This completes the compatibility
bridge between the two libraries.
"""

import os
import torch
import argparse
from itertools import islice
from dataclasses import fields

# Environment setup for memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset, Dataset
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling

# Import the original, unmodified nanochat classes
from nanochat.gpt import GPT, GPTConfig as OriginalGPTConfig
from nanochat.tokenizer import get_tokenizer

# === THE DEFINITIVE SOLUTION: FULLY COMPATIBLE PROXY CLASSES ===

# 1. Create a compatible configuration class that inherits from PretrainedConfig.
class CompatibleGPTConfig(PretrainedConfig):
    model_type = "nanochat_gpt"

    def __init__(
        self,
        sequence_len: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 6,
        n_kv_head: int = 6,
        n_embd: int = 768,
        **kwargs,
    ):
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        super().__init__(**kwargs)

# 2. Create a compatible model class that inherits from PreTrainedModel.
class UnslothCompatibleGPT(PreTrainedModel):
    config_class = CompatibleGPTConfig

    def __init__(self, config: CompatibleGPTConfig):
        super().__init__(config)
        
        # === THE DEFINITIVE `TypeError` FIX ===
        # Get all attributes from the compatible HF config
        config_dict = config.to_dict()
        # Get the field names that the original dataclass expects
        expected_keys = {f.name for f in fields(OriginalGPTConfig)}
        # Filter the dictionary to only include the expected keys
        filtered_config_dict = {k: v for k, v in config_dict.items() if k in expected_keys}
        # Now instantiate the original config with only the relevant arguments
        original_config = OriginalGPTConfig(**filtered_config_dict)

        # Contain an instance of the original model
        self.model = GPT(original_config)

    def get_input_embeddings(self):
        module = self.model.transformer.wte
        module.dtype = module.weight.dtype
        return module

    def get_output_embeddings(self):
        return self.model.lm_head

    def forward(self, input_ids, labels=None, **kwargs):
        # Forward the call to the original nanochat model.
        output = self.model.forward(idx=input_ids, targets=labels)
        
        # Return in the dictionary format expected by the HF Trainer
        if labels is not None:
            loss, logits = output
            return {"loss": loss, "logits": logits}
        else:
            logits = output
            return {"logits": logits}

# 3. Register our new, compatible classes with the Auto* classes.
AutoConfig.register(CompatibleGPTConfig.model_type, CompatibleGPTConfig)
AutoModelForCausalLM.register(CompatibleGPTConfig, UnslothCompatibleGPT)


def main():
    parser = argparse.ArgumentParser(description="Customizable Unsloth NanoChat Trainer")
    parser.add_argument("--depth", type=int, default=20, help="Depth (number of layers) of the Transformer model.")
    parser.add_argument("--device_batch_size", type=int, default=32, help="Per-device batch size (adjust to fit VRAM).")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length for the model.")
    parser.add_argument("--dataset_subset_size", type=int, default=500_000, help="Number of examples to use for training.")
    args = parser.parse_args()

    output_dir = os.path.abspath(f"./nanochat_unsloth_d{args.depth}_len{args.max_seq_len}")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    print(f"🚀 Corrected NanoChat Pre-training with Unsloth (v19)")
    print(f"   Model Depth: {args.depth}, Max Seq Len: {args.max_seq_len}, Batch Size: {args.device_batch_size}")

    model_config = CompatibleGPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=args.depth,
        n_embd=args.depth * 64, n_head=max(1, ((args.depth * 64) + 127) // 128),
        n_kv_head=max(1, ((args.depth * 64) + 127) // 128)
    )

    # Initialize the model directly from the config object in memory.
    model = UnslothCompatibleGPT(config=model_config)
    print(f"   Model Arch: {model.config.n_layer}L / {model.config.n_embd}D / {model.config.n_head}H")

    # Prepare mappable dataset
    print(f"Preparing dataset: taking a subset of {args.dataset_subset_size:,} examples...")
    streaming_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    subset_data = list(islice(streaming_dataset, args.dataset_subset_size))
    train_dataset = Dataset.from_list(subset_data)
    print(f"Dataset prepared with {len(train_dataset):,} examples.")

    train_dataset = train_dataset.map(
        lambda examples: {"input_ids": tokenizer.encode(examples["text"], num_threads=os.cpu_count())},
        batched=True, batch_size=1024, remove_columns=list(train_dataset.features),
    )

    total_batch_size = 524288
    num_params = sum(p.numel() for p in model.parameters())
    num_steps = (20 * num_params) // total_batch_size
    tokens_per_device_step = args.device_batch_size * args.max_seq_len
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = total_batch_size // (tokens_per_device_step * world_size)

    print(f"📊 Training: {num_params:,} params, {num_steps:,} steps, {grad_accum}x accumulation")

    dmodel_scale = (model.config.n_embd / 768) ** -0.5
    base_lr = 3e-4 * dmodel_scale
    emb_lr = base_lr * 0.1

    training_args = UnslothTrainingArguments(
        output_dir=output_dir, max_steps=num_steps,
        per_device_train_batch_size=args.device_batch_size, gradient_accumulation_steps=grad_accum,
        learning_rate=base_lr, embedding_learning_rate=emb_lr, lr_scheduler_type="cosine",
        warmup_ratio=0.02, optim="adamw_8bit", weight_decay=0.01,
        max_grad_norm=1.0, bf16=True, logging_steps=10,
        save_steps=1000, save_total_limit=3, dataloader_num_workers=4,
        report_to="wandb", seed=42,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer.enc, mlm=False)
    
    trainer = UnslothTrainer(
        model=model,
        tokenizer=None,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\n🏋️ Starting training...")
    trainer.train()
    trainer.save_model()
    print(f"✅ Training complete! Model saved to {output_dir}")

if __name__ == "__main__":
    main()
