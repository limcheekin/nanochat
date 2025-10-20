#!/usr/bin/env python3
"""
Fully Customizable and Optimized Base Model Pre-training with Unsloth.
This definitive version (v25) provides the complete and verified solution.
It re-introduces the `_supports_gradient_checkpointing` flag to the model
wrapper, which was inadvertently removed. This is the final required piece of
metadata for full compatibility with the trainer's API.
"""

import os
import torch
import argparse
from itertools import islice
from dataclasses import fields
from typing import List, Dict

# Environment setup for memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset, Dataset
from transformers import (
    PretrainedConfig, PreTrainedModel, PreTrainedTokenizer,
    AutoConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling,
)

# Import the original, unmodified nanochat classes
from nanochat.gpt import GPT, GPTConfig as OriginalGPTConfig
from nanochat.tokenizer import get_tokenizer, RustBPETokenizer

# === THE DEFINITIVE SOLUTION: FULLY COMPATIBLE PROXY AND WRAPPER CLASSES ===

# 1. Create a compatible configuration class
class CompatibleGPTConfig(PretrainedConfig):
    model_type = "nanochat_gpt"
    def __init__(self, sequence_len=1024, vocab_size=50304, n_layer=12, n_head=6, n_kv_head=6, n_embd=768, **kwargs):
        self.sequence_len, self.vocab_size, self.n_layer, self.n_head, self.n_kv_head, self.n_embd = \
            sequence_len, vocab_size, n_layer, n_head, n_kv_head, n_embd
        super().__init__(**kwargs)

# 2. Create a compatible model class that contains the original GPT
class UnslothCompatibleGPT(PreTrainedModel):
    config_class = CompatibleGPTConfig

    # === THE DEFINITIVE `ValueError` FIX for Gradient Checkpointing ===
    # We must explicitly declare support for this feature.
    _supports_gradient_checkpointing = True

    def __init__(self, config: CompatibleGPTConfig):
        super().__init__(config)
        config_dict = config.to_dict()
        expected_keys = {f.name for f in fields(OriginalGPTConfig)}
        filtered_config_dict = {k: v for k, v in config_dict.items() if k in expected_keys}
        self.model = GPT(OriginalGPTConfig(**filtered_config_dict))
    
    def get_input_embeddings(self):
        module = self.model.transformer.wte
        module.dtype = module.weight.dtype
        return module
    def get_output_embeddings(self): return self.model.lm_head
    def forward(self, input_ids, labels=None, **kwargs):
        output = self.model.forward(idx=input_ids, targets=labels)
        loss, logits = output if labels is not None else (None, output)
        return {"loss": loss, "logits": logits}

# 3. Create a compatible tokenizer wrapper
class NanoChatTokenizerWrapper(PreTrainedTokenizer):
    def __init__(self, nanochat_tokenizer: RustBPETokenizer, **kwargs):
        self.nanochat_tokenizer = nanochat_tokenizer
        kwargs["pad_token"] = "<|bos|>"
        kwargs["bos_token"] = "<|bos|>"
        super().__init__(**kwargs)
    @property
    def vocab_size(self) -> int: return self.nanochat_tokenizer.get_vocab_size()
    @property
    def pad_token_id(self) -> int: return self.nanochat_tokenizer.enc.pad_token_id
    @pad_token_id.setter
    def pad_token_id(self, value: int): self.nanochat_tokenizer.enc.pad_token_id = value
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        ids = self.nanochat_tokenizer.encode(text)
        return [self.nanochat_tokenizer.decode([i]) for i in ids]
    def _convert_token_to_id(self, token: str) -> int: return self.nanochat_tokenizer.encode(token)[0]
    def get_vocab(self) -> Dict[str, int]: return {self.nanochat_tokenizer.decode([i]): i for i in range(self.vocab_size)}
    def save_vocabulary(self, sd: str, fp: str | None = None) -> tuple[str,]: return (os.path.join(sd, "vocab.txt"),)
    def build_inputs_with_special_tokens(self, t0: List[int], t1: List[int] | None = None) -> List[int]: return t0

# 4. Register custom classes
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
    original_tokenizer = get_tokenizer()
    vocab_size = original_tokenizer.get_vocab_size()

    hf_tokenizer = NanoChatTokenizerWrapper(original_tokenizer, pad_token_id=original_tokenizer.get_bos_token_id())

    print(f"🚀 Corrected NanoChat Pre-training with Unsloth (v25)")
    print(f"   Model Depth: {args.depth}, Max Seq Len: {args.max_seq_len}, Batch Size: {args.device_batch_size}")

    model_config = CompatibleGPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=args.depth,
        n_embd=args.depth * 64, n_head=max(1, ((args.depth * 64) + 127) // 128),
        n_kv_head=max(1, ((args.depth * 64) + 127) // 128),
        name_or_path=output_dir,
    )
    model = UnslothCompatibleGPT(config=model_config)
    print(f"   Model Arch: {model.config.n_layer}L / {model.config.n_embd}D / {model.config.n_head}H")

    print(f"Preparing dataset: taking a subset of {args.dataset_subset_size:,} examples...")
    streaming_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    subset_data = list(islice(streaming_dataset, args.dataset_subset_size))
    train_dataset = Dataset.from_list(subset_data)
    print(f"Dataset prepared with {len(train_dataset):,} examples.")

    train_dataset = train_dataset.map(
        lambda examples: {"input_ids": original_tokenizer.encode(examples["text"], num_threads=os.cpu_count())},
        batched=True, batch_size=1024, remove_columns=list(train_dataset.features),
    )

    total_batch_size, target_param_data_ratio, base_lr, embedding_lr_scale = 524288, 20, 3e-4, 0.1
    num_params = sum(p.numel() for p in model.parameters())
    num_steps = (target_param_data_ratio * num_params) // total_batch_size
    tokens_per_device_step = args.device_batch_size * args.max_seq_len
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = total_batch_size // (tokens_per_device_step * world_size)

    print(f"📊 Training: {num_params:,} params, {num_steps:,} steps, {grad_accum}x accumulation")

    dmodel_scale = (model.config.n_embd / 768) ** -0.5
    lr = base_lr * dmodel_scale
    emb_lr = lr * embedding_lr_scale

    training_args = UnslothTrainingArguments(
        output_dir=output_dir, max_steps=num_steps,
        per_device_train_batch_size=args.device_batch_size, gradient_accumulation_steps=grad_accum,
        learning_rate=lr, embedding_learning_rate=emb_lr, lr_scheduler_type="cosine",
        warmup_ratio=0.02, optim="adamw_8bit", weight_decay=0.01,
        max_grad_norm=1.0, bf16=True, logging_steps=10,
        save_steps=1000, save_total_limit=3, dataloader_num_workers=4,
        report_to="wandb", seed=42,
        # The trainer enables this by default if it is supported
        # gradient_checkpointing = True, 
    )
    
    data_collator = DataCollatorForLanguageModeling(hf_tokenizer, mlm=False)
    os.makedirs(output_dir, exist_ok=True)
    
    trainer = UnslothTrainer(
        model=model,
        tokenizer=hf_tokenizer,
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
