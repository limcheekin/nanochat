# Comprehensive Step-by-Step Guide: Optimized Base Model Pre-training with Unsloth

## Overview

This guide provides a systematic approach to implement an optimized version of Karpathy's NanoChat base model pre-training (`python -m scripts.base_train --depth=20 --device_batch_size=32`) using Unsloth. The original implementation uses a custom dual-optimizer setup (Muon + AdamW) and a specific GPT architecture. We will train this **exact architecture** with Unsloth to maintain full compatibility with the original evaluation scripts, while benefiting from significant speed and memory improvements.

## Table of Contents

1.  [Understanding the Original Architecture](#1-understanding-the-original-architecture)
2.  [Unsloth Optimization Strategy](#2-unsloth-optimization-strategy)
3.  [Environment Setup](#3-environment-setup)
4.  [Model Configuration](#4-model-configuration)
5.  [Data Preparation](#5-data-preparation)
6.  [Training Implementation](#6-training-implementation)
7.  [Training Execution](#7-training-execution)
8.  [Post-Training Conversion](#8-post-training-conversion)
9.  [The Complete Workflow](#9-the-complete-workflow)
10. [Advanced Optimizations](#10-advanced-optimizations)

***

## 1. Understanding the Original Architecture

### Model Specifications (depth=20)

The original NanoChat implementation with `depth=20` creates a custom GPT model with approximately **561M parameters** and the following features:

**Architecture Parameters:**

*   `n_layer`: 20 layers
*   `n_embd`: 1,280 (depth × 64 aspect ratio)
*   `n_head`: 10 heads
*   `n_kv_head`: 10 (1:1 MQA ratio)
*   `max_seq_len`: 2,048 tokens

**Key Features:** Our objective is to train a model that is **architecturally identical** to this specification to ensure compatibility with the original evaluation scripts.
*   Rotary Position Embeddings (RoPE)
*   QK normalization for stable attention
*   Untied `wte` and `lm_head` weights
*   ReLU² activation in MLP layers
*   Functional RMSNorm without learnable parameters
*   No bias terms in linear layers

### Original Training Configuration

*   **Batch Configuration:** `device_batch_size` of 32, `total_batch_size` of 524,288 tokens, with automatic gradient accumulation.
*   **Dual-Optimizer Setup:** AdamW for embeddings and Muon for transformer linear layers.
*   **Learning Rate Schedule:** A custom schedule with a 20% linear warmdown.

***

## 2. Unsloth Optimization Strategy

Unsloth offers significant advantages for base model pre-training:

*   **Memory Efficiency**: Up to 80% VRAM reduction.
*   **Speed**: Up to 2x faster training.
*   **Flexibility**: Can wrap and optimize custom `torch.nn.Module` classes, not just Hugging Face models.

### Adaptation Strategy

Since the Muon optimizer is not standard, we will use Unsloth's highly optimized **8-bit AdamW** with decoupled embedding learning rates. This provides excellent performance and memory efficiency. Crucially, instead of approximating the architecture with a standard model, we will wrap the original `nanochat.gpt.GPT` class directly with Unsloth.

```python
from nanochat.gpt import GPT
from unsloth import FastLanguageModel

# Directly wrap the custom nanochat model
custom_nanochat_model = GPT(model_config)
model = FastLanguageModel(custom_nanochat_model)
```

This approach guarantees that the trained model has the correct architecture.

***

## 3. Environment Setup

### Step 3.1: Install Dependencies

```bash
# Install Unsloth for your CUDA version
# For CUDA 12.1
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

# For CUDA 11.8
pip install "unsloth[cu118-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Install nanochat dependencies from the pyproject.toml
# Ensure you have 'uv' installed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Build the custom RustBPE tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Step 3.2: Train the NanoChat Tokenizer

Before training the model, you must first train the tokenizer, as the training script depends on it.

```bash
# This downloads data and saves the tokenizer to ~/.cache/nanochat/tokenizer
python -m scripts.tok_train --max_chars=2000000000
```

### Step 3.3: Configure Environment Variables

```python
import os

# Memory optimization for PyTorch CUDA allocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

***

## 4. Model Configuration

### Step 4.1: Define Training Configuration

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for NanoChat pre-training"""
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
    output_dir: str = "./nanochat_unsloth_d20"
```

### Step 4.2: Initialize the NanoChat Model and Wrap with Unsloth

```python
import torch
from unsloth import FastLanguageModel
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

# Initialize config and load the pre-trained tokenizer
config = TrainingConfig()
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

# 1. Instantiate the correct nanochat.gpt.GPT model
model_config = GPTConfig(
    sequence_len=config.max_seq_len,
    vocab_size=vocab_size,
    n_layer=config.depth,
    n_embd=config.depth * 64,
    n_head=max(1, ((config.depth * 64) + 127) // 128),
    n_kv_head=max(1, ((config.depth * 64) + 127) // 128)
)

# Create model on meta device, then materialize on GPU
with torch.device("meta"):
    base_model = GPT(model_config)

model = base_model.to_empty(device="cuda")
model.init_weights() # Use nanochat's own weight initialization

print(f"Model: {model_config.n_layer}L / {model_config.n_embd}D / {model_config.n_head}H")

# 2. Wrap the custom model with Unsloth for optimization
model = FastLanguageModel(model)
```

***

## 5. Data Preparation

### Step 5.1: Dataset Loading and Tokenization

We use the pre-trained `RustBPETokenizer` to prepare the data.

```python
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

# Load dataset in streaming mode
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

def tokenize_function(examples):
    # Use the nanochat tokenizer's batch encoding method
    return {"input_ids": tokenizer.encode(examples["text"], num_threads=8)}

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Create data collator
# The transformers.DataCollator needs a pad_token_id.
# We dynamically add it, using the BOS token ID for padding.
tokenizer.enc.pad_token_id = tokenizer.get_bos_token_id()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer.enc, # Pass the underlying tiktoken encoder
    mlm=False,
)
```

***

## 6. Training Implementation

### Step 6.1: Calculate Training Parameters

```python
def calculate_training_steps(config: TrainingConfig, model):
    num_params = sum(p.numel() for p in model.parameters())
    target_tokens = config.target_param_data_ratio * num_params
    num_steps = target_tokens // config.total_batch_size
    
    tokens_per_device = config.device_batch_size * config.max_seq_len
    num_devices = torch.cuda.device_count()
    world_tokens_per_step = tokens_per_device * num_devices
    gradient_accumulation_steps = config.total_batch_size // world_tokens_per_step
    
    print(f"\n📊 Training Configuration:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Training steps: {num_steps:,}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    
    return num_steps, gradient_accumulation_steps

num_steps, grad_accum_steps = calculate_training_steps(config, model)
```

### Step 6.2: Configure Unsloth Training Arguments

```python
from unsloth import UnslothTrainingArguments

# Calculate learning rates with d_model scaling
dmodel_lr_scale = (model_config.n_embd / 768) ** -0.5
scaled_base_lr = config.base_lr * dmodel_lr_scale
scaled_embedding_lr = scaled_base_lr * config.embedding_lr_scale

training_args = UnslothTrainingArguments(
    output_dir=config.output_dir,
    max_steps=num_steps,
    per_device_train_batch_size=config.device_batch_size,
    gradient_accumulation_steps=grad_accum_steps,
    learning_rate=scaled_base_lr,
    embedding_learning_rate=scaled_embedding_lr,
    lr_scheduler_type="cosine",
    warmup_ratio=config.warmup_ratio,
    optim="adamw_8bit",
    weight_decay=config.weight_decay,
    max_grad_norm=config.grad_clip,
    bf16=config.use_bf16,
    logging_steps=10,
    save_strategy="steps",
    save_steps=config.save_every,
    save_total_limit=3,
    dataloader_num_workers=4,
    report_to="wandb",
    seed=42,
)
```

***

## 7. Training Execution

### Step 7.1: Create Unsloth Trainer and Run Training

The complete, corrected script is provided below. Save it as `train_unsloth_corrected.py` and run it.

```python
#!/usr/bin/env python3
"""
Corrected and Optimized Base Model Pre-training with Unsloth
This script trains an architecturally-identical model to karpathy-nanochat.
"""

import os
import torch
from dataclasses import dataclass

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from unsloth import FastLanguageModel
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

# === CRITICAL: Import the correct model architecture and tokenizer ===
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer # This will load the pre-trained RustBPE tokenizer

@dataclass
class TrainingConfig:
    """Configuration for NanoChat pre-training"""
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
    output_dir: str = "./nanochat_unsloth_d20"

def main():
    config = TrainingConfig()
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    print(f"🚀 Corrected NanoChat Pre-training with Unsloth")
    
    # 1. Instantiate the correct nanochat.gpt.GPT model
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
    
    print(f"   Model: {model_config.n_layer}L / {model_config.n_embd}D / {model_config.n_head}H")

    # 2. Wrap the custom model with Unsloth
    model = FastLanguageModel(model)
    
    # Load and tokenize dataset
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    def tokenize(examples):
        return {"input_ids": tokenizer.encode(examples["text"], num_threads=8)}

    train_dataset = dataset.map(tokenize, batched=True)
    
    # Calculate training steps
    num_params = sum(p.numel() for p in model.parameters())
    num_steps = (config.target_param_data_ratio * num_params) // config.total_batch_size
    tokens_per_device_step = config.device_batch_size * config.max_seq_len
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accum = config.total_batch_size // (tokens_per_device_step * world_size)
    
    print(f"📊 Training: {num_steps:,} steps, {grad_accum}x accumulation")
    
    # Learning rates
    dmodel_scale = (model_config.n_embd / 768) ** -0.5
    base_lr = config.base_lr * dmodel_scale
    emb_lr = base_lr * config.embedding_lr_scale
    
    training_args = UnslothTrainingArguments(
        output_dir=config.output_dir,
        max_steps=num_steps,
        per_device_train_batch_size=config.device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=base_lr,
        embedding_learning_rate=emb_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        optim="adamw_8bit",
        weight_decay=config.weight_decay,
        max_grad_norm=config.grad_clip,
        bf16=config.use_bf16,
        logging_steps=10,
        save_steps=config.save_every,
        save_total_limit=3,
        dataloader_num_workers=4,
        report_to="wandb",
        seed=42,
    )
    
    # Dynamically set pad_token_id for the data collator
    tokenizer.enc.pad_token_id = tokenizer.get_bos_token_id()
    
    trainer = UnslothTrainer(
        model=model,
        tokenizer=None, # Dataset is pre-tokenized
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
```

### Step 7.2: Launch Distributed Training

For multi-GPU training, use `torchrun`:

```bash
# Save the script above as train_unsloth_corrected.py, then run:
torchrun --standalone --nproc_per_node=8 train_unsloth_corrected.py
```

***

## 8. Post-Training Conversion

After training, the saved checkpoint is in the Hugging Face format. We must convert it to the specific format the original evaluation scripts expect.

### Step 8.1: Create the Conversion Script

Save the following code as `convert_checkpoint.py`. This script repackages the weights and creates the required `meta.json` file.

```python
# convert_checkpoint.py

import os
import torch
import argparse
from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.tokenizer import get_tokenizer

def convert_unsloth_to_nanochat(unsloth_path, model_tag, step, depth):
    print(f"Converting checkpoint from: {unsloth_path}")

    # 1. Define the Nanochat GPTConfig by loading the actual tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    
    model_config_kwargs = dict(
        sequence_len=2048, vocab_size=vocab_size, n_layer=depth,
        n_embd=depth * 64, n_head=max(1, ((depth * 64) + 127) // 128),
        n_kv_head=max(1, ((depth * 64) + 127) // 128))
    
    # 2. Load the state dictionary from the Unsloth checkpoint
    weights_path = os.path.join(unsloth_path, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Unsloth adds a "base_model.model." prefix; remove it.
    unwrapped_state_dict = {
        k[len("base_model.model."):] : v for k, v in state_dict.items()
        if k.startswith("base_model.model.")
    }
    
    # 3. Create the metadata file expected by nanochat
    meta_data = {
        "step": step, "val_bpb": 0.0, "model_config": model_config_kwargs,
        "user_config": {"depth": depth, "device_batch_size": 32}
    }
    
    # 4. Use nanochat's own checkpoint manager to save correctly
    base_dir = get_base_dir()
    nanochat_checkpoint_dir = os.path.join(base_dir, "base_checkpoints", model_tag)
    
    save_checkpoint(
        checkpoint_dir=nanochat_checkpoint_dir, step=step,
        model_data=unwrapped_state_dict, optimizer_data=None, meta_data=meta_data)
    
    print(f"✅ Successfully converted and saved to: {nanochat_checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Unsloth checkpoint to NanoChat format")
    parser.add_argument("unsloth_path", type=str, help="Path to the directory containing the Unsloth pytorch_model.bin")
    parser.add_argument("--model_tag", type=str, default="d20", help="Tag for the output directory (e.g., d20)")
    parser.add_argument("--step", type=int, default=999999, help="Step number for the checkpoint filename")
    parser.add_argument("--depth", type=int, default=20, help="Model depth used during training")
    args = parser.parse_args()
    convert_unsloth_to_nanochat(args.unsloth_path, args.model_tag, args.step, args.depth)
```

***

## 9. The Complete Workflow

Follow these steps in order to train your model and evaluate it using the original, unmodified scripts.

**Step 1: Train the Tokenizer**
```bash
python -m scripts.tok_train --max_chars=2000000000
```

**Step 2: Run the Corrected Unsloth Training**```bash
torchrun --standalone --nproc_per_node=8 train_unsloth_corrected.py
```

**Step 3: Convert the Final Checkpoint**
```bash
# This converts the Unsloth output to the format needed by nanochat's eval scripts
# The path should point to the directory where Unsloth saved the model.
python convert_checkpoint.py ./nanochat_unsloth_d20 --model_tag d20 --step 999999 --depth 20
```

**Step 4: Run Unmodified Original Evaluation Scripts**
The scripts will automatically find the converted checkpoint in the `~/.cache/nanochat/base_checkpoints/d20` directory.
```bash
# Evaluate loss and generate samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss

# Evaluate on CORE tasks
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

***

## 10. Advanced Optimizations

For even larger-scale training, consider these techniques, which are compatible with the workflow above.

*   **`torch.compile`:** For a potential 10-20% speedup, wrap the model with `model = torch.compile(model)` *before* passing it to `FastLanguageModel`. Note this adds a one-time compilation overhead at the start of training.
*   **FSDP (Fully Sharded Data Parallel):** For training models that don't fit on a single node, you can integrate FSDP. In your `UnslothTrainingArguments`, add:
    ```python
    fsdp="full_shard auto_wrap",
    fsdp_config={"transformer_layer_cls_to_wrap": ["Block"]}, # Note: "Block" is the class from nanochat.gpt
    ```
*   **DeepSpeed ZeRO:** Unsloth is also compatible with DeepSpeed stages 2 and 3 for offloading optimizer states and gradients to CPU memory, allowing for training larger models with less VRAM.
