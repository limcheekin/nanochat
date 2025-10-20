import os
import torch
import argparse
from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.tokenizer import get_tokenizer

def convert_unsloth_to_nanochat(unsloth_path, model_tag, step, depth, max_seq_len):
    print(f"Converting checkpoint from: {unsloth_path}")
    print(f"Using Depth: {depth}, Max Seq Len: {max_seq_len}")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    
    # NEW: Use max_seq_len argument in model config
    model_config_kwargs = dict(
        sequence_len=max_seq_len, 
        vocab_size=vocab_size, 
        n_layer=depth,
        n_embd=depth * 64, 
        n_head=max(1, ((depth * 64) + 127) // 128),
        n_kv_head=max(1, ((depth * 64) + 127) // 128)
    )
    
    weights_path = os.path.join(unsloth_path, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu")
    
    unwrapped_state_dict = {
        k[len("base_model.model."):] : v for k, v in state_dict.items()
        if k.startswith("base_model.model.")
    }
    
    meta_data = {
        "step": step, 
        "val_bpb": 0.0, 
        "model_config": model_config_kwargs,
        "user_config": {"depth": depth, "device_batch_size": 32, "max_seq_len": max_seq_len}
    }
    
    base_dir = get_base_dir()
    nanochat_checkpoint_dir = os.path.join(base_dir, "base_checkpoints", model_tag)
    
    save_checkpoint(
        checkpoint_dir=nanochat_checkpoint_dir, 
        step=step,
        model_data=unwrapped_state_dict, 
        optimizer_data=None, 
        meta_data=meta_data
    )
    
    print(f"✅ Successfully converted and saved to: {nanochat_checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Unsloth checkpoint to NanoChat format")
    parser.add_argument("unsloth_path", type=str, help="Path to the directory containing the Unsloth pytorch_model.bin")
    parser.add_argument("--model_tag", type=str, default="d20", help="Tag for the output directory (e.g., d20_len4096)")
    parser.add_argument("--step", type=int, default=999999, help="Step number for the checkpoint filename")
    parser.add_argument("--depth", type=int, default=20, help="Model depth used during training")
    # NEW: Added max_seq_len argument
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length the model was trained with.")
    args = parser.parse_args()
    convert_unsloth_to_nanochat(args.unsloth_path, args.model_tag, args.step, args.depth, args.max_seq_len)
