# convert_checkpoint.py

import os
import sys
import torch
import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.tokenizer import get_tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class CheckpointMetadata:
    source_dir: Path
    weights_path: Path
    depth: int
    step: int
    batch_size: int
    model_config_json: dict


def find_latest_checkpoint(unsloth_dir: Path):
    """Finds the latest checkpoint directory if any."""
    checkpoints = sorted(
        [d for d in unsloth_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
        reverse=True
    )
    return checkpoints[0] if checkpoints else None

def discover_metadata(unsloth_path: str, depth_override: int | None, step_override: int | None) -> CheckpointMetadata | None:
    """Discovers metadata from the Unsloth checkpoint directory."""
    unsloth_dir = Path(unsloth_path)
    if not unsloth_dir.is_dir():
        logging.error(f"Input path is not a directory: {unsloth_dir}")
        sys.exit(1)

    source_dir = find_latest_checkpoint(unsloth_dir) or unsloth_dir
    logging.info(f"Reading metadata from: {source_dir}")

    config_path = source_dir / "config.json"
    trainer_state_path = source_dir / "trainer_state.json"
    training_args_path = unsloth_dir / "training_args.json"
    weights_path = source_dir / "pytorch_model.bin"

    if not all(p.exists() for p in [config_path, weights_path]):
        logging.error(f"Could not find required files (config.json, pytorch_model.bin) in {source_dir}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        model_config_json = json.load(f)

    detected_depth = model_config_json.get("n_layer")
    final_depth = depth_override if depth_override is not None else detected_depth
    if final_depth is None:
        logging.error("Could not determine model depth from config.json. Please provide it with --depth.")
        sys.exit(1)

    final_step = step_override
    if final_step is None and trainer_state_path.exists():
        with open(trainer_state_path, 'r') as f:
            final_step = json.load(f).get("global_step")
    if final_step is None:
        logging.warning("Could not determine step from trainer_state.json, using default 999999.")
        final_step = 999999

    batch_size = 32  # Default
    if training_args_path.exists():
        with open(training_args_path, 'r') as f:
            batch_size = json.load(f).get("per_device_train_batch_size", 32)
    
    return CheckpointMetadata(
        source_dir=source_dir,
        weights_path=weights_path,
        depth=final_depth,
        step=final_step,
        batch_size=batch_size,
        model_config_json=model_config_json
    )

def find_state_dict_prefix(state_dict):
    """Dynamically finds the common prefix in the state dictionary keys."""
    if not state_dict:
        return ""
    
    keys = list(state_dict.keys())
    common_prefix = os.path.commonprefix(keys)
    # Ensure the prefix ends at a dot, to avoid partial names.
    if '.' in common_prefix:
        return common_prefix.rsplit('.', 1)[0] + '.'
    return ""

def convert_unsloth_to_nanochat(unsloth_path: str, model_tag_override: str | None, step_override: int | None, depth_override: int | None):
    metadata = discover_metadata(unsloth_path, depth_override, step_override)
    if not metadata:
        return # Error already logged by discover_metadata

    final_model_tag = model_tag_override if model_tag_override is not None else f"d{metadata.depth}"

    logging.info(f"Converting checkpoint from: {metadata.source_dir}")
    logging.info(f"Using depth={metadata.depth}, step={metadata.step}, model_tag='{final_model_tag}'")

    tokenizer = get_tokenizer()
    
    model_config_kwargs = dict(
        sequence_len=metadata.model_config_json.get("max_position_embeddings", 2048),
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=metadata.depth,
        n_embd=metadata.depth * 64,
        n_head=max(1, ((metadata.depth * 64) + 127) // 128),
        n_kv_head=max(1, ((metadata.depth * 64) + 127) // 128)
    )
    
    state_dict = torch.load(metadata.weights_path, map_location="cpu")
    
    prefix = find_state_dict_prefix(state_dict)
    if prefix:
        logging.info(f"Detected and removing state_dict prefix: '{prefix}'")
        unwrapped_state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    else:
        logging.warning("Could not detect a common prefix. Using original state_dict.")
        unwrapped_state_dict = state_dict

    meta_data = {
        "step": metadata.step,
        "val_bpb": 0.0,
        "model_config": model_config_kwargs,
        "user_config": {"depth": metadata.depth, "device_batch_size": metadata.batch_size}
    }
    
    base_dir = get_base_dir()
    nanochat_checkpoint_dir = Path(base_dir) / "base_checkpoints" / final_model_tag
    
    save_checkpoint(
        checkpoint_dir=str(nanochat_checkpoint_dir),
        step=metadata.step,
        model_data=unwrapped_state_dict,
        optimizer_data=None,
        meta_data=meta_data
    )
    
    logging.info(f"✅ Successfully converted and saved to: {nanochat_checkpoint_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Unsloth checkpoint to NanoChat format.")
    parser.add_argument("unsloth_path", type=str, help="Path to the Unsloth output directory.")
    parser.add_argument("--model-tag", type=str, default=None, help="Override tag for the output directory (e.g., 'd20'). Auto-detected.")
    parser.add_argument("--step", type=int, default=None, help="Override step number for the checkpoint. Auto-detected.")
    parser.add_argument("--depth", type=int, default=None, help="Override model depth. Auto-detected.")
    args = parser.parse_args()
    convert_unsloth_to_nanochat(args.unsloth_path, args.model_tag, args.step, args.depth)
