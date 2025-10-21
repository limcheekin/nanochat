#!/usr/bin/env python3
"""
Quick test to verify that the ignore_index fix works correctly.
This test simulates what happens during training with the DataCollatorForLanguageModeling.
"""

import torch
import torch.nn.functional as F

def test_ignore_index():
    """Test that ignore_index=-100 correctly ignores padded positions."""
    
    # Simulate a small batch
    batch_size = 2
    seq_len = 5
    vocab_size = 100
    
    # Create dummy logits (model output)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Create labels with -100 for padding (as DataCollatorForLanguageModeling does)
    labels = torch.tensor([
        [10, 20, 30, -100, -100],  # First sequence: 3 real tokens, 2 padding
        [15, 25, 35, 45, -100],     # Second sequence: 4 real tokens, 1 padding
    ])
    
    print("Testing ignore_index=-100 (CORRECT):")
    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels:\n{labels}")
    
    # Calculate loss with ignore_index=-100 (CORRECT)
    try:
        loss_correct = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        print(f"✅ Loss with ignore_index=-100: {loss_correct.item():.4f}")
        print("   This correctly ignores positions with -100")
    except Exception as e:
        print(f"❌ Error with ignore_index=-100: {e}")
    
    print("\nTesting ignore_index=-1 (INCORRECT):")
    # Calculate loss with ignore_index=-1 (INCORRECT - this was the bug)
    try:
        loss_incorrect = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-1,
            reduction='mean'
        )
        print(f"⚠️  Loss with ignore_index=-1: {loss_incorrect.item():.4f}")
        print("   This does NOT ignore positions with -100!")
        print("   The -100 values are treated as class indices, causing CUDA errors!")
    except Exception as e:
        print(f"❌ Error with ignore_index=-1: {e}")
    
    print("\n" + "="*70)
    print("EXPLANATION:")
    print("="*70)
    print("When DataCollatorForLanguageModeling creates labels, it uses -100 for")
    print("padding positions. If the model uses ignore_index=-1, these -100 values")
    print("are NOT ignored and are treated as class indices, which causes:")
    print("  - CUDA assertion error: `t >= 0 && t < n_classes`")
    print("  - Because -100 is not in the valid range [0, vocab_size)")
    print("\nThe fix: Use ignore_index=-100 to match HuggingFace's convention.")
    print("="*70)

if __name__ == "__main__":
    test_ignore_index()

