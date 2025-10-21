# Fix for CUDA Assertion Error: `t >= 0 && t < n_classes`

## Problem

The training script was failing with the following CUDA error:

```
/pytorch/aten/src/ATen/native/cuda/Loss.cu:245: nll_loss_forward_reduce_cuda_kernel_2d: block: [0,0,0], thread: [854,0,0] Assertion `t >= 0 && t < n_classes` failed.
```

This error occurs when the model encounters a target label that is either:
1. Negative (but not the ignore_index)
2. Greater than or equal to the number of classes (vocab_size)

## Root Cause

The issue was a **mismatch between the ignore_index used by HuggingFace's DataCollatorForLanguageModeling and the ignore_index expected in the model's forward method**.

- **DataCollatorForLanguageModeling** uses `-100` as the default `ignore_index`
- **The model's forward method** was using `-1` as the `ignore_index`

When the data collator created labels with `-100` for padding positions, the model's loss function didn't recognize these as positions to ignore. Instead, it tried to use `-100` as a valid class index, which triggered the CUDA assertion error.

## Solution

Changed the `ignore_index` parameter in the model's `cross_entropy` loss calculation from `-1` to `-100` to match HuggingFace's default:

```python
# Before (INCORRECT):
loss = torch.nn.functional.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=-1,  # ❌ Doesn't match DataCollatorForLanguageModeling
    reduction='mean'
)

# After (CORRECT):
loss = torch.nn.functional.cross_entropy(
    logits.view(-1, logits.size(-1)),
    labels.view(-1),
    ignore_index=-100,  # ✅ Matches DataCollatorForLanguageModeling default
    reduction='mean'
)
```

## Additional Improvements

1. **Fixed type error**: Changed `num_threads=os.cpu_count()` to `num_threads=os.cpu_count() or 1` to handle the case where `os.cpu_count()` returns `None`.

2. **Added validation**: Added a validation step after tokenization to check that all token IDs are within the valid range `[0, vocab_size)`.

## Files Modified

- `scripts/train_final.py`:
  - Line 130: Changed `ignore_index=-1` to `ignore_index=-100`
  - Line 200: Changed `num_threads=os.cpu_count()` to `num_threads=os.cpu_count() or 1`
  - Lines 204-210: Added validation code to check token IDs

## How to Test

Run the training script again:

```bash
python scripts/train_final.py --depth 20 --device_batch_size 32 --max_seq_len 2048 --dataset_subset_size 500000
```

The CUDA assertion error should no longer occur, and training should proceed normally.

## Background: HuggingFace Conventions

In the HuggingFace ecosystem:
- `-100` is the standard `ignore_index` for language modeling tasks
- This convention is used across `DataCollatorForLanguageModeling`, `Trainer`, and most loss calculations
- When integrating custom models with HuggingFace tools, it's important to follow this convention

## Alternative Solutions (Not Recommended)

If you wanted to keep using `-1` as the ignore_index, you would need to:
1. Create a custom data collator that uses `-1` instead of `-100`
2. Pass this custom collator to the trainer

However, this is not recommended because it goes against HuggingFace conventions and could cause issues with other components.

