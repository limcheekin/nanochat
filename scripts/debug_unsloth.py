# Save this file as: debug_unsloth.py

import torch
print(f"PyTorch version: {torch.__version__}")

try:
    import unsloth
    print(f"Unsloth version: {unsloth.__version__}")
    from unsloth import FastLanguageModel

    print("\n--- Unsloth Import Details ---")
    print(f"Successfully imported 'FastLanguageModel'.")
    print(f"type(FastLanguageModel) is: {type(FastLanguageModel)}")
    print("\n--- Attributes of FastLanguageModel ---")
    print(dir(FastLanguageModel))
    print("\n------------------------------------")

    # Let's try to instantiate it with a dummy model to replicate the error
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
        def forward(self, x):
            return self.linear(x)

    print("\nAttempting to wrap a dummy torch.nn.Module...")
    try:
        dummy = DummyModel()
        # This is the line that fails in the main script
        wrapped_model = FastLanguageModel(dummy)
        print("✅ Successfully wrapped a dummy model.")
        print(f"   type(wrapped_model) is: {type(wrapped_model)}")
    except Exception as e:
        print(f"\n❌ ERROR when wrapping dummy model:")
        print(f"   Error Type: {type(e)}")
        print(f"   Error Message: {e}")

except ImportError:
    print("\n❌ ERROR: Could not import 'unsloth'. Please check the installation.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
