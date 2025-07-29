"""
save_lora_adapter.py

Utility to save and load only LoRA adapter weights from a model using custom LoRALinear,
in the style of Hugging Face PEFT.

Usage:
    from lerobot.utils.save_lora_adapter import save_lora_adapter, load_lora_adapter

    # Save only LoRA adapter weights
    save_lora_adapter(model, "adapter_model.bin")

    # Load LoRA adapter weights into a model
    load_lora_adapter(model, "adapter_model.bin")
"""

import torch
import os

def is_lora_linear(module):
    # Import here to avoid circular import
    from lerobot.src.lerobot.scripts.lora import LoRALinear
    return isinstance(module, LoRALinear)

def get_lora_state_dict(model):
    """
    Recursively extract only LoRA adapter parameters from the model.
    Returns a flat state_dict with keys as in model.state_dict().
    """
    lora_state = {}
    for name, module in model.named_modules():
        if is_lora_linear(module):
            # Only save LoRA params (not the frozen base linear)
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    lora_state[f"{name}.{param_name}"] = param.detach().cpu()
            # Save alpha and dropout as buffers if needed
            if hasattr(module, "alpha"):
                lora_state[f"{name}.alpha"] = torch.tensor(module.alpha)
            if hasattr(module, "dropout"):
                # Save dropout p as a tensor for completeness
                lora_state[f"{name}.dropout_p"] = torch.tensor(module.dropout.p)
    return lora_state

def save_lora_adapter(model, path):
    """
    Save only the LoRA adapter weights to disk.
    """
    lora_state = get_lora_state_dict(model)
    torch.save(lora_state, path)
    print(f"Saved LoRA adapter weights to {path} ({len(lora_state)} tensors)")

def load_lora_adapter(model, path, strict=True):
    """
    Load LoRA adapter weights into a model.
    """
    lora_state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Missing keys: {missing}, Unexpected keys: {unexpected}")
    print(f"Loaded LoRA adapter weights from {path}")

# Example CLI usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Save or load LoRA adapter weights.")
    parser.add_argument("action", choices=["save", "load"], help="Action to perform.")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (for loading full model).")
    parser.add_argument("--adapter", required=True, help="Path to adapter file (to save/load).")
    args = parser.parse_args()

    # Example: user must provide code to load their model
    # Here we just show the API
    print("This script is intended to be used as a library, not as a standalone CLI.")