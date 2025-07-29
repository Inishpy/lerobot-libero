import torch
import torch.nn as nn
import logging

import os
import json
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file
from lerobot.utils.save_lora_adapter import save_lora_adapter




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


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        # Only LoRA params are trainable
        for p in self.linear.parameters():
            p.requires_grad = False
        for p in self.lora_A.parameters():
            p.requires_grad = True
        for p in self.lora_B.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.linear(x) + self.dropout(self.lora_B(self.lora_A(x))) * (self.alpha / self.lora_A.out_features)

def replace_linear_with_lora(module, r=8, alpha=16, dropout=0.05, target_modules=None, prefix=''):
    """
    Recursively replace nn.Linear layers with LoRALinear in the given module.
    If target_modules is provided, only replace layers whose names match.
    """
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and (target_modules is None or name in target_modules or full_name in target_modules):
            lora_layer = LoRALinear(child.in_features, child.out_features, r, alpha, dropout)
            # Copy original weights for inference parity
            lora_layer.linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lora_layer.linear.bias.data.copy_(child.bias.data)
            setattr(module, name, lora_layer)
            logging.info(f"Replaced Linear layer '{full_name}' with LoRALinear (r={r}, alpha={alpha}, dropout={dropout})")
        else:
            replace_linear_with_lora(child, r, alpha, dropout, target_modules, prefix=full_name)
            
            
            
"""
push_lora_adapter.py

Utility to save and push only LoRA adapter weights to the Hugging Face Hub,
with a reference to the base model, PEFT-style.

Usage:
    from lerobot.utils.push_lora_adapter import push_lora_adapter_to_hub

    push_lora_adapter_to_hub(
        model,
        repo_id="username/my-lora-adapter",
        base_model="lerobot/smolvla_base",
        adapter_filename="adapter_model.bin",
        config_filename="adapter_config.json",
        private=False,
        token=None,
    )
"""



def push_lora_adapter_to_hub(
    model,
    repo_id,
    base_model,
    adapter_filename="adapter_model.bin",
    config_filename="adapter_config.json",
    private=False,
    token=None,
):
    # Save adapter weights locally
    save_lora_adapter(model, adapter_filename)
    # Write minimal config referencing base model
    config = {"base_model": base_model}
    with open(config_filename, "w") as f:
        json.dump(config, f, indent=2)
    # Create repo if needed
    api = HfApi()
    if token is None:
        token = HfFolder.get_token()
    create_repo(repo_id, private=private, exist_ok=True, token=token)
    # Upload files
    upload_file(
        path_or_fileobj=adapter_filename,
        path_in_repo=adapter_filename,
        repo_id=repo_id,
        token=token,
    )
    upload_file(
        path_or_fileobj=config_filename,
        path_in_repo=config_filename,
        repo_id=repo_id,
        token=token,
    )
    print(f"Pushed LoRA adapter and config to https://huggingface.co/{repo_id}")

# Example CLI usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Push LoRA adapter to Hugging Face Hub.")
    parser.add_argument("--repo_id", required=True, help="Hub repo id (e.g., username/my-lora-adapter)")
    parser.add_argument("--base_model", required=True, help="Base model repo id (e.g., lerobot/smolvla_base)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--token", default=None, help="Hugging Face token")
    args = parser.parse_args()
    print("This script is intended to be used as a library. See docstring for usage.")