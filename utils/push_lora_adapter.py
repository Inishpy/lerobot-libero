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

import os
import json
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file
from lerobot.utils.save_lora_adapter import save_lora_adapter

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