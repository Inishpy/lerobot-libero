import torch
import torch.nn as nn
import logging

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