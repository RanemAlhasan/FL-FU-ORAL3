"""
Faithful port of:
  - models/CNN_Cifar10.py::Model (despite the filename, this is a
    torchvision resnet18(pretrained=True) with a replaced final FC layer —
    the file's actual custom-CNN and ViT code is commented out / unused)
  - models/Model_base.py::Lora (the actual "sparse unlearning adapter" —
    in the real implementation, this is literally a PEFT LoraConfig, not
    a custom random-sparsity mask as the paper's Eq. 23 describes)

Target layers for CIFAR-10's LoRA adapter, copied verbatim from
Model_base.py::Lora.__init__'s `args.data_name == 'cifar10'` branch:
    target_modules = ["layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"]
This is K=4, matching the paper's prose ("last, second-to-last, sixth-to-
last, eighth-to-last layers"), but note it is a HARDCODED literal in the
source — there is no actual Critical Layer Identification (Diff-based)
computation anywhere in FUSED-Code that selects these layers dynamically.
We replicate the hardcoding faithfully rather than "fixing" it with a real
CLI computation, since the goal here is matching the paper's reported
numbers, not implementing Eq. 11-13 as literally described.

LoRA hyperparameters, copied verbatim from Model_base.py::Lora.__init__:
    LoraConfig(r=16, lora_alpha=32, target_modules=target_modules,
               lora_dropout=0.1, bias="none")
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torchvision
from peft import LoraConfig, PeftModel, get_peft_model

CIFAR10_TARGET_MODULES: List[str] = ["layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2", "fc"]
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1


def build_resnet18_cifar10(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Faithful port of CNN_Cifar10.Model.__init__:
        self.model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

    Returns the plain torchvision ResNet (not wrapped in the original's
    thin `Model(MyModel)` class, since that wrapper adds no behavior beyond
    forward() delegation — using the bare ResNet directly is equivalent and
    simpler for our pipeline)."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def build_lora_adapter(base_model: nn.Module) -> PeftModel:
    """Faithful port of Model_base.py::Lora for data_name == 'cifar10':
        config = LoraConfig(r=16, lora_alpha=32, target_modules=target_modules,
                             lora_dropout=0.1, bias="none")
        self.lora_model = get_peft_model(global_model, config)
        for name, param in self.lora_model.named_parameters():
            if not any(target in name for target in config.target_modules):
                param.requires_grad = False

    Note: get_peft_model() already freezes all non-LoRA parameters by
    default — the explicit re-freezing loop in the original is redundant
    with PEFT's own behavior, but we still verify the resulting
    requires_grad pattern matches (see tests), since it costs nothing and
    protects against any PEFT version difference in default freezing
    behavior.
    """
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=CIFAR10_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    lora_model = get_peft_model(base_model, config)
    for name, param in lora_model.named_parameters():
        if not any(target in name for target in config.target_modules):
            param.requires_grad = False
    return lora_model


def count_trainable_parameters(model: nn.Module) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
