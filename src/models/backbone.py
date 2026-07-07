"""
Backbone model factory. Adding a new architecture means writing one builder
function and registering it in MODEL_REGISTRY — nothing else in the pipeline
needs to change (FL client/server code, FUSED CLI/adapters, and evaluation
all operate on `nn.Module` + a uniform `get_named_layers()` contract).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


def _replace_classifier_head(model: nn.Module, in_features: int, num_classes: int,
                              attr_path: List[str]) -> nn.Module:
    """Generic helper to swap a model's final classifier Linear layer.
    attr_path is the dotted attribute chain to the layer to replace, e.g.
    ["fc"] for ResNet, ["classifier"] for DenseNet."""
    new_head = nn.Linear(in_features, num_classes)
    obj = model
    for attr in attr_path[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attr_path[-1], new_head)
    return model


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.resnet18(weights=weights)
    return _replace_classifier_head(model, model.fc.in_features, num_classes, ["fc"])


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tvm.resnet50(weights=weights)
    return _replace_classifier_head(model, model.fc.in_features, num_classes, ["fc"])


def build_densenet121(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.densenet121(weights=weights)
    return _replace_classifier_head(model, model.classifier.in_features, num_classes, ["classifier"])


def build_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    new_head = nn.Linear(in_features, num_classes)
    model.classifier[-1] = new_head
    return model


def build_vit_b16(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "resnet18": build_resnet18,
    "resnet50": build_resnet50,
    "densenet121": build_densenet121,
    "efficientnet_b0": build_efficientnet_b0,
    "vit_b16": build_vit_b16,
}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model backbone '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained)


def get_named_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return (name, module) pairs for every leaf-ish "layer" used by both
    FedBN (to find BatchNorm layers) and FUSED's Critical Layer
    Identification (to enumerate candidate layers for sparse adapters).

    We treat each *parameterized* module that owns its own weight tensor
    directly (Conv2d, Linear, BatchNorm*, LayerNorm) as one "layer" — this
    matches the granularity used in the FUSED paper's layer-wise Diff
    computation (Eq. 11-13), which operates over individual parameter
    tensors per named module rather than coarse blocks.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm1d,
                                nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            # Only count modules with learnable parameters of their own.
            if any(p.requires_grad or True for p in module.parameters(recurse=False)):
                if list(module.parameters(recurse=False)):
                    layers.append((name, module))
    return layers


def get_batchnorm_layer_names(model: nn.Module) -> List[str]:
    """All BatchNorm layer names — used by FedBN to decide which parameters
    stay local (never aggregated) vs. which get federated."""
    return [
        name for name, module in model.named_modules()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ]
