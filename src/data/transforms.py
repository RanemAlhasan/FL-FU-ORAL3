"""Transform factory for the oral cancer dataset. Keeping this separate from
dataset.py makes it trivial to swap in stronger augmentation later, or
backbone-specific normalization (e.g. different stats for ViT vs ResNet),
without touching the dataset class."""
from __future__ import annotations

from typing import Dict

import torchvision.transforms as T

# ImageNet stats are reused across all current backbones (ResNet/DenseNet/
# EfficientNet/ViT all expect this normalization out of the box).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = True,
                      augmentation: str = "standard") -> T.Compose:
    """Build a torchvision transform pipeline.

    augmentation:
        "none"     -> resize + normalize only (useful for sanity-checking data)
        "standard" -> light augmentation, safe for medical images (no heavy
                      color jitter that could destroy diagnostic features)
        "strong"   -> adds color jitter + rotation, for ablations
    """
    if not train:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    if augmentation == "none":
        ops = [T.Resize((image_size, image_size))]
    elif augmentation == "standard":
        ops = [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
        ]
    elif augmentation == "strong":
        ops = [
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation}")

    ops += [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return T.Compose(ops)
