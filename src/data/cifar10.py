"""
CIFAR-10 dataset loading, transforms matched exactly to FUSED-Code's
dataset/data_utils.py::data_set('cifar10') branch:

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

NOTE: the original code applies this SAME (training) transform — including
random crop and horizontal flip — to both trainset AND testset. This is
almost certainly an oversight in the original implementation (test sets are
conventionally evaluated without augmentation), but since this module's
purpose is a faithful reproduction of FUSED-Code's reported numbers rather
than "best practice," we replicate it exactly, including this quirk. If you
want a corrected version for any other use of this codebase, see
`build_eval_transform()` below, provided as an alternative but NOT used by
the faithful-reproduction path.
"""
from __future__ import annotations

import os
from typing import Tuple

import torchvision.transforms as T
from torchvision.datasets import CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def build_fused_transform() -> T.Compose:
    """Exact replica of FUSED-Code's transform, applied to BOTH train and
    test sets in the original (see module docstring)."""
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def build_eval_transform() -> T.Compose:
    """A corrected, augmentation-free transform for evaluation, provided as
    an opt-in alternative. NOT used by run_fused_cifar10.py's default
    faithful-reproduction path — only available if you explicitly want to
    deviate from the original code's behavior for a cleaner eval signal."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def load_cifar10(root: str = "dataset/cifar10", use_eval_transform_for_test: bool = False) -> Tuple[CIFAR10, CIFAR10]:
    """Load CIFAR-10 train/test sets. `root` should point at the directory
    containing (or where torchvision should place) the `cifar-10-batches-py`
    folder — matching the layout already present at dataset/cifar10 per the
    user's setup.

    `use_eval_transform_for_test`: if True, deviates from the original
    FUSED-Code (which uses the augmented transform for test too) and uses a
    clean eval transform for the test set instead. Default False to match
    the original exactly.

    Download is skipped automatically if `cifar-10-batches-py` already
    exists under `root` (e.g. the dataset was already placed there) — this
    avoids any network access at all when the data is already present,
    which matters both for offline/restricted environments and for not
    re-fetching ~170MB unnecessarily on every run.
    """
    os.makedirs(root, exist_ok=True)
    already_present = os.path.isdir(os.path.join(root, "cifar-10-batches-py"))

    train_transform = build_fused_transform()
    test_transform = build_eval_transform() if use_eval_transform_for_test else build_fused_transform()

    trainset = CIFAR10(root=root, train=True, download=not already_present, transform=train_transform)
    testset = CIFAR10(root=root, train=False, download=not already_present, transform=test_transform)
    return trainset, testset
