"""
OralCancerDataset: loads images from the hospital/domain-structured directory
layout described in the project brief:

    dataset/oral_cancer/{Train,Test}/{Hospital}/{ClassDir}/*.jpg
    (+ optional matching *.json metadata file per image)

Each sample carries: image tensor, class label, hospital/domain label,
subtype, and image path. JSON metadata support is kept for compatibility
when load_metadata=True, but it is not required for training/evaluation.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Callable, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

CLASS_DIRS = ["0_Benign", "1_Potentially_Malignant", "2_Malignant"]
CLASS_NAMES = ["Benign", "Potentially_Malignant", "Malignant"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_DIRS)}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class Sample:
    __slots__ = ("image_path", "json_path", "label", "label_name", "hospital", "subtype")

    def __init__(
        self,
        image_path: str,
        json_path: Optional[str],
        label: int,
        label_name: str,
        hospital: str,
        subtype: Optional[str] = None,
    ):
        self.image_path = image_path
        self.json_path = json_path
        self.label = label
        self.label_name = label_name
        self.hospital = hospital
        self.subtype = subtype


def _find_json_for_image(image_path: str) -> Optional[str]:
    base, _ = os.path.splitext(image_path)
    candidate = base + ".json"
    return candidate if os.path.exists(candidate) else None


def index_dataset(root: str, split: str, hospitals: List[str]) -> List[Sample]:
    samples: List[Sample] = []
    split_dir = os.path.join(root, split)

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    for hospital in hospitals:
        hospital_dir = os.path.join(split_dir, hospital)

        if not os.path.isdir(hospital_dir):
            raise FileNotFoundError(
                f"Hospital directory not found: {hospital_dir}. "
                f"Check 'hospitals' in your config against the dataset folder names."
            )

        for class_dir in CLASS_DIRS:
            class_path = os.path.join(hospital_dir, class_dir)

            if not os.path.isdir(class_path):
                continue

            label = CLASS_TO_IDX[class_dir]
            label_name = CLASS_NAMES[label]

            for dirpath, _dirnames, filenames in os.walk(class_path):
                for fname in sorted(filenames):
                    if not fname.lower().endswith(IMAGE_EXTENSIONS):
                        continue

                    image_path = os.path.join(dirpath, fname)
                    json_path = _find_json_for_image(image_path)

                    rel_path = os.path.relpath(dirpath, class_path)
                    subtype = None if rel_path == "." else rel_path.split(os.sep)[0]

                    samples.append(
                        Sample(
                            image_path=image_path,
                            json_path=json_path,
                            label=label,
                            label_name=label_name,
                            hospital=hospital,
                            subtype=subtype,
                        )
                    )

    return samples


class OralCancerDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        transform: Optional[Callable] = None,
        load_metadata: bool = False,
        hospital_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.samples = samples
        self.transform = transform
        self.load_metadata = load_metadata

        if hospital_to_idx is None:
            unique_hospitals = sorted({s.hospital for s in samples})
            hospital_to_idx = {h: i for i, h in enumerate(unique_hospitals)}

        self.hospital_to_idx = hospital_to_idx

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        image = Image.open(sample.image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        item = {
            "image": image,
            "label": sample.label,
            "label_name": sample.label_name,
            "hospital": sample.hospital,
            "hospital_idx": self.hospital_to_idx[sample.hospital],
            "subtype": sample.subtype or "",
            "image_path": sample.image_path,
        }

        if self.load_metadata:
            item["metadata"] = self._load_json(sample.json_path)

        return item

    @staticmethod
    def _load_json(json_path: Optional[str]) -> Optional[Dict]:
        if json_path is None:
            return None

        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def class_distribution(self) -> Counter:
        return Counter(s.label_name for s in self.samples)

    def hospital_distribution(self) -> Counter:
        return Counter(s.hospital for s in self.samples)

    def subtype_distribution(self) -> Counter:
        return Counter(s.subtype or "" for s in self.samples)

    def class_counts_per_hospital(self) -> Dict[str, Counter]:
        out: Dict[str, Counter] = {}
        for s in self.samples:
            out.setdefault(s.hospital, Counter())[s.label_name] += 1
        return out

    def subset_by_hospitals(self, hospitals: List[str]) -> "OralCancerDataset":
        keep = set(hospitals)
        filtered = [s for s in self.samples if s.hospital in keep]
        return OralCancerDataset(
            filtered,
            transform=self.transform,
            load_metadata=self.load_metadata,
            hospital_to_idx=self.hospital_to_idx,
        )

    def subset_by_classes(self, class_indices: List[int]) -> "OralCancerDataset":
        keep = set(class_indices)
        filtered = [s for s in self.samples if s.label in keep]
        return OralCancerDataset(
            filtered,
            transform=self.transform,
            load_metadata=self.load_metadata,
            hospital_to_idx=self.hospital_to_idx,
        )

    def class_sample_weights(self) -> List[float]:
        counts = self.class_distribution()
        total = sum(counts.values())

        class_weight = {
            label_name: total / (len(counts) * count)
            for label_name, count in counts.items()
        }

        return [class_weight[s.label_name] for s in self.samples]