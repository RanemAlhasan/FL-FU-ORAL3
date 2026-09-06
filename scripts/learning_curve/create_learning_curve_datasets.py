#!/usr/bin/env python3
"""
Create deterministic, nested learning-curve views of the existing oral dataset.

The source dataset is NEVER modified or copied. Selected training images are
represented by symbolic links, and every learning-curve view points to the same
original Test directory.

Default output:
    dataset/oral3_learning_curve_seed42/
        p20/
        p40/
        p60/
        p80/
        p100/
        selection_manifest.csv
        subset_summary.csv
        study_metadata.json

Properties:
- stratified independently inside every (hospital, class) group
- deterministic
- nested: p20 ⊂ p40 ⊂ p60 ⊂ p80 ⊂ p100
- Test is identical for every percentage
- p100/Train points directly to the original Train directory
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable, List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

DEFAULT_HOSPITALS = [
    "Spain_Dataset",
    "Canada_Dataset",
    "India_Dataset",
]

DEFAULT_CLASSES = [
    "0_Benign",
    "1_Potentially_Malignant",
    "2_Malignant",
]

DEFAULT_PERCENTAGES = [20, 40, 60, 80, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="dataset/oral3",
        help="Original dataset root containing Train/ and Test/.",
    )
    parser.add_argument(
        "--output-root",
        default="dataset/oral3_learning_curve_seed42",
        help="Where the p20/p40/... views will be created.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--percentages",
        type=int,
        nargs="+",
        default=DEFAULT_PERCENTAGES,
    )
    parser.add_argument(
        "--hospitals",
        nargs="+",
        default=DEFAULT_HOSPITALS,
    )
    parser.add_argument(
        "--classes",
        dest="classes_",
        nargs="+",
        default=DEFAULT_CLASSES,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing pXX views if they already exist.",
    )
    return parser.parse_args()


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def image_files(root: Path) -> List[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def deterministic_key(path: Path, source: Path, seed: int) -> str:
    rel = path.relative_to(source).as_posix()
    token = f"{seed}|{rel}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()


def selected_count(n: int, pct: int) -> int:
    if n == 0:
        return 0
    if pct >= 100:
        return n
    # Conventional half-up rounding, with at least one image from a nonempty
    # hospital/class stratum.
    return min(n, max(1, int(math.floor(n * pct / 100.0 + 0.5))))


def make_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def validate_source(
    source: Path,
    hospitals: Iterable[str],
    classes_: Iterable[str],
) -> None:
    for split in ("Train", "Test"):
        split_dir = source / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing required split: {split_dir}")

    missing = []
    for hospital in hospitals:
        for class_name in classes_:
            path = source / "Train" / hospital / class_name
            if not path.is_dir():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing expected Train hospital/class directories:\n  "
            + "\n  ".join(missing)
        )


def main() -> None:
    args = parse_args()

    source = Path(args.source).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    percentages = sorted(set(args.percentages))

    invalid = [p for p in percentages if p <= 0 or p > 100]
    if invalid:
        raise ValueError(f"Percentages must be in 1..100; got {invalid}")

    validate_source(source, args.hospitals, args.classes_)
    output_root.mkdir(parents=True, exist_ok=True)

    # Build ONE deterministic ordering per hospital/class. Each percentage
    # takes a longer prefix of that same ordering, which guarantees nesting.
    groups = {}
    for hospital in args.hospitals:
        for class_name in args.classes_:
            class_dir = source / "Train" / hospital / class_name
            paths = image_files(class_dir)
            paths.sort(key=lambda p: deterministic_key(p, source, args.seed))
            groups[(hospital, class_name)] = paths

    manifest_rows = []
    summary_rows = []

    total_available = sum(len(paths) for paths in groups.values())

    for pct in percentages:
        view_root = output_root / f"p{pct}"

        if view_root.exists() or view_root.is_symlink():
            if not args.force:
                raise FileExistsError(
                    f"{view_root} already exists. Re-run with --force only "
                    "if you intentionally want to rebuild the views."
                )
            remove_path(view_root)

        view_root.mkdir(parents=True)

        # Every percentage evaluates on EXACTLY the same original Test split.
        make_symlink(source / "Test", view_root / "Test")

        if pct == 100:
            # Keep the full-data view maximally faithful to the original.
            make_symlink(source / "Train", view_root / "Train")
        else:
            (view_root / "Train").mkdir()

        pct_total = 0

        for hospital in args.hospitals:
            for class_name in args.classes_:
                ordered = groups[(hospital, class_name)]
                k = selected_count(len(ordered), pct)
                chosen = ordered[:k]
                pct_total += len(chosen)

                summary_rows.append(
                    {
                        "fraction_pct": pct,
                        "hospital": hospital,
                        "class_name": class_name,
                        "available_count": len(ordered),
                        "selected_count": len(chosen),
                        "actual_group_fraction": (
                            len(chosen) / len(ordered) if ordered else 0.0
                        ),
                    }
                )

                for src_img in chosen:
                    rel = src_img.relative_to(source / "Train")
                    manifest_rows.append(
                        {
                            "fraction_pct": pct,
                            "hospital": hospital,
                            "class_name": class_name,
                            "relative_train_path": rel.as_posix(),
                            "source_image": str(src_img),
                        }
                    )

                    if pct < 100:
                        dst_img = view_root / "Train" / rel
                        make_symlink(src_img, dst_img)

                        # Preserve optional same-basename JSON metadata.
                        json_src = src_img.with_suffix(".json")
                        if json_src.exists():
                            make_symlink(
                                json_src,
                                dst_img.with_suffix(".json"),
                            )

        summary_rows.append(
            {
                "fraction_pct": pct,
                "hospital": "__TOTAL__",
                "class_name": "__TOTAL__",
                "available_count": total_available,
                "selected_count": pct_total,
                "actual_group_fraction": (
                    pct_total / total_available if total_available else 0.0
                ),
            }
        )

        # Simple structural checks.
        if not (view_root / "Train").exists():
            raise RuntimeError(f"Train view was not created: {view_root}")
        if not (view_root / "Test").exists():
            raise RuntimeError(f"Test view was not created: {view_root}")

        print(
            f"p{pct}: {pct_total}/{total_available} training images "
            f"({100.0 * pct_total / max(1, total_available):.2f}%)"
        )

    manifest_path = output_root / "selection_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fraction_pct",
                "hospital",
                "class_name",
                "relative_train_path",
                "source_image",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = output_root / "subset_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fraction_pct",
                "hospital",
                "class_name",
                "available_count",
                "selected_count",
                "actual_group_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "source_dataset": str(source),
        "output_root": str(output_root),
        "seed": args.seed,
        "percentages": percentages,
        "hospitals": args.hospitals,
        "classes": args.classes_,
        "total_training_images": total_available,
        "test_policy": "All pXX/Test paths symlink to the same original Test directory.",
        "sampling_policy": (
            "Deterministic nested sampling stratified by hospital and class; "
            "SHA-256 ordering of seed|relative_path."
        ),
    }
    with (output_root / "study_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone.")
    print(f"Views:     {output_root}")
    print(f"Summary:   {summary_path}")
    print(f"Manifest:  {manifest_path}")
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
