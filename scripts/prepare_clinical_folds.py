"""Create train/validation/test fold directories from a split manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def txt_key(class_name: str, file_name: str) -> str:
    sample_id = file_name[len(class_name) + 1 : -4]
    return f"{class_name}/{sample_id}"


def build_manifest_key_set(config: dict) -> set[str]:
    keys = set()
    for fold_map in config["folds"].values():
        for items in fold_map.values():
            keys.update(items)
    return keys


def build_filesystem_key_set(root_dir: Path) -> set[str]:
    keys = set()
    for class_dir in root_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for file_path in class_dir.glob(f"{class_name}_*.txt"):
            keys.add(txt_key(class_name, file_path.name))
    return keys


def normalize_public_label(class_name: str) -> str:
    return "UNHCC" if class_name in {"ICC", "Transfer"} else class_name


def copy_split_files(data_root: Path, output_root: Path, config: dict, split_names: list[str]) -> list[str]:
    missing_files = []
    for fold_name, fold_map in config["folds"].items():
        for split_name in split_names:
            for item in fold_map.get(split_name, []):
                class_name, sample_id = item.split("/", 1)
                source_path = data_root / class_name / f"{class_name}_{sample_id}.txt"
                target_dir = output_root / fold_name / split_name / normalize_public_label(class_name)
                target_dir.mkdir(parents=True, exist_ok=True)
                if source_path.exists():
                    shutil.copy2(source_path, target_dir / source_path.name)
                else:
                    missing_files.append(str(source_path))
    return missing_files


def copy_unreferenced_files(data_root: Path, output_root: Path, keys: list[str], directory_name: str) -> None:
    for item in keys:
        class_name, sample_id = item.split("/", 1)
        source_path = data_root / class_name / f"{class_name}_{sample_id}.txt"
        if not source_path.exists():
            continue
        target_dir = output_root / directory_name / normalize_public_label(class_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_dir / source_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fold directories from a JSON split manifest.")
    parser.add_argument("--data-root", required=True, help="Directory containing class subdirectories of text files.")
    parser.add_argument("--split-file", required=True, help="JSON manifest with fold assignments.")
    parser.add_argument("--output", default="data/clinical_cv", help="Output directory for generated folds.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split names to copy from each fold.",
    )
    parser.add_argument(
        "--unreferenced-dir",
        default="test_unreferenced",
        help="Directory name used for files missing from the manifest.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_file = Path(args.split_file)
    output_root = Path(args.output)
    if not data_root.exists():
        raise FileNotFoundError(f"Source directory not found: {data_root}")
    if not split_file.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_file}")

    with open(split_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    missing_files = copy_split_files(data_root, output_root, config, args.splits)
    manifest_keys = build_manifest_key_set(config)
    filesystem_keys = build_filesystem_key_set(data_root)
    unreferenced_files = sorted(filesystem_keys - manifest_keys)
    if unreferenced_files:
        copy_unreferenced_files(data_root, output_root, unreferenced_files, args.unreferenced_dir)

    print(f"Copied fold data to {output_root}")
    if missing_files:
        print(f"Missing files declared in the manifest: {len(missing_files)}")
        for item in missing_files:
            print(item)
    if unreferenced_files:
        print(f"Unreferenced source files copied to {output_root / args.unreferenced_dir}: {len(unreferenced_files)}")


if __name__ == "__main__":
    main()
