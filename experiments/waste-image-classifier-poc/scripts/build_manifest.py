#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from waste_poc.config import load_config
from waste_poc.manifests import assign_grouped_stratified_splits, collect_image_rows, find_trashnet_image_root, write_manifest_outputs
from waste_poc.utils import CLASS_NAMES, read_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the reproducible TrashNet manifest used by training and evaluation.")
    parser.add_argument("--config", default="configs/efficientnet_b0_baseline.yaml")
    parser.add_argument("--metadata", default="data/raw/trashnet_source_metadata.json")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    metadata = read_json(ROOT / args.metadata)
    image_root = ROOT / args.image_root if args.image_root else ROOT / metadata["source_directory_detected"]
    if not image_root.exists():
        image_root = find_trashnet_image_root(ROOT / "data" / "raw" / "trashnet-source")
    split_cfg = config.get("split", {"train": 0.70, "validation": 0.15, "test": 0.15})
    split = {key: split_cfg[key] for key in ["train", "validation", "test"]}
    seed = args.seed if args.seed is not None else split_cfg.get("seed", config.get("seed", 42))
    valid_rows, invalid_rows = collect_image_rows(image_root, metadata["resolved_commit_sha"], CLASS_NAMES)
    assigned = assign_grouped_stratified_splits(valid_rows, split, seed=seed)
    write_manifest_outputs(assigned, invalid_rows, metadata, ROOT / args.output_dir)
    print(f"Wrote manifest for {len(assigned)} valid images to {ROOT / args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
