#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from waste_poc.manifests import find_trashnet_image_root
from waste_poc.utils import CLASS_NAMES, utc_now_iso, write_json

REPOSITORY_URL = "https://github.com/garythung/trashnet.git"


def run_git(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone the original TrashNet repository and record reproducible source metadata.")
    parser.add_argument("--revision", default="HEAD", help="Git revision to check out after cloning. Defaults to HEAD.")
    parser.add_argument("--force", action="store_true", help="Delete and reclone data/raw/trashnet-source if it already exists.")
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw"
    source_dir = raw_dir / "trashnet-source"
    if source_dir.exists() and args.force:
        shutil.rmtree(source_dir)
    if not source_dir.exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning original TrashNet repository: {REPOSITORY_URL}")
        run_git(["clone", REPOSITORY_URL, str(source_dir)])
    else:
        print(f"Using existing clone at {source_dir}")
    run_git(["fetch", "--all", "--tags"], cwd=source_dir)
    run_git(["checkout", args.revision], cwd=source_dir)
    commit = run_git(["rev-parse", "HEAD"], cwd=source_dir)
    image_root = find_trashnet_image_root(source_dir)
    metadata = {
        "repository_url": REPOSITORY_URL,
        "requested_revision": args.revision,
        "resolved_commit_sha": commit,
        "download_timestamp": utc_now_iso(),
        "expected_class_names": CLASS_NAMES,
        "source_directory_detected": str(image_root.relative_to(ROOT)),
    }
    write_json(raw_dir / "trashnet_source_metadata.json", metadata)
    print(f"Resolved TrashNet commit: {commit}")
    print(f"Detected image root: {image_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
