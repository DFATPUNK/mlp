from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

from .manifests import find_trashnet_image_root
from .utils import CLASS_NAMES, sha256_file, utc_now_iso

TRASHNET_ARCHIVE_RELATIVE_PATH = Path("data") / "dataset-resized.zip"


@dataclass(frozen=True)
class TrashNetImageRoot:
    image_root: Path
    archive_path: Path | None
    archive_sha256: str | None
    archive_extracted_this_run: bool


def find_dataset_archive(source_dir: str | Path) -> Path:
    source_dir = Path(source_dir)
    archive_path = source_dir / TRASHNET_ARCHIVE_RELATIVE_PATH
    if not archive_path.is_file():
        raise FileNotFoundError(
            "TrashNet image folders were not found, and the expected upstream archive is missing: "
            f"{archive_path}. Re-run scripts/download_trashnet.py --force or inspect the TrashNet clone."
        )
    return archive_path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validated_zip_targets(archive_path: Path, extraction_dir: Path) -> list[tuple[zipfile.ZipInfo, Path]]:
    extraction_root = extraction_dir.resolve()
    targets: list[tuple[zipfile.ZipInfo, Path]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename.replace("\\", "/"))
            if member_path.is_absolute() or ".." in member_path.parts or not member_path.parts:
                raise ValueError(f"Refusing to extract suspicious archive member {member.filename!r}: path traversal is not allowed")
            target = extraction_dir.joinpath(*member_path.parts).resolve()
            if not _is_relative_to(target, extraction_root):
                raise ValueError(f"Refusing to extract suspicious archive member {member.filename!r}: path escapes {extraction_dir}")
            targets.append((member, target))
    return targets


def safe_extract_zip(archive_path: str | Path, extraction_dir: str | Path) -> None:
    archive_path = Path(archive_path)
    extraction_dir = Path(extraction_dir)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    targets = _validated_zip_targets(archive_path, extraction_dir)
    with zipfile.ZipFile(archive_path) as archive:
        for member, target in targets:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def ensure_trashnet_image_root(source_dir: str | Path, expected_classes=CLASS_NAMES) -> TrashNetImageRoot:
    source_dir = Path(source_dir)
    archive_path = source_dir / TRASHNET_ARCHIVE_RELATIVE_PATH
    archive_sha256 = sha256_file(archive_path) if archive_path.is_file() else None
    try:
        image_root = find_trashnet_image_root(source_dir, expected_classes)
        return TrashNetImageRoot(image_root, archive_path if archive_path.is_file() else None, archive_sha256, False)
    except FileNotFoundError:
        pass

    archive_path = find_dataset_archive(source_dir)
    archive_sha256 = sha256_file(archive_path)
    safe_extract_zip(archive_path, archive_path.parent)
    try:
        image_root = find_trashnet_image_root(source_dir, expected_classes)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Extracted {archive_path}, but could not find a valid TrashNet image root afterward.") from exc
    return TrashNetImageRoot(image_root, archive_path, archive_sha256, True)


def _relative_path_or_none(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(root))


def trashnet_source_metadata(
    *,
    repository_url: str,
    requested_revision: str,
    resolved_commit_sha: str,
    root: str | Path,
    extraction: TrashNetImageRoot,
) -> dict:
    root = Path(root)
    return {
        "repository_url": repository_url,
        "requested_revision": requested_revision,
        "resolved_commit_sha": resolved_commit_sha,
        "download_timestamp": utc_now_iso(),
        "expected_class_names": CLASS_NAMES,
        "source_directory_detected": str(extraction.image_root.relative_to(root)),
        "archive_relative_path": _relative_path_or_none(extraction.archive_path, root),
        "archive_sha256": extraction.archive_sha256,
        "archive_extracted_this_run": extraction.archive_extracted_this_run,
    }
