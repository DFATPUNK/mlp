from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from waste_poc.trashnet_download import ensure_trashnet_image_root, safe_extract_zip, trashnet_source_metadata
from waste_poc.utils import CLASS_NAMES, sha256_file


def make_archive(source_dir: Path, members: dict[str, bytes] | None = None) -> Path:
    archive_path = source_dir / "data" / "dataset-resized.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    members = members or {f"dataset-resized/{label}/{label}-example.jpg": b"fake image" for label in CLASS_NAMES}
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return archive_path


def make_extracted_image_root(source_dir: Path) -> Path:
    image_root = source_dir / "data" / "dataset-resized"
    for label in CLASS_NAMES:
        class_dir = image_root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / f"{label}-example.jpg").write_bytes(b"fake image")
    return image_root


class TrashNetDownloadTests(unittest.TestCase):
    def test_valid_archive_extracts_and_returns_image_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "trashnet-source"
            archive_path = make_archive(source_dir)

            result = ensure_trashnet_image_root(source_dir)

            self.assertEqual(result.image_root, source_dir / "data" / "dataset-resized")
            self.assertTrue(result.archive_extracted_this_run)
            self.assertEqual(result.archive_path, archive_path)
            self.assertEqual(result.archive_sha256, sha256_file(archive_path))

    def test_existing_image_root_skips_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "trashnet-source"
            image_root = make_extracted_image_root(source_dir)
            make_archive(source_dir)

            with mock.patch("waste_poc.trashnet_download.safe_extract_zip") as extract_zip:
                result = ensure_trashnet_image_root(source_dir)

            self.assertEqual(result.image_root, image_root)
            self.assertFalse(result.archive_extracted_this_run)
            extract_zip.assert_not_called()

    def test_archive_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "trashnet-source"
            archive_path = make_archive(source_dir, {"../escape.txt": b"nope"})

            with self.assertRaisesRegex(ValueError, "path traversal"):
                safe_extract_zip(archive_path, archive_path.parent)

            self.assertFalse((source_dir / "escape.txt").exists())
            self.assertFalse((root / "escape.txt").exists())

    def test_missing_archive_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "trashnet-source"
            source_dir.mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "data/dataset-resized.zip"):
                ensure_trashnet_image_root(source_dir)

    def test_extraction_is_idempotent_after_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "trashnet-source"
            make_archive(source_dir)

            first = ensure_trashnet_image_root(source_dir)
            second = ensure_trashnet_image_root(source_dir)

            self.assertTrue(first.archive_extracted_this_run)
            self.assertFalse(second.archive_extracted_this_run)
            self.assertEqual(first.image_root, second.image_root)

    def test_archive_sha256_is_recorded_in_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "data" / "raw" / "trashnet-source"
            archive_path = make_archive(source_dir)
            extraction = ensure_trashnet_image_root(source_dir)

            metadata = trashnet_source_metadata(
                repository_url="https://example.test/trashnet.git",
                requested_revision="HEAD",
                resolved_commit_sha="abc123",
                root=root,
                extraction=extraction,
            )

            self.assertEqual(metadata["archive_relative_path"], str(archive_path.relative_to(root)))
            self.assertEqual(metadata["archive_sha256"], sha256_file(archive_path))
            self.assertTrue(metadata["archive_extracted_this_run"])


if __name__ == "__main__":
    unittest.main()
