from __future__ import annotations

import base64
import csv
import hashlib
import time
import zipfile
from pathlib import Path

NAME = "waste_image_classifier_poc"
VERSION = "0.6.0"
DIST_INFO = f"{NAME}-{VERSION}.dist-info"


def _metadata() -> str:
    return f"Metadata-Version: 2.1\nName: waste-image-classifier-poc\nVersion: {VERSION}\nSummary: Isolated TrashNet waste image classifier proof of concept\nRequires-Python: >=3.10\n"


def _wheel() -> str:
    return "Wheel-Version: 1.0\nGenerator: waste-poc-build-backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n"


def _hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"sha256={digest}"


def _write_metadata(metadata_directory: str | None = None) -> str:
    if metadata_directory is None:
        return DIST_INFO
    dist = Path(metadata_directory) / DIST_INFO
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(_metadata(), encoding="utf-8")
    (dist / "WHEEL").write_text(_wheel(), encoding="utf-8")
    (dist / "RECORD").write_text("", encoding="utf-8")
    return DIST_INFO


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _write_metadata(metadata_directory)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return _write_metadata(metadata_directory)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _build(wheel_directory, editable=False)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _build(wheel_directory, editable=True)


def _build(wheel_directory, editable: bool):
    wheel_directory = Path(wheel_directory)
    wheel_directory.mkdir(parents=True, exist_ok=True)
    wheel_name = f"{NAME}-{VERSION}-py3-none-any.whl"
    wheel_path = wheel_directory / wheel_name
    project_root = Path(__file__).resolve().parent
    records = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        def write_file(arcname: str, data: bytes):
            zf.writestr(arcname, data)
            records.append((arcname, _hash(data), str(len(data))))

        if editable:
            pth = f"{project_root / 'src'}\n".encode("utf-8")
            write_file(f"{NAME}.pth", pth)
        else:
            for path in (project_root / "src").rglob("*.py"):
                write_file(path.relative_to(project_root / "src").as_posix(), path.read_bytes())
        write_file(f"{DIST_INFO}/METADATA", _metadata().encode("utf-8"))
        write_file(f"{DIST_INFO}/WHEEL", _wheel().encode("utf-8"))
        record_rows = [*records, (f"{DIST_INFO}/RECORD", "", "")]
        record_text = "".join([",".join(row) + "\n" for row in record_rows]).encode("utf-8")
        zf.writestr(f"{DIST_INFO}/RECORD", record_text)
    return wheel_name
