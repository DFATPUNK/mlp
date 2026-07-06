from __future__ import annotations

from pathlib import Path

PREPROCESSING_VERSION = "exif_transpose_rgb_v1"


def load_rgb_image(path: str | Path):
    """Open an image, apply EXIF orientation, and return an RGB PIL image.

    The raw source file is never modified or re-saved.
    """
    path = Path(path)
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError

        with Image.open(path) as image:
            transposed = ImageOps.exif_transpose(image)
            return transposed.convert("RGB")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image file not found: {path}") from exc
    except Exception as exc:  # Pillow raises several decoder-specific exception types.
        try:
            from PIL import UnidentifiedImageError  # noqa: F401
        except Exception:
            pass
        raise ValueError(f"Could not decode image at {path}: {exc}") from exc
