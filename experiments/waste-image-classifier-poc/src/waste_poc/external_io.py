from __future__ import annotations

from pathlib import Path

from .utils import utc_now_iso


def prepare_external_output_dir(path: str | None, checkpoint: str, overwrite: bool) -> Path:
    if path:
        output_dir = Path(path)
        if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"Output directory already contains files: {output_dir}. Use --overwrite or choose a new directory.")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    run_dir = Path(checkpoint).resolve().parent
    output_dir = run_dir / f"external_evaluation_{utc_now_iso().replace(':', '').replace('+', 'Z')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir
