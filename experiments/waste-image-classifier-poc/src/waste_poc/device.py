from __future__ import annotations

import os
import platform
from typing import Any

_PRINTED_DEVICE = False


def _mps_usable(torch_module: Any) -> bool:
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is None or not mps.is_built() or not mps.is_available():
        return False
    try:
        device = torch_module.device("mps")
        tensor = torch_module.ones(1, device=device)
        _ = (tensor + 1).cpu().item()
        return True
    except Exception:
        return False


def resolve_device(requested: str | None = "auto", *, torch_module: Any | None = None, print_selection: bool = True):
    global _PRINTED_DEVICE
    if torch_module is None:
        import torch

        torch_module = torch
    requested = (requested or "auto").lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("--device must be one of: auto, cuda, mps, cpu")

    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available. Use --device auto or --device cpu.")
        selected = "cuda"
    elif requested == "mps":
        if not _mps_usable(torch_module):
            raise RuntimeError("Requested --device mps, but MPS is not available or failed a test tensor operation. Use --device cpu.")
        selected = "mps"
    elif requested == "cpu":
        selected = "cpu"
    elif torch_module.cuda.is_available():
        selected = "cuda"
    elif _mps_usable(torch_module):
        selected = "mps"
    else:
        selected = "cpu"

    if print_selection and not _PRINTED_DEVICE:
        print(f"Selected device: {selected}")
        _PRINTED_DEVICE = True
    return torch_module.device(selected)


def reset_device_print_state() -> None:
    global _PRINTED_DEVICE
    _PRINTED_DEVICE = False


def resolve_num_workers(value="auto", *, system: str | None = None, cpu_count: int | None = None) -> int:
    if value is None or value == "auto":
        system = system or platform.system()
        if system == "Darwin":
            return 0
        cpu_count = os.cpu_count() if cpu_count is None else cpu_count
        return max(0, min(2, int(cpu_count or 0)))
    workers = int(value)
    if workers < 0:
        raise ValueError("DataLoader num_workers must be non-negative")
    return workers


def mps_operation_error_hint(exc: Exception) -> RuntimeError:
    return RuntimeError(f"MPS operation failed: {exc}. Re-run with --device cpu if this operation is not supported on MPS.")
