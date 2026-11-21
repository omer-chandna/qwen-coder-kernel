"""Loads build/adakernel.dll (the Qwen3.5 runtime) and the CUDA runtime DLLs it links against."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class GenStats(ctypes.Structure):
    _fields_ = [
        ("prefill_ms", ctypes.c_float),
        ("decode_ms", ctypes.c_float),
        ("ttft_ms", ctypes.c_float),
        ("graph_capture_ms", ctypes.c_float),
        ("prompt_tokens", ctypes.c_int32),
        ("generated_tokens", ctypes.c_int32),
        ("backend", ctypes.c_int32),
        ("launches_per_token", ctypes.c_int32),
        ("decode_tps", ctypes.c_float),
        ("prefill_tps", ctypes.c_float),
        ("vram_peak_mb", ctypes.c_float),
        ("weight_vram_mb", ctypes.c_float),
        ("host_weight_mb", ctypes.c_float),
        ("n_gpu_layers", ctypes.c_int32),
        ("persistent_blocks", ctypes.c_int32),
        ("status", ctypes.c_int32),
        ("graph_nodes", ctypes.c_int32),
    ]


def _cuda_bins() -> list[Path]:
    """Folders holding cudart / cublas: the toolkit scripts/build.ps1 used (build/cuda_bin.txt), then CUDA_PATH."""
    bins: list[Path] = []
    rec = ROOT / "build" / "cuda_bin.txt"
    if rec.exists():
        bins.extend(Path(p) for p in rec.read_text(encoding="utf-8-sig").splitlines() if p.strip())
    env = os.environ.get("CUDA_PATH")
    if env:
        bins.append(Path(env) / "bin")
        bins.append(Path(env) / "bin" / "x64")
    return [p for p in bins if p.exists()]


def dll_path() -> Path:
    override = os.environ.get("ADAKERNEL_DLL")
    return Path(override) if override else ROOT / "build" / "adakernel.dll"


class Native:
    def __init__(self, path: Path | None = None):
        for folder in _cuda_bins():
            os.add_dll_directory(str(folder))
        location = path or dll_path()
        if not location.exists():
            raise FileNotFoundError(f"adakernel.dll was not found at {location}. Build it with scripts/build.ps1.")
        os.add_dll_directory(str(location.parent))
        self.lib = ctypes.WinDLL(str(location))
        self.lib.ak_last_error.restype = ctypes.c_char_p

    def check(self, status: int) -> None:
        if status != 0:
            raw = self.lib.ak_last_error()
            message = raw.decode("utf-8", errors="replace") if raw else f"status {status}"
            raise RuntimeError(message)

    @staticmethod
    def _ptr(array: np.ndarray | None):
        if array is None:
            return None
        if not array.flags["C_CONTIGUOUS"]:
            raise ValueError("array must be contiguous")
        return array.ctypes.data_as(ctypes.c_void_p)
