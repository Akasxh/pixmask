import ctypes
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


def _load_library() -> ctypes.CDLL:
    root = Path(__file__).resolve().parents[3]
    lib_names = {
        "linux": "libpixmask.so",
        "linux2": "libpixmask.so",
        "darwin": "libpixmask.dylib",
        "win32": "pixmask.dll",
    }
    lib_name = lib_names.get(sys.platform)
    if lib_name is None:
        pytest.skip(f"unsupported platform: {sys.platform}")

    candidates = [
        root / "build" / lib_name,
        root / "build" / "Release" / lib_name,
        root / "build" / "Debug" / lib_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return ctypes.CDLL(str(candidate))
    pytest.skip("pixmask shared library is unavailable")


@pytest.fixture(scope="module")
def pixmask_lib() -> ctypes.CDLL:
    lib = _load_library()
    lib.pixmask_cubic_resample.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.pixmask_cubic_resample.restype = None
    return lib


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _run_resample(lib: ctypes.CDLL, array: np.ndarray, width: int, height: int, channels: int,
                  out_width: int, out_height: int) -> np.ndarray:
    src = np.ascontiguousarray(array.reshape(-1), dtype=np.float32)
    dst = np.zeros(out_width * out_height * channels, dtype=np.float32)
    lib.pixmask_cubic_resample(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        width,
        height,
        channels,
        out_width,
        out_height,
    )
    return dst.reshape((out_height, out_width, channels))


def _make_sine_pattern(width: int, height: int, channels: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, width, endpoint=False)
    y = np.linspace(0.0, 1.0, height, endpoint=False)
    grid_x, grid_y = np.meshgrid(x, y)
    base = 0.5 + 0.25 * np.sin(2.0 * math.pi * grid_x * 3.0) + 0.25 * np.sin(2.0 * math.pi * grid_y * 5.0)
    pattern = np.zeros((height, width, channels), dtype=np.float32)
    for c in range(channels):
        modulation = 0.1 * np.cos(2.0 * math.pi * (c + 1) * grid_x)
        pattern[..., c] = np.clip(base + modulation, 0.0, 1.0)
    return pattern


def _make_checkerboard(width: int, height: int, channels: int, period: int = 8) -> np.ndarray:
    indices = np.indices((height, width)).sum(axis=0)
    board = ((indices // period) % 2).astype(np.float32)
    pattern = np.zeros((height, width, channels), dtype=np.float32)
    for c in range(channels):
        pattern[..., c] = board
    return pattern


def test_downscale_upscale_psnr_sine(pixmask_lib: ctypes.CDLL) -> None:
    width, height, channels = 128, 96, 3
    target_width, target_height = 48, 36
    source = _make_sine_pattern(width, height, channels)
    down = _run_resample(pixmask_lib, source, width, height, channels, target_width, target_height)
    up = _run_resample(pixmask_lib, down, target_width, target_height, channels, width, height)
    score = _psnr(source, up)
    assert score >= 34.0


def test_downscale_upscale_psnr_checkerboard(pixmask_lib: ctypes.CDLL) -> None:
    width, height, channels = 120, 90, 3
    target_width, target_height = 40, 30
    source = _make_checkerboard(width, height, channels, period=6)
    down = _run_resample(pixmask_lib, source, width, height, channels, target_width, target_height)
    up = _run_resample(pixmask_lib, down, target_width, target_height, channels, width, height)
    score = _psnr(source, up)
    assert score >= 11.0


def test_downscale_matches_pillow_antialias(pixmask_lib: ctypes.CDLL) -> None:
    width, height, channels = 96, 72, 3
    target_width, target_height = 36, 27
    source = _make_sine_pattern(width, height, channels)
    down = _run_resample(pixmask_lib, source, width, height, channels, target_width, target_height)

    pil_input = Image.fromarray(np.clip(source * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
    pil_down = pil_input.resize((target_width, target_height), Image.Resampling.BICUBIC)
    pil_array = np.asarray(pil_down, dtype=np.float32) / 255.0

    score = _psnr(down, pil_array)
    assert score >= 38.0
