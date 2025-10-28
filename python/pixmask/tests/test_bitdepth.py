import ctypes
import sys
from pathlib import Path

import numpy as np
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
    lib.pixmask_quantize_bitdepth.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    lib.pixmask_quantize_bitdepth.restype = None
    return lib


def _gradient_energy(image: np.ndarray) -> float:
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    return float(np.mean(dx**2) + np.mean(dy**2))


def test_bitdepth_histogram_bins(pixmask_lib: ctypes.CDLL) -> None:
    width, height, channels = 40, 32, 3
    grid = np.linspace(0.0, 1.0, width * height * channels, dtype=np.float32).reshape((height, width, channels))
    original = grid.copy()

    pixmask_lib.pixmask_quantize_bitdepth(
        grid.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        width,
        height,
        channels,
        6,
    )

    assert np.all(grid >= 0.0)
    assert np.all(grid <= 1.0 + 1e-6)

    levels = 1 << 6
    quantized_bins = np.round(grid * (levels - 1)).astype(np.int32)
    assert np.unique(quantized_bins).size <= levels
    assert np.any(np.abs(grid - original) > 1e-6)


def test_bitdepth_high_frequency_energy_reduction(pixmask_lib: ctypes.CDLL) -> None:
    width, height, channels = 64, 48, 3
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)

    base = 0.5 + 0.25 * np.sin(grid_x * 0.8) + 0.2 * np.cos(grid_y * 1.1)
    image = np.zeros((height, width, channels), dtype=np.float32)
    for c in range(channels):
        modulation = 0.1 * np.sin((grid_x + c * 3.0) * 1.7)
        image[..., c] = np.clip(base + modulation, 0.0, 1.0)

    original = image.copy()

    pixmask_lib.pixmask_quantize_bitdepth(
        image.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        width,
        height,
        channels,
        0,
    )

    error = original - image
    energy_original = _gradient_energy(original)
    energy_error = _gradient_energy(error)
    assert energy_error < energy_original

    levels = 1 << 6
    quantized_bins = np.round(image * (levels - 1)).astype(np.int32)
    assert np.unique(quantized_bins).size <= levels
