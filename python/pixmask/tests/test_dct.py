import ctypes
import math
import sys
from pathlib import Path

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
    lib.pixmask_dct8x8_hf_attenuate.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.pixmask_dct8x8_hf_attenuate.restype = None
    return lib


def _make_array(values):
    array_type = ctypes.c_float * len(values)
    return array_type(*values)


def _run_dct(lib: ctypes.CDLL, src_values, width: int, height: int, channels: int, quality: int):
    src = _make_array(src_values)
    dst = _make_array([0.0] * len(src_values))
    lib.pixmask_dct8x8_hf_attenuate(src, dst, width, height, channels, quality)
    return [dst[i] for i in range(len(src_values))]


def _hf_energy(values, width: int, height: int, channels: int) -> float:
    total = 0.0
    for c in range(channels):
        for y in range(height):
            for x in range(width - 1):
                idx0 = (y * width + x) * channels + c
                idx1 = (y * width + (x + 1)) * channels + c
                diff = values[idx1] - values[idx0]
                total += diff * diff
        for y in range(height - 1):
            for x in range(width):
                idx0 = (y * width + x) * channels + c
                idx1 = ((y + 1) * width + x) * channels + c
                diff = values[idx1] - values[idx0]
                total += diff * diff
    return total


def test_preserves_constant_field(pixmask_lib: ctypes.CDLL) -> None:
    width, height, channels = 16, 16, 3
    constant_value = 0.25
    src = [constant_value] * (width * height * channels)
    filtered = _run_dct(pixmask_lib, src, width, height, channels, 25)
    assert len(filtered) == len(src)
    for value in filtered:
        assert value == pytest.approx(constant_value, rel=0.0, abs=1e-5)


def test_quality_controls_high_frequency_energy(pixmask_lib: ctypes.CDLL) -> None:
    width, height, channels = 32, 24, 1
    src = []
    for y in range(height):
        for x in range(width):
            src.append(math.sin((x + y) * math.pi / 4.0))

    high_quality = _run_dct(pixmask_lib, src, width, height, channels, 90)
    low_quality = _run_dct(pixmask_lib, src, width, height, channels, 10)

    assert len(high_quality) == len(src) == len(low_quality)

    energy_high = _hf_energy(high_quality, width, height, channels)
    energy_low = _hf_energy(low_quality, width, height, channels)

    assert energy_low < energy_high


def test_quality_100_identity(pixmask_lib: ctypes.CDLL) -> None:
    width, height, channels = 20, 12, 2
    src = []
    for y in range(height):
        for x in range(width):
            base = (x * 3 + y * 5) % 17 / 17.0
            for c in range(channels):
                src.append(base + c * 0.01)

    restored = _run_dct(pixmask_lib, src, width, height, channels, 100)
    assert len(restored) == len(src)
    for original, output in zip(src, restored):
        assert output == pytest.approx(original, rel=0.0, abs=1e-5)
