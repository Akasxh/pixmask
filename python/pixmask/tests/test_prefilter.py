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
    lib.pixmask_cubic_b_spline_prefilter.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.pixmask_cubic_b_spline_prefilter.restype = None
    return lib


def _make_array(values):
    array_type = ctypes.c_float * len(values)
    return array_type(*values)


def _run_prefilter(lib: ctypes.CDLL, src_values, width: int, height: int, channels: int):
    src = _make_array(src_values)
    dst = _make_array([0.0] * len(src_values))
    lib.pixmask_cubic_b_spline_prefilter(src, dst, width, height, channels)
    return [dst[i] for i in range(len(src_values))]


def test_prefilter_preserves_constant_field(pixmask_lib: ctypes.CDLL) -> None:
    width, height, channels = 5, 4, 2
    constant_value = 0.5
    src = [constant_value] * (width * height * channels)
    filtered = _run_prefilter(pixmask_lib, src, width, height, channels)
    assert len(filtered) == len(src)
    for value in filtered:
        assert value == pytest.approx(constant_value, rel=0.0, abs=1e-5)


def test_prefilter_monotonic_gradient(pixmask_lib: ctypes.CDLL) -> None:
    width, height, channels = 7, 6, 3
    total = width * height
    src = []
    for y in range(height):
        for x in range(width):
            base = (y * width + x) / (total - 1)
            for c in range(channels):
                src.append(base + c * 0.01)
    filtered = _run_prefilter(pixmask_lib, src, width, height, channels)

    def assert_non_decreasing(sequence):
        for i in range(1, len(sequence)):
            assert sequence[i] + 1e-6 >= sequence[i - 1]

    for y in range(height):
        for c in range(channels):
            row = [filtered[(y * width + x) * channels + c] for x in range(width)]
            assert_non_decreasing(row)

    for x in range(width):
        for c in range(channels):
            column = [
                filtered[(y * width + x) * channels + c] for y in range(height)
            ]
            assert_non_decreasing(column)


def test_prefilter_produces_finite_output(pixmask_lib: ctypes.CDLL) -> None:
    width, height, channels = 8, 5, 1
    src = []
    for y in range(height):
        for x in range(width):
            src.append(((x + 1) * (y + 2)) % 13 / 13.0)
    filtered = _run_prefilter(pixmask_lib, src, width, height, channels)

    for value in filtered:
        assert not math.isnan(value)
        assert math.isfinite(value)
