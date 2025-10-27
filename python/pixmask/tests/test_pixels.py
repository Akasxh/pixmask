import ctypes
import sys
from enum import IntEnum
from pathlib import Path

import pytest


class PixelType(IntEnum):
    U8_RGB = 0
    U8_RGBA = 1
    F32_RGB = 2


class CpuImage(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
        ("stride_bytes", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
    ]


PIXEL_STRIDE = {
    PixelType.U8_RGB: 3,
    PixelType.U8_RGBA: 4,
    PixelType.F32_RGB: 3 * ctypes.sizeof(ctypes.c_float),
}


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


class ImageHolder:
    def __init__(self, pixel_type: PixelType, width: int, height: int, stride_bytes: int | None = None):
        self.pixel_type = pixel_type
        self.width = width
        self.height = height
        row_bytes = width * PIXEL_STRIDE[pixel_type]
        if stride_bytes is None:
            stride_bytes = row_bytes
        self.stride_bytes = stride_bytes
        if pixel_type == PixelType.F32_RGB:
            assert stride_bytes % ctypes.sizeof(ctypes.c_float) == 0
            element_count = (stride_bytes // ctypes.sizeof(ctypes.c_float)) * height
            self.buffer = (ctypes.c_float * element_count)()
        else:
            element_count = stride_bytes * height
            self.buffer = (ctypes.c_uint8 * element_count)()
        self.ptr = ctypes.cast(self.buffer, ctypes.c_void_p)
        self.image = CpuImage(pixel_type.value, width, height, stride_bytes, self.ptr)

    def u8_buffer(self):
        return self.buffer

    def f32_buffer(self):
        return self.buffer


@pytest.fixture(scope="module")
def pixmask_lib() -> ctypes.CDLL:
    lib = _load_library()
    lib.pixmask_validate_image.argtypes = [ctypes.POINTER(CpuImage)]
    lib.pixmask_validate_image.restype = ctypes.c_bool
    lib.pixmask_convert_image.argtypes = [ctypes.POINTER(CpuImage), ctypes.POINTER(CpuImage)]
    lib.pixmask_convert_image.restype = ctypes.c_bool
    return lib


def test_validate_image_rejects_invalid_stride(pixmask_lib: ctypes.CDLL) -> None:
    holder = ImageHolder(PixelType.U8_RGB, width=4, height=2, stride_bytes=10)
    assert not pixmask_lib.pixmask_validate_image(ctypes.byref(holder.image))


def test_u8_rgb_to_f32_roundtrip(pixmask_lib: ctypes.CDLL) -> None:
    width, height = 5, 4
    src = ImageHolder(PixelType.U8_RGB, width, height, stride_bytes=width * 3 + 5)
    f32 = ImageHolder(PixelType.F32_RGB, width, height, stride_bytes=width * 3 * ctypes.sizeof(ctypes.c_float) + 8)
    dst = ImageHolder(PixelType.U8_RGB, width, height, stride_bytes=width * 3 + 7)

    src_bytes = src.u8_buffer()
    for y in range(height):
        row_start = y * src.stride_bytes
        for x in range(width):
            base = row_start + x * 3
            src_bytes[base + 0] = (x * 17 + y * 9) % 256
            src_bytes[base + 1] = (x * 11 + y * 5) % 256
            src_bytes[base + 2] = (x * 7 + y * 3) % 256

    assert pixmask_lib.pixmask_convert_image(ctypes.byref(src.image), ctypes.byref(f32.image))

    inv255 = 1.0 / 255.0
    f32_vals = f32.f32_buffer()
    stride_floats = f32.stride_bytes // ctypes.sizeof(ctypes.c_float)
    for y in range(height):
        row_start_f32 = y * stride_floats
        row_start_src = y * src.stride_bytes
        for x in range(width):
            src_base = row_start_src + x * 3
            dst_base = row_start_f32 + x * 3
            expected = [src_bytes[src_base + c] * inv255 for c in range(3)]
            actual = [f32_vals[dst_base + c] for c in range(3)]
            assert actual == pytest.approx(expected, rel=0, abs=1e-6)

    # Prepare destination padding with a sentinel and verify it persists.
    dst_bytes = dst.u8_buffer()
    for i in range(len(dst_bytes)):
        dst_bytes[i] = 0x7F

    assert pixmask_lib.pixmask_convert_image(ctypes.byref(f32.image), ctypes.byref(dst.image))

    for y in range(height):
        row_start = y * dst.stride_bytes
        for x in range(width):
            base = row_start + x * 3
            for c in range(3):
                src_value = src_bytes[y * src.stride_bytes + x * 3 + c]
                dst_value = dst_bytes[base + c]
                assert abs(int(src_value) - int(dst_value)) <= 1
        # padding bytes should remain untouched by conversion
        padding_start = row_start + width * 3
        padding_end = row_start + dst.stride_bytes
        for index in range(padding_start, padding_end):
            assert dst_bytes[index] == 0x7F


def test_rgba_to_f32_and_back(pixmask_lib: ctypes.CDLL) -> None:
    width, height = 3, 3
    rgba = ImageHolder(PixelType.U8_RGBA, width, height, stride_bytes=width * 4 + 6)
    f32 = ImageHolder(PixelType.F32_RGB, width, height)
    out = ImageHolder(PixelType.U8_RGBA, width, height, stride_bytes=width * 4 + 4)

    rgba_bytes = rgba.u8_buffer()
    for y in range(height):
        row_start = y * rgba.stride_bytes
        for x in range(width):
            base = row_start + x * 4
            rgba_bytes[base + 0] = (x * 31 + y * 7) % 256
            rgba_bytes[base + 1] = (x * 13 + y * 17) % 256
            rgba_bytes[base + 2] = (x * 5 + y * 19) % 256
            rgba_bytes[base + 3] = (x * 3 + y * 11) % 256

    assert pixmask_lib.pixmask_convert_image(ctypes.byref(rgba.image), ctypes.byref(f32.image))
    assert pixmask_lib.pixmask_convert_image(ctypes.byref(f32.image), ctypes.byref(out.image))

    out_bytes = out.u8_buffer()
    for y in range(height):
        src_row = y * rgba.stride_bytes
        out_row = y * out.stride_bytes
        for x in range(width):
            src_base = src_row + x * 4
            out_base = out_row + x * 4
            for c in range(3):
                assert abs(int(rgba_bytes[src_base + c]) - int(out_bytes[out_base + c])) <= 1
            assert out_bytes[out_base + 3] == 0xFF

    # Unsupported conversions should fail cleanly.
    invalid_dst = ImageHolder(PixelType.U8_RGB, width, height)
    assert not pixmask_lib.pixmask_convert_image(ctypes.byref(rgba.image), ctypes.byref(invalid_dst.image))
