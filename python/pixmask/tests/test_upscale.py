import ctypes
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.append(str(_TEST_DIR))

from test_pixels import ImageHolder, PixelType, CpuImage, _load_library


class ResampleMode:
    CUBIC = 0


@pytest.fixture(scope="module")
def pixmask_lib() -> ctypes.CDLL:
    lib = _load_library()
    lib.pixmask_resize.argtypes = [
        ctypes.POINTER(CpuImage),
        ctypes.POINTER(CpuImage),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.pixmask_resize.restype = ctypes.c_bool
    return lib


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _write_image(holder: ImageHolder, data: np.ndarray) -> None:
    view = np.ctypeslib.as_array(holder.buffer)
    view = view.reshape(holder.height, holder.stride_bytes)
    row_bytes = holder.width * 3
    for y in range(holder.height):
        view[y, :row_bytes] = data[y].reshape(-1)


def _read_image(holder: ImageHolder) -> np.ndarray:
    view = np.ctypeslib.as_array(holder.buffer)
    view = view.reshape(holder.height, holder.stride_bytes)
    row_bytes = holder.width * 3
    trimmed = view[:, :row_bytes].copy()
    return trimmed.reshape(holder.height, holder.width, 3)


def _make_gradient(width: int, height: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, width, endpoint=False)
    y = np.linspace(0.0, 1.0, height, endpoint=False)
    grid_x, grid_y = np.meshgrid(x, y)
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    gradient[..., 0] = np.clip(255.0 * grid_x, 0.0, 255.0).astype(np.uint8)
    gradient[..., 1] = np.clip(255.0 * grid_y, 0.0, 255.0).astype(np.uint8)
    gradient[..., 2] = np.clip(255.0 * (0.5 * grid_x + 0.5 * grid_y), 0.0, 255.0).astype(np.uint8)
    return gradient


def test_upscale_matches_pillow_bicubic(pixmask_lib: ctypes.CDLL) -> None:
    width, height = 48, 40
    scale_x = 1.75
    scale_y = 1.5

    source = _make_gradient(width, height)
    src_holder = ImageHolder(PixelType.U8_RGB, width, height)
    _write_image(src_holder, source)

    dst_width = int(round(width * scale_x))
    dst_height = int(round(height * scale_y))
    dst_holder = ImageHolder(PixelType.U8_RGB, dst_width, dst_height)

    ok = pixmask_lib.pixmask_resize(
        ctypes.byref(src_holder.image),
        ctypes.byref(dst_holder.image),
        scale_x,
        scale_y,
        ResampleMode.CUBIC,
    )
    assert ok

    output = _read_image(dst_holder).astype(np.float32) / 255.0

    pil_input = Image.fromarray(source, mode="RGB")
    pil_resized = pil_input.resize((dst_width, dst_height), Image.Resampling.BICUBIC)
    pil_array = np.asarray(pil_resized, dtype=np.float32) / 255.0

    score = _psnr(output, pil_array)
    assert score >= 38.0
