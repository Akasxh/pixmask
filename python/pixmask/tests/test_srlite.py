import ctypes
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.append(str(_TEST_DIR))

from test_pixels import ImageHolder, PixelType, CpuImage, _load_library


@pytest.fixture(scope="module")
def pixmask_lib() -> ctypes.CDLL:
    lib = _load_library()
    lib.pixmask_sr_lite.argtypes = [ctypes.POINTER(CpuImage), ctypes.POINTER(CpuImage)]
    lib.pixmask_sr_lite.restype = ctypes.c_bool
    return lib


def _write_image(holder: ImageHolder, data: np.ndarray) -> None:
    assert data.shape[0] == holder.height
    assert data.shape[1] == holder.width
    view = np.ctypeslib.as_array(holder.buffer)
    view = view.reshape(holder.height, holder.stride_bytes)
    row_bytes = holder.width * 3
    scaled = np.clip(np.round(data * 255.0), 0.0, 255.0).astype(np.uint8)
    for y in range(holder.height):
        view[y, :row_bytes] = scaled[y].reshape(-1)


def _read_image(holder: ImageHolder) -> np.ndarray:
    view = np.ctypeslib.as_array(holder.buffer)
    view = view.reshape(holder.height, holder.stride_bytes)
    row_bytes = holder.width * 3
    trimmed = view[:, :row_bytes].copy()
    return trimmed.reshape(holder.height, holder.width, 3).astype(np.float32) / 255.0


def _mirror_index(idx: int, length: int) -> int:
    if length <= 1:
        return 0
    period = 2 * length - 2
    value = idx % period
    if value < 0:
        value += period
    if value >= length:
        value = period - value
    return int(value)


def _reference_sr(image: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    out_height = height * 2
    out_width = width * 2
    output = np.zeros((out_height, out_width, 3), dtype=np.float32)

    main = 1.2
    strong = -0.1
    weak = -0.05
    luma = 0.05

    strong_pairs = {
        0: ((1, 3), (2, 4)),  # TL uses up+left, weak down+right
        1: ((1, 4), (2, 3)),  # TR uses up+right, weak down+left
        2: ((2, 3), (1, 4)),  # BL uses down+left, weak up+right
        3: ((2, 4), (1, 3)),  # BR uses down+right, weak up+left
    }

    for y in range(height):
        for x in range(width):
            samples = np.zeros((3, 5), dtype=np.float32)
            for c in range(3):
                center = image[_mirror_index(y, height), _mirror_index(x, width), c]
                up = image[_mirror_index(y - 1, height), _mirror_index(x, width), c]
                down = image[_mirror_index(y + 1, height), _mirror_index(x, width), c]
                left = image[_mirror_index(y, height), _mirror_index(x - 1, width), c]
                right = image[_mirror_index(y, height), _mirror_index(x + 1, width), c]
                samples[c] = np.array([center, up, down, left, right], dtype=np.float32)

            luminance = samples[:, 0].mean()

            for sub in range(4):
                dy = sub // 2
                dx = sub % 2
                strong_axes, weak_axes = strong_pairs[sub]
                for c in range(3):
                    center, up, down, left, right = samples[c]
                    weights = {
                        1: up,
                        2: down,
                        3: left,
                        4: right,
                    }
                    value = main * center
                    for axis in strong_axes:
                        value += strong * weights[axis]
                    for axis in weak_axes:
                        value += weak * weights[axis]
                    value += luma * luminance
                    output[y * 2 + dy, x * 2 + dx, c] = np.clip(value, 0.0, 1.0)

    return output


def _gaussian_blur(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blurred = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            window = padded[y : y + 3, x : x + 3]
            blurred[y, x] = (window * kernel[:, :, None]).sum(axis=(0, 1))
    return blurred


def test_sr_lite_pixelshuffle_alignment(pixmask_lib: ctypes.CDLL) -> None:
    width, height = 6, 5
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 1.0, width, endpoint=False),
                                 np.linspace(0.0, 1.0, height, endpoint=False))
    pattern = np.zeros((height, width, 3), dtype=np.float32)
    pattern[..., 0] = np.clip(0.5 + 0.3 * np.sin(2.0 * np.pi * grid_x), 0.0, 1.0)
    pattern[..., 1] = np.clip(0.25 + 0.5 * grid_y, 0.0, 1.0)
    pattern[..., 2] = np.clip(0.75 - 0.4 * grid_x + 0.2 * grid_y, 0.0, 1.0)

    src_holder = ImageHolder(PixelType.U8_RGB, width, height)
    _write_image(src_holder, pattern)

    dst_holder = ImageHolder(PixelType.U8_RGB, width * 2, height * 2)
    ok = pixmask_lib.pixmask_sr_lite(ctypes.byref(src_holder.image), ctypes.byref(dst_holder.image))
    assert ok

    output = _read_image(dst_holder)
    quantized_input = np.clip(np.round(pattern * 255.0), 0.0, 255.0).astype(np.uint8).astype(np.float32) / 255.0
    expected = _reference_sr(quantized_input)
    assert output.shape == expected.shape
    quantized_expected = np.clip(np.round(expected * 255.0), 0.0, 255.0).astype(np.uint8)
    expected_u8 = quantized_expected.astype(np.float32) / 255.0
    assert np.allclose(output, expected_u8, atol=1e-5)


def test_sr_lite_sharpens_edges(pixmask_lib: ctypes.CDLL) -> None:
    width, height = 24, 18
    upscale = 2
    hi_width, hi_height = width * upscale, height * upscale

    high_res = np.zeros((hi_height, hi_width, 3), dtype=np.float32)
    high_res[:, : hi_width // 2, :] = 0.2
    high_res[:, hi_width // 2 :, :] = 0.85

    low_res = high_res.reshape(height, upscale, width, upscale, 3).mean(axis=(1, 3))
    low_res = _gaussian_blur(low_res)

    src_holder = ImageHolder(PixelType.U8_RGB, width, height)
    _write_image(src_holder, low_res)

    dst_holder = ImageHolder(PixelType.U8_RGB, hi_width, hi_height)
    ok = pixmask_lib.pixmask_sr_lite(ctypes.byref(src_holder.image), ctypes.byref(dst_holder.image))
    assert ok

    sr_output = _read_image(dst_holder)

    baseline_img = Image.fromarray(np.clip(low_res * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
    baseline = np.asarray(baseline_img.resize((hi_width, hi_height), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0

    center_row = hi_height // 2
    sr_grad = np.abs(np.diff(sr_output[center_row, :, 0]))
    baseline_grad = np.abs(np.diff(baseline[center_row, :, 0]))

    assert sr_grad.max() > baseline_grad.max() * 1.05

    # Ensure outputs remain bounded and finite.
    assert np.isfinite(sr_output).all()
    assert sr_output.min() >= 0.0 and sr_output.max() <= 1.0
