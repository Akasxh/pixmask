from __future__ import annotations

import ctypes
import numpy as np
import pytest

from test_pixels import ImageHolder, PixelType, CpuImage, _load_library


@np.vectorize

def _checkerboard_value(x: int, y: int) -> float:
    tile = ((x // 4) + (y // 4)) % 2
    return 0.25 if tile == 0 else 0.75


def _write_image(holder: ImageHolder, data: np.ndarray) -> None:
    assert data.shape[:2] == (holder.height, holder.width)
    view = np.ctypeslib.as_array(holder.buffer)
    view = view.reshape(holder.height, holder.stride_bytes)
    row_bytes = holder.width * 3
    u8 = np.clip(np.round(data * 255.0), 0.0, 255.0).astype(np.uint8)
    flat = u8.reshape(holder.height, row_bytes)
    for y in range(holder.height):
        view[y, :row_bytes] = flat[y]


def _read_image(holder: ImageHolder) -> np.ndarray:
    view = np.ctypeslib.as_array(holder.buffer)
    view = view.reshape(holder.height, holder.stride_bytes)
    row_bytes = holder.width * 3
    trimmed = view[:, :row_bytes]
    return trimmed.reshape(holder.height, holder.width, 3).astype(np.float32) / 255.0


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = (image * weights[None, None, :]).sum(axis=2)
    return gray.astype(np.float32)


def _laplacian_energy(image: np.ndarray) -> float:
    gray = _to_grayscale(image)
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    energy = 0.0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            window = padded[y : y + 3, x : x + 3]
            lap = float(np.sum(window * kernel))
            energy += lap * lap
    return energy / float(image.shape[0] * image.shape[1])


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    gray_a = _to_grayscale(a)
    gray_b = _to_grayscale(b)
    mu_a = float(gray_a.mean())
    mu_b = float(gray_b.mean())
    diff_a = gray_a - mu_a
    diff_b = gray_b - mu_b
    sigma_a = float(np.mean(diff_a * diff_a))
    sigma_b = float(np.mean(diff_b * diff_b))
    sigma_ab = float(np.mean(diff_a * diff_b))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return numerator / denominator


@pytest.fixture(scope="module")
def pixmask_lib() -> ctypes.CDLL:
    lib = _load_library()
    lib.pixmask_sanitize.argtypes = [ctypes.POINTER(CpuImage), ctypes.POINTER(CpuImage)]
    lib.pixmask_sanitize.restype = ctypes.c_bool
    return lib


def test_sanitize_pipeline_reduces_noise(pixmask_lib: ctypes.CDLL) -> None:
    width, height = 96, 96
    xs = np.arange(width, dtype=np.int32)
    ys = np.arange(height, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    base = _checkerboard_value(grid_x, grid_y).astype(np.float32)
    base = np.repeat(base[:, :, None], 3, axis=2)

    noise = (((grid_x * 13 + grid_y * 17) % 31).astype(np.float32) / 30.0) - 0.5
    noise = noise[..., None] * np.array([0.08, -0.06, 0.04], dtype=np.float32)
    noisy = np.clip(base + noise, 0.0, 1.0)

    src_holder = ImageHolder(PixelType.U8_RGB, width, height)
    _write_image(src_holder, noisy)

    dst_holder = ImageHolder(PixelType.U8_RGB, width, height)
    ok = pixmask_lib.pixmask_sanitize(ctypes.byref(src_holder.image), ctypes.byref(dst_holder.image))
    assert ok

    output = _read_image(dst_holder)
    assert output.shape == noisy.shape
    assert np.isfinite(output).all()

    input_energy = _laplacian_energy(noisy)
    output_energy = _laplacian_energy(output)
    assert output_energy < input_energy

    score = _ssim(output, base)
    assert score >= 0.85
