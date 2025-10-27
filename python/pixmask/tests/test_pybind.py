from __future__ import annotations

import numpy as np
import pytest

from pixmask import _native


def _make_gradient(width: int, height: int, channels: int = 3, dtype: np.dtype = np.uint8) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    base = np.stack([grid_x, grid_y, 1.0 - grid_x], axis=2)
    if channels == 4:
        alpha = np.full((height, width, 1), 0.75, dtype=np.float32)
        base = np.concatenate([base, alpha], axis=2)
    if dtype == np.uint8:
        return np.clip(np.round(base * 255.0), 0.0, 255.0).astype(np.uint8)
    return base.astype(np.float32)


def test_version_returns_string() -> None:
    assert isinstance(_native.version(), str)


def test_sanitize_uint8_rgb_returns_uint8() -> None:
    image = _make_gradient(64, 64, channels=3, dtype=np.uint8)
    original = image.copy()

    result = _native.sanitize(image)

    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8
    assert np.isfinite(result).all()
    # input array should remain untouched
    assert np.array_equal(image, original)


def test_sanitize_accepts_rgba_and_drops_alpha() -> None:
    image = _make_gradient(64, 64, channels=4, dtype=np.uint8)
    result = _native.sanitize(image)

    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8


def test_sanitize_float32_defaults_to_float32() -> None:
    image = _make_gradient(64, 64, channels=3, dtype=np.float32)
    result = _native.sanitize(image)

    assert result.shape == (64, 64, 3)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_sanitize_respects_output_dtype_request() -> None:
    image = _make_gradient(64, 64, channels=3, dtype=np.float32)

    as_uint8 = _native.sanitize(image, output_dtype="uint8")
    assert as_uint8.dtype == np.uint8

    as_float = _native.sanitize(image.astype(np.uint8), output_dtype="float32")
    assert as_float.dtype == np.float32


@pytest.mark.parametrize(
    "array, expected_exception",
    [
        (np.zeros((4, 4, 3), dtype=np.float64), ValueError),
        (np.zeros((5, 4, 3), dtype=np.uint8), ValueError),
        (np.zeros((4, 4, 2), dtype=np.uint8), ValueError),
    ],
)
def test_invalid_inputs_raise(array: np.ndarray, expected_exception: type[Exception]) -> None:
    with pytest.raises(expected_exception):
        _native.sanitize(array)


def test_unexpected_kwarg_raises() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        _native.sanitize(image, unknown=True)
