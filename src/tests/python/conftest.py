"""Shared fixtures for pixmask Python tests."""

from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: minimal PNG encoder (no Pillow dependency)
# ---------------------------------------------------------------------------

def _encode_png(arr: np.ndarray) -> bytes:
    """Encode uint8 HWC array to minimal PNG bytes."""
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    h, w, c = arr.shape
    color_type = {1: 0, 3: 2, 4: 6}[c]

    raw_rows = []
    for y in range(h):
        raw_rows.append(b"\x00")
        raw_rows.append(arr[y].tobytes())
    compressed = zlib.compress(b"".join(raw_rows))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        chunk_data = tag + data
        crc = zlib.crc32(chunk_data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_data + struct.pack(">I", crc)

    buf = io.BytesIO()
    buf.write(b"\x89PNG\r\n\x1a\n")
    buf.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)))
    buf.write(_chunk(b"IDAT", compressed))
    buf.write(_chunk(b"IEND", b""))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------------------

@pytest.fixture
def solid_red_array() -> np.ndarray:
    """8x8 solid red image (HWC uint8)."""
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    return img


@pytest.fixture
def gradient_array() -> np.ndarray:
    """16x16 horizontal gradient (HWC uint8 RGB)."""
    grad = np.linspace(0, 255, 16, dtype=np.uint8)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :, 0] = grad[np.newaxis, :]
    img[:, :, 1] = grad[:, np.newaxis]
    img[:, :, 2] = 128
    return img


@pytest.fixture
def noise_array() -> np.ndarray:
    """32x32 random noise image (HWC uint8 RGB)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


@pytest.fixture
def gray_array() -> np.ndarray:
    """16x16 single-channel grayscale image."""
    return np.full((16, 16, 1), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# PNG bytes fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def solid_red_bytes(solid_red_array: np.ndarray) -> bytes:
    """Solid red image as PNG bytes."""
    return _encode_png(solid_red_array)


@pytest.fixture
def gradient_bytes(gradient_array: np.ndarray) -> bytes:
    """Gradient image as PNG bytes."""
    return _encode_png(gradient_array)


@pytest.fixture
def noise_bytes(noise_array: np.ndarray) -> bytes:
    """Noise image as PNG bytes."""
    return _encode_png(noise_array)


# ---------------------------------------------------------------------------
# File path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def solid_red_path(solid_red_bytes: bytes, tmp_path: Path) -> Path:
    """Write solid red PNG to a temp file and return path."""
    p = tmp_path / "solid_red.png"
    p.write_bytes(solid_red_bytes)
    return p


@pytest.fixture
def gradient_path(gradient_bytes: bytes, tmp_path: Path) -> Path:
    """Write gradient PNG to a temp file and return path."""
    p = tmp_path / "gradient.png"
    p.write_bytes(gradient_bytes)
    return p


@pytest.fixture
def noise_path(noise_bytes: bytes, tmp_path: Path) -> Path:
    """Write noise PNG to a temp file and return path."""
    p = tmp_path / "noise.png"
    p.write_bytes(noise_bytes)
    return p
