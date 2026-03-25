"""Tests for pixmask.sanitize() — the primary public API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pixmask
from pixmask import SanitizeResult, sanitize


# ---------------------------------------------------------------------------
# Input type acceptance
# ---------------------------------------------------------------------------


class TestNumpyInput:
    """sanitize() accepts HWC uint8 numpy arrays."""

    def test_rgb_array(self, solid_red_array: np.ndarray) -> None:
        result = sanitize(solid_red_array)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.ndim == 3

    def test_preserves_spatial_dims(self, gradient_array: np.ndarray) -> None:
        result = sanitize(gradient_array)
        h, w, _ = gradient_array.shape
        assert result.shape[0] == h
        assert result.shape[1] == w

    def test_preserves_channels(self, solid_red_array: np.ndarray) -> None:
        result = sanitize(solid_red_array)
        assert result.shape[2] == solid_red_array.shape[2]

    def test_gray_array(self, gray_array: np.ndarray) -> None:
        result = sanitize(gray_array)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8


class TestBytesInput:
    """sanitize() accepts raw PNG/JPEG bytes."""

    def test_png_bytes(self, solid_red_bytes: bytes) -> None:
        result = sanitize(solid_red_bytes)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.ndim == 3

    def test_output_shape(self, gradient_bytes: bytes) -> None:
        result = sanitize(gradient_bytes)
        assert result.shape[0] > 0
        assert result.shape[1] > 0
        assert result.shape[2] in (1, 3, 4)


class TestPathInput:
    """sanitize() accepts Path objects."""

    def test_path_object(self, solid_red_path: Path) -> None:
        result = sanitize(solid_red_path)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_str_path(self, gradient_path: Path) -> None:
        result = sanitize(str(gradient_path))
        assert isinstance(result, np.ndarray)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            sanitize(tmp_path / "nonexistent.png")


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    """Preset parameter selects different sanitization strength."""

    @pytest.mark.parametrize("preset", ["fast", "balanced", "paranoid"])
    def test_valid_presets(
        self, preset: str, noise_bytes: bytes
    ) -> None:
        result = sanitize(noise_bytes, preset=preset)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_invalid_preset_raises(self, solid_red_bytes: bytes) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            sanitize(solid_red_bytes, preset="nonexistent")


# ---------------------------------------------------------------------------
# Parameter overrides
# ---------------------------------------------------------------------------


class TestOverrides:
    """Custom bit_depth / jpeg_quality / median_radius."""

    @pytest.mark.parametrize("bd", [1, 4, 5, 8])
    def test_bit_depth_range(self, bd: int, noise_bytes: bytes) -> None:
        result = sanitize(noise_bytes, bit_depth=bd)
        assert isinstance(result, np.ndarray)

    def test_bit_depth_out_of_range(self, solid_red_bytes: bytes) -> None:
        with pytest.raises(ValueError, match="bit_depth"):
            sanitize(solid_red_bytes, bit_depth=0)
        with pytest.raises(ValueError, match="bit_depth"):
            sanitize(solid_red_bytes, bit_depth=9)

    def test_jpeg_quality_tuple(self, noise_bytes: bytes) -> None:
        result = sanitize(noise_bytes, jpeg_quality=(60, 80))
        assert isinstance(result, np.ndarray)

    def test_jpeg_quality_invalid(self, solid_red_bytes: bytes) -> None:
        with pytest.raises(ValueError, match="jpeg_quality"):
            sanitize(solid_red_bytes, jpeg_quality=(90, 80))

    def test_median_radius(self, noise_bytes: bytes) -> None:
        result = sanitize(noise_bytes, median_radius=0)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


class TestReturnMetadata:
    """return_metadata=True wraps output in SanitizeResult."""

    def test_returns_dataclass(self, noise_bytes: bytes) -> None:
        result = sanitize(noise_bytes, return_metadata=True)
        assert isinstance(result, SanitizeResult)
        assert isinstance(result.image, np.ndarray)
        assert isinstance(result.preset, str)
        assert isinstance(result.warnings, list)

    def test_preset_in_result(self, noise_bytes: bytes) -> None:
        result = sanitize(noise_bytes, preset="fast", return_metadata=True)
        assert isinstance(result, SanitizeResult)
        assert result.preset == "fast"


class TestOutputFormat:
    """output_format returns encoded bytes."""

    def test_png_output(self, noise_bytes: bytes) -> None:
        result = sanitize(noise_bytes, output_format="png")
        assert isinstance(result, bytes)
        # PNG signature
        assert result[:4] == b"\x89PNG"

    def test_bytes_alias(self, noise_bytes: bytes) -> None:
        result = sanitize(noise_bytes, output_format="bytes")
        assert isinstance(result, bytes)

    def test_invalid_format_raises(self, solid_red_bytes: bytes) -> None:
        with pytest.raises(ValueError, match="Unknown output_format"):
            sanitize(solid_red_bytes, output_format="bmp")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    """Invalid inputs produce clear errors."""

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported image type"):
            sanitize(42)  # type: ignore[arg-type]

    def test_float_array_raises(self) -> None:
        arr = np.ones((8, 8, 3), dtype=np.float32)
        with pytest.raises(TypeError, match="uint8"):
            sanitize(arr)

    def test_corrupt_bytes_raises(self) -> None:
        with pytest.raises(RuntimeError):
            sanitize(b"\x00\x01\x02\x03" * 10)

    def test_empty_bytes_raises(self) -> None:
        with pytest.raises(RuntimeError):
            sanitize(b"")


# ---------------------------------------------------------------------------
# Output validity
# ---------------------------------------------------------------------------


class TestOutputValidity:
    """Sanitized output has valid shape and dtype."""

    def test_contiguous(self, noise_bytes: bytes) -> None:
        result = sanitize(noise_bytes)
        assert result.flags["C_CONTIGUOUS"]

    def test_values_in_uint8_range(self, noise_bytes: bytes) -> None:
        result = sanitize(noise_bytes)
        assert result.min() >= 0
        assert result.max() <= 255
