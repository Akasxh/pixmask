"""pixmask: Image sanitization for multimodal LLM security."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np

from pixmask._version import __version__

# Lazy import — the C++ extension may not be installed in dev mode.
try:
    from pixmask.pixmask_ext import sanitize_bytes as _sanitize_bytes
except ImportError as _exc:
    _import_error = _exc

    def _sanitize_bytes(*args: object, **kwargs: object) -> object:
        raise ImportError(
            "pixmask C++ extension not found. "
            "Install with: pip install --no-build-isolation -ve ."
        ) from _import_error


# ---------------------------------------------------------------------------
# Presets (DECISIONS.md §1, §3)
# ---------------------------------------------------------------------------
_PRESETS: dict[str, dict[str, object]] = {
    "fast": {
        "bit_depth": 5,
        "median_radius": 0,       # skip median filter
        "jpeg_quality_lo": 75,
        "jpeg_quality_hi": 85,
    },
    "balanced": {
        "bit_depth": 5,
        "median_radius": 1,       # 3x3 median
        "jpeg_quality_lo": 70,
        "jpeg_quality_hi": 85,
    },
    "paranoid": {
        "bit_depth": 4,
        "median_radius": 1,
        "jpeg_quality_lo": 60,
        "jpeg_quality_hi": 75,
    },
}

# One-time warning about paranoid preset (DECISIONS.md §1).
_WARNED_PARANOID = False


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SanitizeResult:
    """Result of sanitization when ``return_metadata=True``."""

    image: np.ndarray
    """Sanitized image as uint8 HWC numpy array."""

    preset: str
    """Preset name used."""

    warnings: list[str] = field(default_factory=list)
    """Any warnings produced during sanitization."""


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _ndarray_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode a uint8 HWC numpy array to PNG bytes (no Pillow needed)."""
    # Minimal PNG encoder using zlib — avoids Pillow dependency.
    import struct
    import zlib

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.dtype != np.uint8:
        raise TypeError(f"Expected uint8 array, got {arr.dtype}")

    h, w, c = arr.shape
    if c not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3, or 4 channels, got {c}")

    # PNG color type: 0=gray, 2=RGB, 6=RGBA
    color_type = {1: 0, 3: 2, 4: 6}[c]

    # Build raw IDAT data: each row prefixed with filter byte 0 (none).
    raw_rows = []
    for y in range(h):
        raw_rows.append(b"\x00")  # filter: none
        raw_rows.append(arr[y].tobytes())
    raw_data = b"".join(raw_rows)
    compressed = zlib.compress(raw_data)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        chunk_data = tag + data
        crc = zlib.crc32(chunk_data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_data + struct.pack(">I", crc)

    out = io.BytesIO()
    # PNG signature
    out.write(b"\x89PNG\r\n\x1a\n")
    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    out.write(_chunk(b"IHDR", ihdr_data))
    # IDAT
    out.write(_chunk(b"IDAT", compressed))
    # IEND
    out.write(_chunk(b"IEND", b""))

    return out.getvalue()


def _coerce_to_bytes(image: Union[np.ndarray, bytes, "Path"]) -> bytes:
    """Convert various input types to raw image bytes for the C++ pipeline."""
    if isinstance(image, bytes):
        return image

    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        return path.read_bytes()

    if isinstance(image, np.ndarray):
        return _ndarray_to_png_bytes(image)

    # Try PIL.Image if available.
    try:
        import PIL.Image  # type: ignore[import-untyped]

        if isinstance(image, PIL.Image.Image):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()
    except ImportError:
        pass

    raise TypeError(
        f"Unsupported image type: {type(image).__name__}. "
        "Expected numpy ndarray, bytes, Path, or PIL.Image."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize(
    image: Union[np.ndarray, bytes, "Path"],
    *,
    preset: str = "balanced",
    bit_depth: int | None = None,
    jpeg_quality: tuple[int, int] | None = None,
    median_radius: int | None = None,
    output_format: str | None = None,
    return_metadata: bool = False,
) -> Union[np.ndarray, bytes, SanitizeResult]:
    """Sanitize an image for safe use with multimodal LLMs.

    Progressive disclosure API — see architecture/DECISIONS.md §6.

    Args:
        image: numpy ndarray (HWC uint8), raw bytes (JPEG/PNG), Path to
            image file, or PIL.Image.
        preset: ``"fast"``, ``"balanced"`` (default), or ``"paranoid"``.
        bit_depth: Override bit depth (1-8). Default depends on preset.
        jpeg_quality: Override JPEG quality range as ``(lo, hi)`` tuple.
        median_radius: Override median filter radius (0=disabled, 1=3x3).
        output_format: If ``"bytes"``, ``"jpeg"``, or ``"png"``, return
            encoded bytes instead of a numpy array.
        return_metadata: If True, return a :class:`SanitizeResult` with
            ``.image`` and ``.warnings`` attributes.

    Returns:
        numpy ndarray (HWC uint8) by default. If *output_format* is set,
        returns ``bytes``. If *return_metadata* is True, returns
        :class:`SanitizeResult`.

    Raises:
        ValueError: Invalid preset, bit_depth, or image dimensions.
        TypeError: Unsupported input type or dtype.
        RuntimeError: C++ pipeline failure (decode error, etc.).
    """
    global _WARNED_PARANOID  # noqa: PLW0603

    # Validate preset.
    if preset not in _PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}. Choose from: {', '.join(_PRESETS)}"
        )

    # One-time advisory about paranoid mode.
    if not _WARNED_PARANOID:
        _WARNED_PARANOID = True
        warnings.warn(
            "pixmask defaults to preset='balanced'. For maximum protection "
            "against adversarial images, use preset='paranoid'.",
            UserWarning,
            stacklevel=2,
        )

    # Resolve options: preset defaults, then overrides.
    p = _PRESETS[preset]
    bd = bit_depth if bit_depth is not None else int(p["bit_depth"])  # type: ignore[arg-type]
    mr = median_radius if median_radius is not None else int(p["median_radius"])  # type: ignore[arg-type]
    jq_lo = jpeg_quality[0] if jpeg_quality is not None else int(p["jpeg_quality_lo"])  # type: ignore[arg-type]
    jq_hi = jpeg_quality[1] if jpeg_quality is not None else int(p["jpeg_quality_hi"])  # type: ignore[arg-type]

    # Validate ranges.
    if not 1 <= bd <= 8:
        raise ValueError(f"bit_depth must be 1-8, got {bd}")
    if not 0 <= mr <= 3:
        raise ValueError(f"median_radius must be 0-3, got {mr}")
    if not 1 <= jq_lo <= jq_hi <= 100:
        raise ValueError(
            f"jpeg_quality must satisfy 1 <= lo <= hi <= 100, got ({jq_lo}, {jq_hi})"
        )

    # Coerce input to bytes.
    raw = _coerce_to_bytes(image)

    # Call C++ pipeline.
    result_array: np.ndarray = _sanitize_bytes(
        raw,
        bit_depth=bd,
        median_radius=mr,
        jpeg_quality_lo=jq_lo,
        jpeg_quality_hi=jq_hi,
    )

    # Collect warnings.
    warn_list: list[str] = []

    # Output encoding if requested.
    if output_format is not None:
        fmt = output_format.lower()
        if fmt in ("bytes", "png"):
            out_bytes = _ndarray_to_png_bytes(result_array)
        elif fmt == "jpeg":
            # Minimal JPEG encoding requires either PIL or returning PNG.
            # For v0.1, we use PIL if available, else raise.
            try:
                import PIL.Image  # type: ignore[import-untyped]

                pil_img = PIL.Image.fromarray(result_array)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=jq_hi)
                out_bytes = buf.getvalue()
            except ImportError as exc:
                raise ImportError(
                    "output_format='jpeg' requires Pillow. "
                    "Install with: pip install Pillow"
                ) from exc
        else:
            raise ValueError(
                f"Unknown output_format {output_format!r}. "
                "Choose 'bytes', 'png', or 'jpeg'."
            )

        if return_metadata:
            return SanitizeResult(
                image=result_array, preset=preset, warnings=warn_list
            )
        return out_bytes

    if return_metadata:
        return SanitizeResult(
            image=result_array, preset=preset, warnings=warn_list
        )
    return result_array
