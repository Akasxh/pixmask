"""Type stubs for pixmask public API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union, overload

import numpy as np
from numpy.typing import NDArray

__version__: str

@dataclass(frozen=True)
class SanitizeResult:
    image: NDArray[np.uint8]
    preset: str
    warnings: list[str]

# Overload 1: return_metadata=True → SanitizeResult (highest priority)
@overload
def sanitize(
    image: Union[NDArray[np.uint8], bytes, str, Path],
    *,
    preset: str = ...,
    bit_depth: int | None = ...,
    jpeg_quality: tuple[int, int] | None = ...,
    median_radius: int | None = ...,
    output_format: str | None = ...,
    return_metadata: Literal[True],
) -> SanitizeResult: ...

# Overload 2: output_format set → bytes
@overload
def sanitize(
    image: Union[NDArray[np.uint8], bytes, str, Path],
    *,
    preset: str = ...,
    bit_depth: int | None = ...,
    jpeg_quality: tuple[int, int] | None = ...,
    median_radius: int | None = ...,
    output_format: str,
    return_metadata: Literal[False] = ...,
) -> bytes: ...

# Overload 3: default return → ndarray
@overload
def sanitize(
    image: Union[NDArray[np.uint8], bytes, str, Path],
    *,
    preset: str = ...,
    bit_depth: int | None = ...,
    jpeg_quality: tuple[int, int] | None = ...,
    median_radius: int | None = ...,
    output_format: None = ...,
    return_metadata: Literal[False] = ...,
) -> NDArray[np.uint8]: ...

# General signature
def sanitize(
    image: Union[NDArray[np.uint8], bytes, str, Path],
    *,
    preset: str = "balanced",
    bit_depth: int | None = None,
    jpeg_quality: tuple[int, int] | None = None,
    median_radius: int | None = None,
    output_format: str | None = None,
    return_metadata: bool = False,
) -> Union[NDArray[np.uint8], bytes, SanitizeResult]:
    """Sanitize an image for safe use with multimodal LLMs.

    Args:
        image: numpy ndarray (HWC uint8), raw bytes (JPEG/PNG),
            Path to image file, or PIL.Image.
        preset: "fast", "balanced" (default), or "paranoid".
        bit_depth: Override bit depth (1-8).
        jpeg_quality: Override JPEG quality range as (lo, hi).
        median_radius: Override median filter radius (0=off, 1=3x3).
        output_format: If "bytes", "jpeg", or "png", return encoded bytes.
        return_metadata: If True, return SanitizeResult with metadata.

    Returns:
        numpy ndarray (HWC uint8), bytes, or SanitizeResult.
    """
    ...
