"""Native bindings for the pixmask library."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import numpy as np

def _load_extension():
    try:
        return importlib.import_module('._pixmask', __name__.rsplit('.', 1)[0])
    except ImportError as exc:
        root = Path(__file__).resolve().parents[2]
        search_dirs = [
            root / 'build' / 'python' / 'pixmask',
            root / 'build' / 'Release' / 'python' / 'pixmask',
            root / 'build' / 'Debug' / 'python' / 'pixmask',
        ]
        suffixes = importlib.machinery.EXTENSION_SUFFIXES
        for directory in search_dirs:
            if not directory.exists():
                continue
            for suffix in suffixes:
                candidate = directory / f'_pixmask{suffix}'
                if candidate.exists():
                    spec = importlib.util.spec_from_file_location('pixmask._pixmask', candidate)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        sys.modules.setdefault('pixmask._pixmask', module)
                        return module
        raise exc

try:  # pragma: no cover - import is validated in tests
    _ext = _load_extension()
except ImportError as exc:  # pragma: no cover - optional native module
    _ext = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_extension() -> None:
    if _ext is None:  # pragma: no cover - exercised when extension missing
        raise RuntimeError('pixmask native extension is unavailable') from _IMPORT_ERROR


def version() -> str:
    """Return the semantic version string reported by the native library."""

    _require_extension()
    return _ext.version()


def sanitize(image: np.ndarray, **kwargs: object) -> np.ndarray:
    """Run the sanitize pipeline on ``image``.

    Parameters
    ----------
    image:
        Input image as a NumPy array with shape ``(H, W, 3)`` or ``(H, W, 4)``.
    **kwargs:
        Forwarded to the native binding. Currently supports ``output_dtype``
        with values ``"uint8"`` or ``"float32"``.
    """

    _require_extension()
    return _ext.sanitize(image, **kwargs)


def exceeds_pixel_cap(width: int, height: int, cap_megapixels: float) -> bool:
    """Return ``True`` when ``width`` × ``height`` exceeds ``cap_megapixels``."""

    _require_extension()
    return _ext.security.exceeds_pixel_cap(int(width), int(height), float(cap_megapixels))


def suspicious_polyglot_bytes(buffer: object) -> bool:
    """Return ``True`` if ``buffer`` contains suspicious polyglot signatures."""

    _require_extension()
    return _ext.security.suspicious_polyglot_bytes(buffer)


__all__ = ['sanitize', 'version', 'exceeds_pixel_cap', 'suspicious_polyglot_bytes']
