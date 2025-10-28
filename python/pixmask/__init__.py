"""Python interface and CLI entry point for the pixmask project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from . import _native

sanitize = _native.sanitize
version = _native.version
exceeds_pixel_cap = _native.exceeds_pixel_cap
suspicious_polyglot_bytes = _native.suspicious_polyglot_bytes

__all__ = [
    "sanitize",
    "version",
    "exceeds_pixel_cap",
    "suspicious_polyglot_bytes",
    "main",
]


def _add_stage_flag(parser: argparse.ArgumentParser, name: str, *, help_text: str) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name,
        action="store_true",
        default=True,
        help=f"{help_text} (enabled by default; use --no-{name} to disable)",
    )
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")


def _load_array(path: Path) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise SystemExit(f"input file '{path}' does not exist") from exc
    except ValueError as exc:
        raise SystemExit(f"failed to load '{path}': {exc}") from exc


def _validate_image(array: np.ndarray) -> None:
    if array.ndim != 3:
        raise SystemExit("input array must have shape (H, W, C)")
    if array.shape[2] not in (3, 4):
        raise SystemExit("input array must have 3 (RGB) or 4 (RGBA) channels")


def _warn_disabled_stages(flags: Dict[str, bool]) -> None:
    disabled = [name for name, enabled in flags.items() if not enabled]
    if disabled:
        message = (
            "Warning: stage toggles %s are currently informational; running the "
            "full sanitize pipeline."
        ) % ", ".join(sorted(disabled))
        print(message, file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pixmask",
        description="Run the pixmask sanitize pipeline on .npy image arrays.",
    )
    parser.add_argument("input", type=Path, help="Path to the input NumPy .npy file")
    parser.add_argument("output", type=Path, help="Destination path for the sanitized .npy file")
    parser.add_argument(
        "--output-dtype",
        choices=["auto", "uint8", "float32"],
        default="auto",
        help="Output dtype selection (default: auto)",
    )
    _add_stage_flag(parser, "down", help_text="Include the 0.25× downscale stage")
    _add_stage_flag(parser, "squeeze", help_text="Include the bit-depth squeeze stage")
    _add_stage_flag(parser, "dct", help_text="Include the DCT attenuation stage")
    _add_stage_flag(parser, "sr", help_text="Include the SR-lite refinement stage")

    args = parser.parse_args(argv)

    image = _load_array(args.input)
    _validate_image(image)

    stage_flags = {name: getattr(args, name) for name in ("down", "squeeze", "dct", "sr")}
    _warn_disabled_stages(stage_flags)

    sanitize_kwargs = {}
    if args.output_dtype != "auto":
        sanitize_kwargs["output_dtype"] = args.output_dtype

    try:
        result = sanitize(image, **sanitize_kwargs)
    except Exception as exc:  # pragma: no cover - error path exercised in CLI
        print(f"pixmask sanitize failed: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
