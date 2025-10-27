import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def _pythonpath(root: Path) -> str:
    candidates = [
        root / "python",
        root / "build" / "python",
        root / "build" / "Release" / "python",
        root / "build" / "Debug" / "python",
    ]
    parts = [str(path) for path in candidates if path.exists()]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _run_cli(args: List[str], *, env: Dict[str, str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "pixmask", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)


def test_cli_roundtrip(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath(root)

    height, width = 64, 64
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, 1, width, endpoint=False),
        np.linspace(0, 1, height, endpoint=False),
        indexing="xy",
    )
    rgb_u8 = np.stack(
        [
            (grid_x * 255.0).astype(np.uint8),
            (grid_y * 255.0).astype(np.uint8),
            ((1.0 - grid_x) * 255.0).astype(np.uint8),
        ],
        axis=-1,
    )

    input_u8 = tmp_path / "input_u8.npy"
    output_u8 = tmp_path / "output_u8.npy"
    np.save(input_u8, rgb_u8)
    _run_cli([str(input_u8), str(output_u8)], env=env)
    result_u8 = np.load(output_u8, allow_pickle=False)
    assert result_u8.shape == rgb_u8.shape
    assert result_u8.dtype == np.uint8

    rgb_f32 = np.stack(
        [
            grid_x.astype(np.float32),
            grid_y.astype(np.float32),
            np.full_like(grid_x, 0.5, dtype=np.float32),
        ],
        axis=-1,
    )

    input_f32 = tmp_path / "input_f32.npy"
    output_f32 = tmp_path / "output_f32.npy"
    np.save(input_f32, rgb_f32)
    _run_cli(
        [str(input_f32), str(output_f32), "--output-dtype", "float32", "--no-sr"],
        env=env,
    )
    result_f32 = np.load(output_f32, allow_pickle=False)
    assert result_f32.shape == rgb_f32.shape
    assert result_f32.dtype == np.float32
