import subprocess
import time
from pathlib import Path

import numpy as np


def _bench_binary() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "build" / "benchmarks" / "bench"


def test_cpp_benchmark_output():
    bench = _bench_binary()
    assert bench.exists(), f"benchmark binary missing at {bench}"
    result = subprocess.run([str(bench)], check=True, capture_output=True, text=True)
    stdout = result.stdout.strip().splitlines()
    assert stdout, "benchmark produced no output"
    header = stdout[0]
    assert "pixmask benchmark" in header
    stages = {line.split(":", 1)[0].strip() for line in stdout[1:]}
    expected = {"to_float", "downscale", "quantize", "dct", "blend_low", "upscale",
                "sr_prep", "sr_lite", "blend_final", "to_u8", "total"}
    missing = expected.difference(stages)
    assert not missing, f"missing timings for: {sorted(missing)}"


def test_python_pipeline_timing():
    import pixmask as pm

    h, w = 256, 256
    grid = np.arange(h * w * 3, dtype=np.float32).reshape(h, w, 3)
    grid = (grid % 256) / 255.0

    start = time.perf_counter()
    output = pm.sanitize(grid)
    duration = time.perf_counter() - start

    assert output.shape == grid.shape
    assert output.dtype == np.float32
    assert duration < 1.0, f"sanitize took too long: {duration:.3f}s"
