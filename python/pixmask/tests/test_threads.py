import ctypes
import sys
import threading
from pathlib import Path

import pytest

CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_size_t, ctypes.c_void_p)


def _load_library() -> ctypes.CDLL:
    root = Path(__file__).resolve().parents[3]
    lib_names = {
        "linux": "libpixmask.so",
        "linux2": "libpixmask.so",
        "darwin": "libpixmask.dylib",
        "win32": "pixmask.dll",
    }
    lib_name = lib_names.get(sys.platform)
    if lib_name is None:
        pytest.skip(f"unsupported platform: {sys.platform}")

    candidates = [
        root / "build" / lib_name,
        root / "build" / "Release" / lib_name,
        root / "build" / "Debug" / lib_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return ctypes.CDLL(str(candidate))

    pytest.skip("pixmask shared library is unavailable")


@pytest.fixture(scope="module")
def pixmask_lib() -> ctypes.CDLL:
    lib = _load_library()
    lib.pixmask_set_threads.argtypes = [ctypes.c_size_t]
    lib.pixmask_parallel_for.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        CALLBACK,
        ctypes.c_void_p,
    ]
    lib.pixmask_thread_count.argtypes = []
    lib.pixmask_thread_count.restype = ctypes.c_size_t
    return lib


def test_set_threads_roundtrip(pixmask_lib: ctypes.CDLL) -> None:
    original = pixmask_lib.pixmask_thread_count()
    try:
        pixmask_lib.pixmask_set_threads(3)
        assert pixmask_lib.pixmask_thread_count() == 3
    finally:
        pixmask_lib.pixmask_set_threads(original)


def test_parallel_for_executes_all_indices(pixmask_lib: ctypes.CDLL) -> None:
    shared = {"hits": [0] * 512, "threads": set()}
    shared_obj = ctypes.py_object(shared)
    shared_ptr = ctypes.pointer(shared_obj)

    def _callback(index: int, opaque: int) -> None:
        container = ctypes.cast(opaque, ctypes.POINTER(ctypes.py_object)).contents.value
        container["hits"][index] += 1
        container["threads"].add(threading.get_ident())

    cb = CALLBACK(_callback)

    pixmask_lib.pixmask_set_threads(4)
    pixmask_lib.pixmask_parallel_for(0, len(shared["hits"]), cb, ctypes.cast(shared_ptr, ctypes.c_void_p))

    assert all(value == 1 for value in shared["hits"])
    # Expect at least two distinct worker threads when more than one is available.
    worker_count = pixmask_lib.pixmask_thread_count()
    if worker_count > 1:
        assert len(shared["threads"]) >= 2
    else:
        assert len(shared["threads"]) == 1

