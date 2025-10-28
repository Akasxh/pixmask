import importlib


def test_import() -> None:
    module = importlib.import_module("pixmask._native")
    assert hasattr(module, "_ext")
