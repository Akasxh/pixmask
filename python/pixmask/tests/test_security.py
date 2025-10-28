import numpy as np

from pixmask import _native


def test_exceeds_pixel_cap_basic():
    assert not _native.exceeds_pixel_cap(4000, 3000, 12.0)
    assert _native.exceeds_pixel_cap(6000, 4000, 12.0)
    assert _native.exceeds_pixel_cap(1, 1, 0.0)
    assert not _native.exceeds_pixel_cap(0, 123, 1.0)


def test_polyglot_signatures_detected():
    signatures = [
        b"%PDF-1.7\n", b"PK\x03\x04zip", b"7zXZpayload", b"Rar!data", b"<?xml version=\"1.0\"?>",
        b"<!DOCTYPE html>", b"MZ\x90\x00", b"\x7fELF\x02",
    ]
    for blob in signatures:
        assert _native.suspicious_polyglot_bytes(blob)


def test_polyglot_non_contiguous_buffer():
    buf = memoryview(b"abcdefgh")[::2]
    assert not _native.suspicious_polyglot_bytes(buf)

    array = np.frombuffer(b"safe-bytes", dtype=np.uint8)[::2]
    assert not _native.suspicious_polyglot_bytes(array)


def test_polyglot_negative_case():
    assert not _native.suspicious_polyglot_bytes(b"plain data with no signatures")

