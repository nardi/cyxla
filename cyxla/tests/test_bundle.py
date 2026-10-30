"""Tests of the single-file container."""

import struct
from pathlib import Path

import numpy as np
import pytest

import cyxla as cx
from cyxla import _bundle


def make_bundle() -> bytes:
    """Returns a bundle with two blobs and some metadata."""
    writer = cx.BundleWriter("test.object", 3)
    writer.add("bytes", b"hello")
    writer.add("array", np.arange(5, dtype=np.int32))
    return writer.to_bytes({"answer": 42, "names": ["a", "b"]})


def test_round_trip_keeps_blobs_and_metadata() -> None:
    """Checks that blobs and the header come back, and that blobs start aligned."""
    bundle = cx.Bundle(make_bundle(), "test.object", 3)
    assert bundle.meta == {"answer": 42, "names": ["a", "b"]}
    assert bytes(bundle.blob("bytes")) == b"hello"
    np.testing.assert_array_equal(bundle.array("array", np.int32), np.arange(5))
    assert all(
        (bundle.start + entry.offset) % _bundle.ALIGNMENT == 0
        for entry in bundle.entries.values()
    )


def test_file_round_trip(tmp_path: Path) -> None:
    """Checks that a bundle written to a file reads back and leaves no temporary file."""
    writer = cx.BundleWriter("test.object", 1)
    writer.add("blob", memoryview(b"data"))
    path = tmp_path / "object.test"
    writer.write(path, {})
    assert [item.name for item in tmp_path.iterdir()] == ["object.test"]
    assert bytes(cx.Bundle.read(path, "test.object", 1).blob("blob")) == b"data"


def test_empty_bundle() -> None:
    """Checks that a bundle without blobs is valid."""
    data = cx.BundleWriter("empty", 1).to_bytes({})
    assert cx.Bundle(data, "empty", 1).entries == {}


def test_blob_names_are_unique() -> None:
    """Checks that a blob name cannot be added twice."""
    writer = cx.BundleWriter("test.object", 1)
    writer.add("blob", b"1")
    with pytest.raises(ValueError, match="already has a blob"):
        writer.add("blob", b"2")


def test_wrong_kind_and_version_are_rejected() -> None:
    """Checks that a reader only takes the kind and version it asks for."""
    data = make_bundle()
    with pytest.raises(cx.BundleError, match="holds test.object, not other"):
        cx.Bundle(data, "other", 3)
    with pytest.raises(cx.BundleError, match="has version 3 of test.object"):
        cx.Bundle(data, "test.object", 4)


def test_malformed_files_are_rejected() -> None:
    """Checks the errors for text, a short file, a wrong container version and cut files."""
    data = make_bundle()
    with pytest.raises(cx.BundleError, match="too short"):
        cx.Bundle(b"abc", "test.object", 3)
    with pytest.raises(cx.BundleError, match="not a bundle"):
        cx.Bundle(b"x" * 64, "test.object", 3)
    other_version = struct.pack("<8sIQ", _bundle.MAGIC, 99, 10) + data[20:]
    with pytest.raises(cx.BundleError, match="container version 99"):
        cx.Bundle(other_version, "test.object", 3)
    with pytest.raises(cx.BundleError, match="cut off in its header"):
        cx.Bundle(data[:30], "test.object", 3)
    with pytest.raises(cx.BundleError, match="cut off"):
        cx.Bundle(data[:-1], "test.object", 3)
    garbage = _bundle.PREFIX.pack(_bundle.MAGIC, 1, 5) + b"{no}}" + bytes(64)
    with pytest.raises(cx.BundleError, match="header of the bundle is malformed"):
        cx.Bundle(garbage, "test.object", 3)


def test_damaged_and_missing_blobs_are_reported() -> None:
    """Checks that the CRC catches a changed blob and that unknown names fail."""
    data = bytearray(make_bundle())
    bundle = cx.Bundle(bytes(data), "test.object", 3)
    data[bundle.start] ^= 0xFF
    damaged = cx.Bundle(bytes(data), "test.object", 3)
    with pytest.raises(cx.BundleError, match="blob bytes is damaged"):
        damaged.blob("bytes")
    assert bytes(damaged.blob("array")[:4]) == np.int32(0).tobytes()
    with pytest.raises(cx.BundleError, match="no blob named nothing"):
        bundle.blob("nothing")
    # A blob that was checked once is not checked again.
    assert bytes(bundle.blob("bytes")) == b"hello"
    assert bytes(bundle.blob("bytes")) == b"hello"
