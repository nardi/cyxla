"""Tests of the disk cache of lowered programs and executables."""

import importlib.metadata
import os
from pathlib import Path, PurePosixPath

import jax
import jax.numpy as jnp
import pytest

import cyxla as cx
from cyxla import _cache, _plugins
from cyxla._cache import (
    decode_lowered,
    distribution_identity,
    encode_lowered,
    executable_key,
)

PROGRAM = cx.LoweredProgram(code=b"\x00code\xff", text="module @x {}")
"""A stand-in for a lowered program, enough for encoding."""

PROGRAMS = (PROGRAM, None, PROGRAM, None)
"""Lowered programs with two of them left out."""

SHAPES = (jax.ShapeDtypeStruct((2,), jnp.float64),)
"""Arguments of the lowered test functions."""


def test_encoding_round_trip() -> None:
    """Checks that decoding returns the encoded programs, for any number of them."""
    assert decode_lowered(encode_lowered(PROGRAMS)) == PROGRAMS
    assert decode_lowered(encode_lowered(())) == ()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"other", "Not a file"),
        (encode_lowered(PROGRAMS)[:-3], "past the end"),
        (encode_lowered(PROGRAMS)[:30], "Malformed"),
        (encode_lowered(PROGRAMS) + b"x", "trailing bytes"),
        (encode_lowered(PROGRAMS).replace(b"module", b"\xffodule", 1), "Malformed"),
    ],
)
def test_malformed_files(data: bytes, message: str) -> None:
    """Checks that malformed files raise a `ValueError`."""
    with pytest.raises(ValueError, match=message):
        decode_lowered(data)


def test_store_and_load(cache_directory: Path) -> None:
    """Checks that stored programs load and that a malformed file is removed."""
    assert cx.load_lowered("missing") is None
    cx.store_lowered("key", PROGRAMS)
    assert cx.load_lowered("key") == PROGRAMS
    path = cache_directory / "key.lowered"
    path.write_bytes(b"broken")
    assert cx.load_lowered("key") is None
    assert not path.exists()


def test_cache_directory_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Checks that the cache lives under the home directory without the variable."""
    monkeypatch.delenv(cx.CACHE_ENVIRONMENT_VARIABLE)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert cx.cache_directory() == tmp_path / ".cache" / "cyxla"
    assert cx.cache_directory().is_dir()


def test_cache_keys() -> None:
    """Checks that keys depend on every part, on part boundaries and on the libraries."""
    key = cx.cache_key("jaxpr", b"options")
    assert key == cx.cache_key("jaxpr", b"options")
    assert key != cx.cache_key("jaxp", "roptions")
    assert key != cx.cache_key("jaxpr", b"options", distributions=("numpy",))


def test_distribution_identity() -> None:
    """Checks that a distribution installed from a directory is hashed by its files.

    `cyxla-pjrt-cpu` comes from a path source in development, and `numpy` from an index.
    """
    local = distribution_identity("cyxla-pjrt-cpu")
    assert local == distribution_identity("cyxla-pjrt-cpu")
    assert distribution_identity("numpy") != local


class FakeDistribution:
    """A distribution installed from a directory, which lists a file that is gone."""

    version = "1.0"
    files = [PurePosixPath("package/kept.py"), PurePosixPath("package/gone.py")]

    def __init__(self, root: Path) -> None:
        self.root = root

    def read_text(self, name: str) -> str | None:
        """Returns a `direct_url.json` and no other metadata file."""
        return "{}" if name == "direct_url.json" else None

    def locate_file(self, file: PurePosixPath) -> Path:
        """Returns the path of a listed file under the root."""
        return self.root / file


def test_distribution_identity_of_missing_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Checks that listed files that are gone are skipped and that kept ones count."""
    (tmp_path / "package").mkdir()
    kept = tmp_path / "package" / "kept.py"
    kept.write_text("a = 1")
    distribution = FakeDistribution(tmp_path)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    first = distribution_identity("fake")
    kept.write_text("a = 2")
    assert distribution_identity("fake") != first


def test_executable_key_needs_a_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that executable keys depend on the loaded plugin."""
    key = executable_key(PROGRAM)
    identity = _plugins.loaded_plugin
    assert identity is not None
    monkeypatch.setattr(_plugins, "loaded_plugin", identity._replace(size=0))
    assert executable_key(PROGRAM) != key
    monkeypatch.setattr(_plugins, "loaded_plugin", None)
    with pytest.raises(RuntimeError, match="No PJRT plugin"):
        executable_key(PROGRAM)


def test_executable_key_ignores_locations() -> None:
    """Checks that programs that differ only in source locations share a key."""
    first = cx.lower(lambda x: jnp.sin(x) * 2, SHAPES)

    second = cx.lower(lambda x: jnp.sin(x) * 2, SHAPES)
    assert first.text != second.text
    assert executable_key(first) == executable_key(second)


def test_compile_program_uses_the_disk_cache(cache_directory: Path) -> None:
    """Checks that executables are stored, loaded, and compiled again when corrupted."""
    program = cx.lower(jnp.cos, SHAPES)
    path = cache_directory / f"{executable_key(program)}.pjrt"
    path.unlink(missing_ok=True)
    _, statistics = cx.compile_program(program)
    assert statistics.compiled == 1 and path.exists()
    _, statistics = cx.compile_program(program)
    assert statistics.loaded == 1 and statistics.compiled == 0
    path.write_bytes(b"corrupted")
    _, statistics = cx.compile_program(program)
    assert statistics.compiled == 1
    assert path.read_bytes() != b"corrupted"
    modified = os.stat(path).st_mtime_ns
    _, statistics = cx.compile_program(program, disk_cache=False)
    assert statistics.compiled == 1 and os.stat(path).st_mtime_ns == modified


def test_statistics_add() -> None:
    """Checks that statistics add field by field."""
    first = cx.CompileStatistics(compile_seconds=1.0, compiled=1)
    second = cx.CompileStatistics(load_seconds=0.5, loaded=2)
    assert first.add(second) == cx.CompileStatistics(1.0, 0.5, 1, 2)


def test_enable_x64() -> None:
    """Checks that JAX makes 64-bit arrays after the switch."""
    _cache.enable_x64()
    assert jnp.zeros(1).dtype == jnp.float64
