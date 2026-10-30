"""A single-file container for the data of a serialized object.

A bundle holds a JSON header and named binary blobs, such as serialized executables and index
arrays. The header says what kind of object the bundle holds, and the blobs are aligned and
checked with a CRC when they are read. A reader takes the blobs it needs as `memoryview`s of
the file's bytes, so a blob that is never read costs nothing.

The layout is the 8 bytes of `MAGIC`, the format version as an unsigned 32-bit integer, the
size of the header as an unsigned 64-bit integer, the header as UTF-8 JSON, and then the
blobs. Each blob starts at a multiple of `ALIGNMENT` bytes from the start of the file.
"""

import json
import os
import struct
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

MAGIC = b"CYXLABND"
"""First bytes of a bundle."""

FORMAT_VERSION = 1
"""Version of the layout above, which changes when the container itself changes."""

ALIGNMENT = 64
"""Blobs start at multiples of this many bytes, which keeps arrays and executables aligned."""

PREFIX = struct.Struct("<8sIQ")
"""The magic, the format version and the size of the header."""


class BundleError(ValueError):
    """A bundle is malformed, damaged or of another kind or version than expected."""


class BlobEntry(NamedTuple):
    """Where a blob is, relative to the start of the data section."""

    offset: int
    size: int
    crc32: int


def align(size: int) -> int:
    """Returns the smallest multiple of `ALIGNMENT` that is at least `size`."""
    return -(-size // ALIGNMENT) * ALIGNMENT


class BundleWriter:
    """Collects blobs and writes them with a header as one bundle.

    `kind` names the library and the object, such as `xipopt.problem`, and `version` is the
    version of that library's payload. A reader has to ask for the same kind and version.
    """

    def __init__(self, kind: str, version: int) -> None:
        self.kind = kind
        self.version = version
        self.blobs: dict[str, bytes] = {}

    def add(self, name: str, data: bytes | memoryview | np.ndarray) -> None:
        """Adds a blob under a name, which has to be new."""
        if name in self.blobs:
            raise ValueError(f"The bundle already has a blob named {name}.")
        self.blobs[name] = (
            data.tobytes() if isinstance(data, np.ndarray) else bytes(data)
        )

    def to_bytes(self, meta: Mapping[str, Any]) -> bytes:
        """Returns the bundle with `meta`, a JSON-compatible mapping, in its header."""
        entries: dict[str, list[int]] = {}
        offset = 0
        for name, data in self.blobs.items():
            entries[name] = [offset, len(data), zlib.crc32(data)]
            offset = align(offset + len(data))
        header = json.dumps(
            {
                "kind": self.kind,
                "version": self.version,
                "meta": meta,
                "blobs": entries,
            }
        ).encode()
        start = align(PREFIX.size + len(header))
        parts = [PREFIX.pack(MAGIC, FORMAT_VERSION, len(header)), header]
        parts.append(bytes(start - PREFIX.size - len(header)))
        position = 0
        for name, data in self.blobs.items():
            parts.append(bytes(entries[name][0] - position))
            parts.append(data)
            position = entries[name][0] + len(data)
        return b"".join(parts)

    def write(self, path: str | os.PathLike[str], meta: Mapping[str, Any]) -> None:
        """Writes the bundle to a file, which appears under its name only when complete."""
        target = Path(path)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        temporary.write_bytes(self.to_bytes(meta))
        temporary.replace(target)


class Bundle:
    """A bundle that was read, with its header and lazily checked blobs."""

    def __init__(self, data: bytes, kind: str, version: int) -> None:
        self.data = memoryview(data)
        if len(data) < PREFIX.size:
            raise BundleError("The file is too short to be a bundle.")
        magic, format_version, header_size = PREFIX.unpack_from(data)
        if magic != MAGIC:
            raise BundleError("The file is not a bundle.")
        if format_version != FORMAT_VERSION:
            raise BundleError(
                f"The bundle has container version {format_version}, but this version of "
                f"cyxla reads version {FORMAT_VERSION}."
            )
        end = PREFIX.size + header_size
        if end > len(data):
            raise BundleError("The bundle is cut off in its header.")
        try:
            header = json.loads(bytes(self.data[PREFIX.size : end]))
            self.kind: str = header["kind"]
            self.version: int = header["version"]
            self.meta: dict[str, Any] = header["meta"]
            self.entries = {
                name: BlobEntry(*values) for name, values in header["blobs"].items()
            }
        except (ValueError, KeyError, TypeError) as error:
            raise BundleError(
                f"The header of the bundle is malformed: {error}."
            ) from error
        if self.kind != kind:
            raise BundleError(f"The bundle holds {self.kind}, not {kind}.")
        if self.version != version:
            raise BundleError(
                f"The bundle has version {self.version} of {kind}, but this library "
                f"reads version {version}."
            )
        self.start = align(end)
        total = self.start + max(
            (align(entry.offset + entry.size) for entry in self.entries.values()),
            default=0,
        )
        if len(data) < self.start + max(
            (entry.offset + entry.size for entry in self.entries.values()), default=0
        ):
            raise BundleError(f"The bundle is cut off: it needs {total} bytes.")
        self.verified: set[str] = set()

    @classmethod
    def read(cls, path: str | os.PathLike[str], kind: str, version: int) -> "Bundle":
        """Reads a bundle from a file."""
        return cls(Path(path).read_bytes(), kind, version)

    def blob(self, name: str) -> memoryview:
        """Returns a blob, after checking its CRC the first time it is read."""
        try:
            entry = self.entries[name]
        except KeyError:
            raise BundleError(f"The bundle has no blob named {name}.") from None
        begin = self.start + entry.offset
        view = self.data[begin : begin + entry.size]
        if name not in self.verified:
            if zlib.crc32(view) != entry.crc32:
                raise BundleError(f"The blob {name} is damaged.")
            self.verified.add(name)
        return view

    def array(self, name: str, dtype: np.dtype | type) -> np.ndarray:
        """Returns a blob as a read-only array of the given dtype."""
        return np.frombuffer(self.blob(name), dtype=dtype)
