"""The disk cache, which lets later builds skip lowering and compiling.

Building a program has two slow steps, lowering it to StableHLO and compiling it with the
PJRT plugin. The cache stores the result of both. Lowered programs are stored under a key
that the caller computes with `cache_key` from whatever determines them, such as a jaxpr and
its options. Executables are stored under a hash of their StableHLO, the compile options and
the loaded plugin.
"""

import functools
import hashlib
import importlib.metadata
import json
import os
import re
import struct
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

from cyxla import _plugins
from cyxla._lowering import LoweredProgram
from cyxla._runtime import Executable
from cyxla._types import ArrayType

CACHE_ENVIRONMENT_VARIABLE = "CYXLA_CACHE_DIR"
"""Environment variable with the directory of the disk cache."""

LOCATION_PATTERN = re.compile(r"^#loc.*$|\s*loc\((?:[^()]|\([^()]*\))*\)", re.MULTILINE)
"""Source locations in StableHLO text, which differ between traces of the same function."""

LOWERED_MAGIC = b"CYXLA-LOWERED-3\n"
"""First bytes of a file of lowered programs."""


class CompileStatistics(NamedTuple):
    """Time and counts of compiling programs."""

    compile_seconds: float = 0.0
    """Seconds spent compiling StableHLO with the PJRT plugin."""
    load_seconds: float = 0.0
    """Seconds spent reading executables from the disk cache."""
    compiled: int = 0
    """Programs compiled."""
    loaded: int = 0
    """Programs whose executables came from the disk cache."""

    def add(self, other: "CompileStatistics") -> "CompileStatistics":
        """Returns the sum of two statistics."""
        return CompileStatistics(
            *(first + second for first, second in zip(self, other, strict=True))
        )


@functools.cache
def compile_options() -> bytes:
    """Returns the serialized `CompileOptionsProto` for one replica and one partition."""
    from jax._src import compiler

    options = compiler.get_compile_options(num_replicas=1, num_partitions=1)
    return options.SerializeAsString()


def distribution_identity(name: str) -> bytes:
    """Returns what identifies the code of an installed distribution.

    A distribution installed from an index is identified by its version. One installed from
    a directory or a repository can change without a new version, so the contents of its
    Python files are hashed as well.
    """
    distribution = importlib.metadata.distribution(name)
    digest = hashlib.sha256(f"{name}={distribution.version}\0".encode())
    if distribution.read_text("direct_url.json") is not None:
        for file in sorted(distribution.files or (), key=str):
            if file.suffix == ".py":
                path = Path(str(distribution.locate_file(file)))
                if path.exists():
                    digest.update(path.read_bytes())
    return digest.digest()


@functools.cache
def library_identity(distributions: tuple[str, ...] = ()) -> bytes:
    """Returns what identifies the libraries that lowered programs depend on.

    It always covers `jax`, `jaxlib` and `cyxla`, and also the given distributions.
    """
    digest = hashlib.sha256()
    for name in ("jax", "jaxlib", "cyxla", *distributions):
        digest.update(distribution_identity(name))
    return digest.digest()


def cache_key(*parts: str | bytes, distributions: tuple[str, ...] = ()) -> str:
    """Returns a cache key of parts that determine a result, and of the libraries.

    `distributions` names the distributions besides JAX and cyxla whose code the result
    depends on. A new version of any of them changes the key.
    """
    digest = hashlib.sha256(library_identity(distributions))
    for part in parts:
        data = part.encode() if isinstance(part, str) else part
        digest.update(struct.pack("<Q", len(data)) + data)
    return digest.hexdigest()


def cache_directory() -> Path:
    """Returns the directory of the disk cache, creating it if needed."""
    value = os.environ.get(CACHE_ENVIRONMENT_VARIABLE)
    directory = Path(value) if value else Path.home() / ".cache" / "cyxla"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_atomically(path: Path, data: bytes) -> None:
    """Writes a file that appears under its name only when complete."""
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def encode_lowered(programs: Sequence[LoweredProgram | None]) -> bytes:
    """Returns lowered programs as bytes, with `None` for a program that is left out."""
    parts = [LOWERED_MAGIC, struct.pack("<Q", len(programs))]
    for program in programs:
        if program is None:
            parts.append(struct.pack("<B", 0))
            continue
        text = program.text.encode()
        metadata = encode_metadata(program)
        parts.append(
            struct.pack("<BQQQ", 1, len(program.code), len(text), len(metadata))
        )
        parts += [program.code, text, metadata]
    return b"".join(parts)


def encode_metadata(program: LoweredProgram) -> bytes:
    """Returns the platform and argument types of a program as JSON.

    Either is `null` when it is not known.
    """
    types = program.input_types
    return json.dumps(
        {
            "platform": program.platform,
            "input_types": None
            if types is None
            else [[dtype.name, list(shape)] for dtype, shape in types],
        }
    ).encode()


def decode_metadata(data: bytes) -> tuple[tuple[ArrayType, ...] | None, str | None]:
    """Returns the argument types and the platform that `encode_metadata` wrote.

    Raises a `ValueError` if the JSON is malformed, and a `TypeError` for an unknown dtype.
    """
    metadata = json.loads(data)
    entries = metadata["input_types"]
    types = (
        None
        if entries is None
        else tuple(ArrayType(np.dtype(name), tuple(shape)) for name, shape in entries)
    )
    return types, metadata["platform"]


def decode_lowered(data: bytes) -> tuple[LoweredProgram | None, ...]:
    """Returns lowered programs from bytes, raising a `ValueError` if they are malformed."""
    if not data.startswith(LOWERED_MAGIC):
        raise ValueError("Not a file of lowered programs.")
    offset = len(LOWERED_MAGIC)
    programs: list[LoweredProgram | None] = []
    try:
        (count,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        for _ in range(count):
            (is_present,) = struct.unpack_from("<B", data, offset)
            offset += 1
            if not is_present:
                programs.append(None)
                continue
            code_size, text_size, types_size = struct.unpack_from("<QQQ", data, offset)
            offset += 24
            end = offset + code_size + text_size + types_size
            if end > len(data):
                raise struct.error("the programs run past the end of the file")
            code = data[offset : offset + code_size]
            text_end = offset + code_size + text_size
            text = data[offset + code_size : text_end].decode()
            types, platform = decode_metadata(data[text_end:end])
            offset = end
            programs.append(LoweredProgram(code, text, types, platform))
    except (
        struct.error,
        UnicodeDecodeError,
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        raise ValueError(f"Malformed file of lowered programs: {error}.") from error
    if offset != len(data):
        raise ValueError("Malformed file of lowered programs: trailing bytes.")
    return tuple(programs)


def load_lowered(key: str) -> tuple[LoweredProgram | None, ...] | None:
    """Returns the lowered programs stored under a key, or `None` if there are none.

    A malformed file is removed.
    """
    path = cache_directory() / f"{key}.lowered"
    if not path.exists():
        return None
    try:
        return decode_lowered(path.read_bytes())
    except ValueError:
        path.unlink(missing_ok=True)
        return None


def store_lowered(key: str, programs: Sequence[LoweredProgram | None]) -> None:
    """Stores lowered programs under a key."""
    write_atomically(cache_directory() / f"{key}.lowered", encode_lowered(programs))


def executable_key(program: LoweredProgram) -> str:
    """Returns the cache key of a program's executable for the loaded plugin."""
    _plugins.require_platform(program.platform)
    plugin = _plugins.loaded_plugin
    if plugin is None:
        raise RuntimeError("The loaded PJRT plugin is not recorded.")
    digest = hashlib.sha256(LOCATION_PATTERN.sub("", program.text).encode())
    digest.update(b"\0" + compile_options() + b"\0" + repr(plugin).encode())
    return digest.hexdigest()


def compile_program(
    program: LoweredProgram, disk_cache: bool = True
) -> tuple[Executable, CompileStatistics]:
    """Compiles a program, or loads its executable from the disk cache.

    The plugin for the platform of the program loads first, unless a plugin is loaded
    already. A cached executable that the plugin rejects is removed and compiled again. The
    executable takes the argument types of the program, so `Executable.run` converts its
    arguments to them.
    """
    _plugins.require_platform(program.platform)
    path = cache_directory() / f"{executable_key(program)}.pjrt" if disk_cache else None
    if path is not None and path.exists():
        start = time.perf_counter()
        try:
            executable = Executable.deserialize(path.read_bytes())
        except RuntimeError:
            path.unlink(missing_ok=True)
        else:
            executable.input_types = program.input_types
            return executable, CompileStatistics(
                load_seconds=time.perf_counter() - start, loaded=1
            )
    start = time.perf_counter()
    try:
        executable = Executable.compile(program.code, compile_options())
    except RuntimeError as error:
        if "No FFI handler registered" not in str(error):
            raise
        raise RuntimeError(
            f"{error}\nRegister the target with `cyxla.register_ffi_target` or "
            "`jax.ffi.register_ffi_target`."
        ) from error
    executable.input_types = program.input_types
    statistics = CompileStatistics(
        compile_seconds=time.perf_counter() - start, compiled=1
    )
    if path is not None:
        write_atomically(path, executable.serialize())
    return executable, statistics
