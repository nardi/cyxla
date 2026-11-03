"""Programs in bundles: the StableHLO of a program with the executables that were compiled from it.

A record keeps the StableHLO bytecode of a program, which any machine with the right plugin
can compile, and the serialized executable for each target that has one. Loading uses the
executable of the current target when there is one and compiles the bytecode otherwise.
"""

import importlib
import importlib.metadata
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, NamedTuple

import numpy as np

from cyxla import _ffi, _native_functions, _targets
from cyxla._bundle import Bundle, BundleWriter
from cyxla._cache import CompileStatistics, compile_program
from cyxla._lowering import LoweredProgram
from cyxla._runtime import Executable
from cyxla._types import ArrayType, refuse_callbacks


class TargetMismatchError(RuntimeError):
    """A bundle has no executable that this machine can load, and recompiling is off."""


class MissingTargetError(RuntimeError):
    """A program calls an FFI target that no library has registered in this process."""


class MissingTarget(NamedTuple):
    """An FFI target or native function that a stored program needs and this process lacks."""

    name: str
    """The name of the custom call target."""
    provider: str | None
    """The package that registered the target when the program was stored, if known."""
    version: str | None
    """The version of the distribution of the provider at that time, if known."""
    kind: str = "FFI target"
    """Either `FFI target` or `native function`."""


def distribution_version(package: str | None) -> str | None:
    """Returns the version of the distribution that provides a top-level package, if any."""
    if package is None:
        return None
    for distribution in importlib.metadata.packages_distributions().get(package, ()):
        return importlib.metadata.version(distribution)
    return None


def registered_targets(
    names: Sequence[str], platform: str | None
) -> dict[str, dict[str, str | None]]:
    """Returns the provider and its version for each target that was registered with cyxla.

    A target that the plugin has built in is not in the registry. Nothing has to register it
    when a program loads, so it is left out.
    """
    if platform is None:
        return {}
    found: dict[str, dict[str, str | None]] = {}
    for name in names:
        # Native functions are recorded by their own names, which tell more than the target
        # of the handler that calls them.
        if name == _targets.NATIVE_TARGET:
            continue
        target = _ffi.targets.get((_ffi.platform_key(platform), name))
        if target is not None:
            found[name] = {
                "provider": target.provider,
                "version": distribution_version(target.provider),
            }
    return found


def native_function_origins(text: str) -> dict[str, dict[str, str | None]]:
    """Returns the provider and its version for each native function that a module calls."""
    found: dict[str, dict[str, str | None]] = {}
    for name in _targets.native_function_names(text):
        info = _native_functions.native_functions.get(name)
        provider = None if info is None else info.provider
        found[name] = {"provider": provider, "version": distribution_version(provider)}
    return found


def pack_program(
    writer: BundleWriter, name: str, executable: Executable
) -> dict[str, Any]:
    """Adds an executable and its source to a bundle and returns the record for the header.

    The executable has to come from `compile_program` or `load_program`, so that its
    StableHLO is known. The record goes into the header of the bundle, and `load_program`
    reads it from there.
    """
    program = executable.program
    if program is None or executable.input_types is None:
        raise ValueError(
            "The executable has no lowered program, so it cannot be stored."
        )
    refuse_callbacks(program, "stored")
    code_blob = f"{name}.code"
    executable_blob = f"{name}.executable"
    writer.add(code_blob, program.code)
    writer.add(executable_blob, executable.serialize())
    custom_calls = _targets.custom_call_targets(program.text)
    return {
        "name": name,
        "platform": program.platform,
        "input_types": [
            [item.dtype.name, list(item.shape)] for item in executable.input_types
        ],
        "custom_calls": list(custom_calls),
        "registered_targets": registered_targets(custom_calls, program.platform),
        "native_functions": native_function_origins(program.text),
        "code": code_blob,
        "executables": {_targets.target_fingerprint().key: executable_blob},
    }


def record_custom_calls(records: list[Mapping[str, Any]]) -> tuple[str, ...]:
    """Returns the custom call targets that any of the records needs, sorted.

    A process registers their handlers before it loads the programs.
    """
    return tuple(
        sorted({target for record in records for target in record["custom_calls"]})
    )


def missing_targets(records: Sequence[Mapping[str, Any]]) -> tuple[MissingTarget, ...]:
    """Returns the targets and native functions that records need which are not registered.

    Only targets that were registered with cyxla when the program was stored count, since
    the plugin has the others built in. Every native function that a program calls counts.
    Call it after `initialize`, which registers the handlers of jaxlib, and after the
    libraries of the programs are imported. A record that an older version of cyxla wrote
    has none of them and needs none.
    """
    found: dict[tuple[str, str], MissingTarget] = {}
    for record in records:
        platform = _ffi.platform_key(record["platform"])
        for name, origin in record.get("registered_targets", {}).items():
            if (platform, name) not in _ffi.applied:
                found.setdefault(
                    ("FFI target", name),
                    MissingTarget(name, origin["provider"], origin["version"]),
                )
        for name, origin in record.get("native_functions", {}).items():
            if name not in _native_functions.native_functions:
                found.setdefault(
                    ("native function", name),
                    MissingTarget(
                        name, origin["provider"], origin["version"], "native function"
                    ),
                )
    return tuple(sorted(found.values(), key=lambda target: (target.kind, target.name)))


def check_targets(
    records: Sequence[Mapping[str, Any]], import_providers: bool = False
) -> None:
    """Raises a `MissingTargetError` unless the targets and functions of records are registered.

    The error names each one with the package and version that provided it. With
    `import_providers` the missing ones are imported first, which makes the library register
    its handlers. A name in a file is code to run, so this stays off unless the caller
    trusts the file.
    """
    missing = missing_targets(records)
    if missing and import_providers:
        for package in sorted(
            {target.provider for target in missing if target.provider}
        ):
            try:
                importlib.import_module(package)
            except ImportError as error:
                raise MissingTargetError(
                    f"The package {package}, which provides FFI targets, cannot be "
                    f"imported: {error}"
                ) from error
        missing = missing_targets(records)
    if not missing:
        return
    lines = [
        "The programs need FFI targets or native functions that nothing has registered:"
    ]
    for target in missing:
        match target.provider, target.version:
            case None, _:
                origin = "the provider is not known"
            case provider, None:
                origin = f"provided by {provider}"
            case _:
                origin = f"provided by {target.provider} {target.version}"
        lines.append(f"  {target.name} ({target.kind}, {origin})")
    lines.append("Import the library that provides them before loading the programs.")
    raise MissingTargetError("\n".join(lines))


def load_program(
    bundle: Bundle,
    record: Mapping[str, Any],
    on_mismatch: Literal["recompile", "raise"] = "recompile",
) -> tuple[Executable, CompileStatistics]:
    """Loads the executable of a record, and returns it with the cost of loading it.

    The executable of the current target is deserialized. If the record has none, or the
    plugin rejects it, the StableHLO is compiled instead, unless `on_mismatch` is `raise`.
    The executable has its argument types and its `program` set, as if it came from
    `compile_program`.
    """
    input_types = tuple(
        ArrayType(np.dtype(dtype), tuple(shape))
        for dtype, shape in record["input_types"]
    )
    blob = record["executables"].get(_targets.target_fingerprint().key)
    if blob is not None:
        start = time.perf_counter()
        try:
            executable = Executable.deserialize(bundle.blob(blob))
        except RuntimeError as error:
            failure = str(error)
        else:
            executable.input_types = input_types
            executable.program = LoweredProgram(
                bytes(bundle.blob(record["code"])), "", input_types, record["platform"]
            )
            return executable, CompileStatistics(
                load_seconds=time.perf_counter() - start, loaded=1
            )
    else:
        failure = "the bundle has no executable for this machine"
    if on_mismatch == "raise":
        raise TargetMismatchError(
            f"The program {record['name']} cannot be loaded: {failure}."
        )
    program = LoweredProgram(
        bytes(bundle.blob(record["code"])), "", input_types, record["platform"]
    )
    return compile_program(program, disk_cache=False)
