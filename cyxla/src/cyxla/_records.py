"""Programs in bundles: the StableHLO of a program with the executables that were compiled from it.

A record keeps the StableHLO bytecode of a program, which any machine with the right plugin
can compile, and the serialized executable for each target that has one. Loading uses the
executable of the current target when there is one and compiles the bytecode otherwise.
"""

import time
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from cyxla import _targets
from cyxla._bundle import Bundle, BundleWriter
from cyxla._cache import CompileStatistics, compile_program
from cyxla._lowering import LoweredProgram
from cyxla._runtime import Executable
from cyxla._types import ArrayType


class TargetMismatchError(RuntimeError):
    """A bundle has no executable that this machine can load, and recompiling is off."""


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
    code_blob = f"{name}.code"
    executable_blob = f"{name}.executable"
    writer.add(code_blob, program.code)
    writer.add(executable_blob, executable.serialize())
    return {
        "name": name,
        "platform": program.platform,
        "input_types": [
            [item.dtype.name, list(item.shape)] for item in executable.input_types
        ],
        "custom_calls": list(_targets.custom_call_targets(program.text)),
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
