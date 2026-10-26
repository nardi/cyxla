"""Runs a cyxla program of one input and one output without the GIL."""

import cython
from cython.cimports.cyxla._runtime import (
    ABI_VERSION,
    MAX_OUTPUTS,
    Executable,
    Input,
    ProgramSpec,
    alias_buffer,
    aligned_doubles,
    destroy_buffer,
    execute,
    free_doubles,
)

import cyxla

cyxla.check_abi(ABI_VERSION)

COMPILED_ABI_VERSION = ABI_VERSION
"""The `ABI_VERSION` that this module was compiled against."""


def run(executable: Executable, values: list[float], zero_copy: bool) -> list[float]:
    """Runs a program whose output has the size of its input and returns the output."""
    size: cython.int = len(values)
    spec: ProgramSpec
    spec.handle = executable.handle
    spec.num_inputs = 1
    spec.num_outputs = 1
    spec.output_sizes[0] = size
    inputs: Input[1]
    inputs[0].size = size
    inputs[0].host = aligned_doubles(size)
    inputs[0].buffer = alias_buffer(inputs[0].host, size) if zero_copy else cython.NULL
    index: cython.int
    for index in range(size):
        inputs[0].host[index] = values[index]
    result: cython.p_double = aligned_doubles(size)
    outputs: cython.p_double[MAX_OUTPUTS]
    outputs[0] = result
    status: cython.int
    with cython.nogil:
        status = execute(cython.address(spec), inputs, outputs, zero_copy)
    answer = [result[index] for index in range(size)]
    destroy_buffer(inputs[0].buffer)
    free_doubles(inputs[0].host)
    free_doubles(result)
    if status != 0:
        raise RuntimeError("The program failed.")
    return answer
