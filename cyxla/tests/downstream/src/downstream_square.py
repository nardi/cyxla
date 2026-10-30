"""Runs a cyxla program of one `float32` input and one output without the GIL."""

import cython
from cython.cimports.cyxla import (
    ABI_VERSION,
    MAX_OUTPUTS,
    Executable,
    Input,
    PJRT_Buffer_Type_F32,
    ProgramSpec,
    alias_buffer,
    aligned_memory,
    destroy_buffer,
    execute,
    free_aligned,
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
    spec.output_bytes[0] = size * cython.sizeof(cython.float)
    inputs: Input[1]
    inputs[0].type = PJRT_Buffer_Type_F32
    inputs[0].size = size
    inputs[0].host = aligned_memory(spec.output_bytes[0])
    inputs[0].buffer = (
        alias_buffer(inputs[0].host, PJRT_Buffer_Type_F32, size)
        if zero_copy
        else cython.NULL
    )
    host: cython.p_float = cython.cast(cython.p_float, inputs[0].host)
    index: cython.int
    for index in range(size):
        host[index] = values[index]
    result: cython.p_float = cython.cast(
        cython.p_float, aligned_memory(spec.output_bytes[0])
    )
    outputs: cython.p_void[MAX_OUTPUTS]
    outputs[0] = result
    status: cython.int
    with cython.nogil:
        status = execute(cython.address(spec), inputs, outputs, zero_copy)
    answer = [result[index] for index in range(size)]
    destroy_buffer(inputs[0].buffer)
    free_aligned(inputs[0].host)
    free_aligned(result)
    if status != 0:
        raise RuntimeError("The program failed.")
    return answer
