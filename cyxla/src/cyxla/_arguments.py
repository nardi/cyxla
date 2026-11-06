"""The rules for the arguments and results of an executable, in plain Python.

Everything here runs before and after the call into the runtime, so it needs no extension
module. `Executable.run` in `_runtime` applies these rules and then copies the arrays into
aligned buffers.
"""

from collections.abc import Collection, Sequence

import numpy as np
from numpy.typing import ArrayLike

from cyxla._types import ArrayType


def as_input(
    argument: ArrayLike,
    expected: ArrayType | None,
    index: int,
    supported: Collection[np.dtype],
) -> np.ndarray:
    """Returns an argument as a flat, contiguous array of the type that the program takes.

    Without a known type the argument keeps its dtype, which has to be one of `supported`.
    Otherwise it converts to the type of the program when no information is lost beyond
    rounding, so `float64` values pass for a `float32` argument, but `float64` values do not
    pass for an `int32` one.
    """
    array = np.ascontiguousarray(argument)
    if expected is None:
        if array.dtype not in supported:
            raise TypeError(
                f"Argument {index} has the dtype {array.dtype}, which cyxla does not "
                "support."
            )
        return array.ravel()
    if array.dtype != expected.dtype:
        lossless = array.dtype.kind in "biu" and expected.dtype.kind in "fc"
        if not (lossless or np.can_cast(array.dtype, expected.dtype, "same_kind")):
            raise TypeError(
                f"Argument {index} has the dtype {array.dtype}, which cannot convert to "
                f"the {expected.dtype} that the program takes."
            )
        array = array.astype(expected.dtype)
    if array.size != expected.size:
        raise ValueError(
            f"Argument {index} has {array.size} elements, but the program takes "
            f"{expected.size}."
        )
    return array.ravel()


def check_counts(
    input_count: int,
    output_count: int,
    expected: Sequence[ArrayType] | None,
    input_limit: int,
    output_limit: int,
) -> None:
    """Raises a `ValueError` for more arguments or outputs than the runtime takes.

    A program whose argument types are known also has to get as many arguments as it takes.
    """
    if input_count > input_limit or output_count > output_limit:
        raise ValueError(
            f"The runtime takes at most {input_limit} arguments and returns at most "
            f"{output_limit} outputs, not {input_count} and {output_count}."
        )
    if expected is not None and len(expected) != input_count:
        raise ValueError(
            f"The program takes {len(expected)} arguments, not {input_count}."
        )


def prepared_arguments(
    arguments: Sequence[ArrayLike],
    expected: Sequence[ArrayType] | None,
    supported: Collection[np.dtype],
) -> list[np.ndarray]:
    """Returns every argument as a flat array of the type that the program takes."""
    return [
        as_input(
            argument, None if expected is None else expected[index], index, supported
        )
        for index, argument in enumerate(arguments)
    ]


def empty_results(output_types: Sequence[ArrayType]) -> list[np.ndarray]:
    """Returns an uninitialized array for each output, which the runtime fills."""
    return [np.empty(output.shape, output.dtype) for output in output_types]


def zero_arguments(expected: Sequence[ArrayType] | None) -> list[np.ndarray]:
    """Returns zeros of the types that a program takes, to run it once.

    The types have to be known, as they are for an executable from `compile_program` or
    `load_program`.
    """
    if expected is None:
        raise ValueError("The types of the arguments are not known.")
    return [np.zeros(item.shape, item.dtype) for item in expected]
