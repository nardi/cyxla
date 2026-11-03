"""A native function written in Cython that programs of cyxla call by name without the GIL."""

import cython
from cython.cimports.cyxla import (
    NativeArray,
    NativeAttribute,
    native_attribute_double,
    native_find_attribute,
    native_size,
    register_native,
)
from cython.cimports.libc.stdint import int64_t


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def scale(
    user_data: cython.p_void,
    argument_count: int64_t,
    arguments: cython.pointer[cython.const[NativeArray]],
    result_count: int64_t,
    results: cython.pointer[NativeArray],
    attribute_count: int64_t,
    attributes: cython.pointer[cython.const[NativeAttribute]],
) -> cython.pointer[cython.const[cython.char]]:
    """Multiplies the one argument by the attribute `factor`, which is a float64."""
    source: cython.pointer[cython.const[cython.double]] = cython.cast(
        cython.pointer[cython.const[cython.double]], arguments[0].data
    )
    target: cython.p_double = cython.cast(cython.p_double, results[0].data)
    factor: cython.double = native_attribute_double(
        native_find_attribute(attributes, attribute_count, b"factor")
    )
    index: int64_t
    for index in range(native_size(arguments)):
        target[index] = factor * source[index]
    return cython.NULL


register_native(b"downstream.scale", scale, cython.NULL)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def scale_and_shift(
    user_data: cython.p_void,
    argument_count: int64_t,
    arguments: cython.pointer[cython.const[NativeArray]],
    result_count: int64_t,
    results: cython.pointer[NativeArray],
    attribute_count: int64_t,
    attributes: cython.pointer[cython.const[NativeAttribute]],
) -> cython.pointer[cython.const[cython.char]]:
    """Multiplies the argument by the attribute `factor` and adds the state, a double."""
    source: cython.pointer[cython.const[cython.double]] = cython.cast(
        cython.pointer[cython.const[cython.double]], arguments[0].data
    )
    target: cython.p_double = cython.cast(cython.p_double, results[0].data)
    factor: cython.double = native_attribute_double(
        native_find_attribute(attributes, attribute_count, b"factor")
    )
    shift: cython.double = cython.cast(cython.p_double, user_data)[0]
    index: int64_t
    for index in range(native_size(arguments)):
        target[index] = factor * source[index] + shift
    return cython.NULL


@cython.cclass
class ScaleAndShift:
    """A kernel for `native_primitive` that owns the state that its function reads."""

    shift: cython.double
    name = "downstream.scale_and_shift"

    def __init__(self, shift: cython.double) -> None:
        self.shift = shift

    @property
    def function_address(self) -> cython.size_t:
        """The address of the C function."""
        return cython.cast(cython.size_t, scale_and_shift)

    @property
    def user_data_address(self) -> cython.size_t:
        """The address of the state, which lives as long as the kernel."""
        return cython.cast(cython.size_t, cython.address(self.shift))

    def result_avals(self, x, *, factor):
        """The result has the type of the argument."""
        return (x,)
