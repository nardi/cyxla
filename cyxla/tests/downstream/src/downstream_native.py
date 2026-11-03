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
