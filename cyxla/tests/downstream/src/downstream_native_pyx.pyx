"""A native function and a kernel in the classic Cython syntax, as a downstream project writes them."""

from libc.stdint cimport int64_t

import cyxla
import jax.numpy as jnp
from cyxla cimport (
    NativeArray,
    NativeAttribute,
    native_attribute_double,
    native_find_attribute,
    native_size,
    register_native,
)


cdef const char* scale_function(
    void* user_data,
    int64_t argument_count,
    const NativeArray* arguments,
    int64_t result_count,
    NativeArray* results,
    int64_t attribute_count,
    const NativeAttribute* attributes,
) noexcept nogil:
    # Multiplies the one argument by the attribute factor and adds the double that the user
    # data points to, if there is one.
    cdef const double* source = <const double*> arguments[0].data
    cdef double* target = <double*> results[0].data
    cdef double factor = native_attribute_double(
        native_find_attribute(attributes, attribute_count, b"factor")
    )
    cdef double offset = 0.0
    if user_data != NULL:
        offset = (<const double*> user_data)[0]
    cdef int64_t index
    for index in range(native_size(&arguments[0])):
        target[index] = factor * source[index] + offset
    return NULL


register_native(b"downstream.scale_pyx", scale_function, NULL)


cdef class Scale:
    """A kernel for `native_primitive` that owns the offset that its function adds."""

    cdef double offset
    name = "downstream.scale_offset_pyx"

    def __cinit__(self, double offset=0.0):
        self.offset = offset

    @property
    def function_address(self):
        return <size_t> scale_function

    @property
    def user_data_address(self):
        return <size_t> &self.offset

    def result_avals(self, x, *, factor):
        return (x,)

    def jvp(self, primals, tangents, *, factor):
        # The function is affine in its argument, so the tangent only scales.
        return [scale(primals[0], factor=factor)], [factor * tangents[0]]

    def batch(self, arguments, dimensions, *, factor):
        return [scale(arguments[0], factor=factor)], [dimensions[0]]


scale = cyxla.native_primitive(Scale(10.0))
