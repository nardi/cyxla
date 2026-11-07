# Native functions

A native function is a C function that a program calls by name. It receives the buffers
of the arguments and writes its results in place, so no data is copied and the GIL is
not acquired. Libraries that already have a C, C++, Rust or Cython kernel can make it
available to JAX programs this way without writing an FFI handler by hand.

## The interface

`cyxla/native.h` declares the interface, and `cyxla.get_include()` returns the directory
that contains it. A function has the type `CyxlaNativeFunction`:

```c
const char* scale(void* user_data, int64_t argument_count,
                  const CyxlaNativeArray* arguments, int64_t result_count,
                  CyxlaNativeArray* results, int64_t attribute_count,
                  const CyxlaNativeAttribute* attributes);
```

- An array has an element type, a rank, its dimensions and a pointer to dense, row-major
  data. The data of an argument is read-only. The data of a result has the size that the
  program declares.
- The attributes are the keyword arguments of the call that are numbers or strings.
- The function returns `NULL` on success, or an error message that the run raises.
- It is called on the threads of the runtime and has to be thread safe.

## Registration and calls

A function is registered under a name, and a program only contains that name. The same
name therefore has to refer to the same function in every process that runs the program.

```python
import ctypes

import cyxla

library = ctypes.CDLL("libmykernels.so")
cyxla.register_native_function(
    "mykernels.scale", ctypes.cast(library.scale, ctypes.c_void_p).value
)


def scaled(values):
    return cyxla.native_call(
        "mykernels.scale",
        jax.ShapeDtypeStruct(values.shape, values.dtype),
        values,
        factor=3.0,
    )
```

`native_call` is `jax.ffi.ffi_call` for the handler of cyxla, with the name as its first
argument, the result structure as its second, and the keyword arguments as attributes. A
function is registered by address, so any language that can export a C function is
supported. A function that was registered once cannot be replaced, and it has to be
registered before the programs that call it run.

`cyxla.lower` allows the target of this handler without an entry in `custom_calls`. The
same call also works in plain `jax.jit`, where `jaxlib` runs the handler, so one
function serves both cases.

## From Cython

A Cython module registers a function with `register_native`, which checks the type of
the function at compile time. A `cdef` function, or a `@cython.cfunc` one in pure Python
mode, is not a Python object, so it can only be passed this way. The registration runs
when the module is imported.

```cython
from libc.stdint cimport int64_t

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
    # Multiplies the one argument by the attribute factor and adds the double that the
    # user data points to, if there is one.
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
```

`native_size`, `native_find_attribute` and `native_attribute_double` are small inline
helpers for reading the arrays and attributes. The module `downstream_native_pyx` in
`tests/downstream` builds this example, and `downstream_native` has the same function in
pure Python mode.

## Derivatives

A native function has no derivative of its own, and there are three ways to give it one,
all tested in `tests/test_native_primitives.py`.

`jax.custom_vjp` is the simplest option, because the forward pass and the backward pass
are both calls of native functions and reverse mode needs no transpose rule.

```python
@jax.custom_vjp
def square(values):
    return cyxla.native_call("mykernels.square", spec(values), values)


def forward(values):
    return square(values), values


def backward(values, cotangent):
    gradient = cyxla.native_call(
        "mykernels.square_backward", spec(values), values, cotangent
    )
    return (gradient,)


square.defvjp(forward, backward)
```

`jax.custom_jvp` supports forward mode, and also reverse mode when the tangent is
computed with operations that JAX can transpose, such as `jnp` multiplications.

A primitive from [`native_primitive`][cyxla.native_primitive] supports `jit`, `vmap`,
`grad`, `jvp` and `linearize` at once. A kernel is any object with the members of the
protocol [`NativeKernel`][cyxla.NativeKernel], and a Cython `cdef class` is a good fit,
because it can own the state that `user_data_address` points to.

```cython
cdef class Scale:
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

    def result_avals(self, values, *, factor):
        return (values,)

    def jvp(self, primals, tangents, *, factor):
        return [scale(primals[0], factor=factor)], [factor * tangents[0]]

    def batch(self, arguments, dimensions, *, factor):
        return [scale(arguments[0], factor=factor)], [dimensions[0]]


scale = cyxla.native_primitive(Scale(10.0))
```

The rules are optional methods, where `jvp(primals, tangents, **params)` returns the
primal results and the tangents of the results,
`transpose(cotangents, *arguments, **params)` returns a cotangent for each argument, and
`batch(arguments, dimensions, **params)` returns the results and their batch dimensions.
For a linear function, `linear = True` defines the JVP as the function applied to the
tangents, so only `transpose` has to be written. A function that is not a class can use
[`NativeFunctionKernel`][cyxla.NativeFunctionKernel]. The params of a call, such as
`factor`, are keyword arguments and are passed to the function as attributes.

A program that calls a primitive only contains the call of the native function, so it
can be stored and loaded like any program with native calls.

## State

`user_data` is the address of state that is passed to the function. It has to stay valid
for the life of the process. Values that are specific to one call are attributes and are
saved in the program. State that is specific to the function is not saved, and is
created again when a process imports the library that registers it.

## Saved programs

A program that calls a native function can be stored with `pack_program`. The record
contains the name of each function and the package and version that registered it, as
for the targets of other libraries. The library that registers the function has to be
imported before the program is loaded. This can be checked with
[`check_targets`][cyxla.check_targets], which names each missing function with its
provider, and `import_providers=True` imports the providers itself. The library
registers its functions when it is imported, so the same name means the same function in
both processes.

## Limits

- A call takes only dense arrays of the element types in `native.h`, at most 32
  arguments, 32 results and 16 attributes, and attributes are scalars or strings.
- The functions run on the CPU only.
