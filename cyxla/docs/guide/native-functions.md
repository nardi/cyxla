# Native functions

A native function is a C function that a program calls by name. It receives the buffers
of the arguments and writes its results in place, so there is no copy and no GIL.
Libraries that already have a C, C++, Rust or Cython kernel can offer it to JAX programs
this way without writing an FFI handler by hand.

## The interface

`cyxla/native.h` declares the interface, and `cyxla.get_include()` returns the directory
that holds it. A function has the type `CyxlaNativeFunction`:

```c
const char* scale(void* user_data, int64_t argument_count,
                  const CyxlaNativeArray* arguments, int64_t result_count,
                  CyxlaNativeArray* results, int64_t attribute_count,
                  const CyxlaNativeAttribute* attributes);
```

- An array has an element type, a rank, its dimensions and a pointer to dense, row-major
  data. The data of an argument is read-only. The data of a result has the size that the
  program declared.
- The attributes are the keyword arguments of the call that are numbers or strings.
- The function returns `NULL`, or a message that becomes the error of the run.
- It runs on the threads of the runtime and has to be thread safe.

## Registration and calls

A function is registered under a name. The name is what a program holds, so the same
name has to mean the same function in every process that runs the program.

```python
import ctypes

import cyxla

library = ctypes.CDLL("libmykernels.so")
cyxla.register_native_function(
    "mykernels.scale", ctypes.cast(library.scale, ctypes.c_void_p).value
)


def scaled(x):
    return cyxla.native_call(
        "mykernels.scale", jax.ShapeDtypeStruct(x.shape, x.dtype), x, factor=3.0
    )
```

`native_call` is `jax.ffi.ffi_call` for the handler of cyxla. Its first argument is the
name, its second the result structure, and the keyword arguments become attributes.
Registration is by address, so it works for any language that can export a C function. A
function that was registered once cannot be replaced, and it has to be registered before
the programs that call it run.

`cyxla.lower` allows the target of the handler on its own. The same call also works in
plain `jax.jit`, where jaxlib runs the handler, so a library offers one function for
both.

## From Cython

A Cython module registers a function with `register_native`, which checks the type of
the function when the module compiles. A `cdef` function, or a `@cython.cfunc` one in
pure Python mode, is not a Python object, so this is the way to pass it. The module's
code runs when it is imported.

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
```

`native_size`, `native_find_attribute` and `native_attribute_double` are small inline
helpers for reading the arrays and attributes. The module `downstream_native_pyx` in
`tests/downstream` builds this example, and `downstream_native` has the same function in
pure Python mode.

## Derivatives

A native function has no derivative of its own. There are three ways to give it one. All
of them are tested in `tests/test_native_primitives.py`.

**`jax.custom_vjp`** is the simplest. The forward pass and the backward pass are both
calls of native functions, and reverse mode needs no transpose rule.

```python
@jax.custom_vjp
def square(x):
    return cyxla.native_call("mykernels.square", spec(x), x)


def forward(x):
    return square(x), x


def backward(x, cotangent):
    return (cyxla.native_call("mykernels.square_backward", spec(x), x, cotangent),)


square.defvjp(forward, backward)
```

**`jax.custom_jvp`** gives forward mode, and reverse mode when the tangent is computed
with operations that JAX can transpose, such as `jnp` multiplications.

**A primitive** from [`native_primitive`][cyxla.native_primitive] gives the function a
place in `jit`, `vmap`, `grad`, `jvp` and `linearize` at once. A kernel is any object
with the members of the protocol [`NativeKernel`][cyxla.NativeKernel], and a Cython
`cdef class` is the natural one, because it can own the state that `user_data_address`
points to.

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

    def result_avals(self, x, *, factor):
        return (x,)

    def jvp(self, primals, tangents, *, factor):
        return [scale(primals[0], factor=factor)], [factor * tangents[0]]

    def batch(self, arguments, dimensions, *, factor):
        return [scale(arguments[0], factor=factor)], [dimensions[0]]


scale = cyxla.native_primitive(Scale(10.0))
```

The rules are optional methods. `jvp(primals, tangents, **params)` returns the primal
results and the tangents of the results. `transpose(cotangents, *arguments, **params)`
returns a cotangent for each argument, and `batch(arguments, dimensions, **params)`
returns the results and their batch dimensions. `linear = True` gives a linear function
the rule of differentiation of applying itself to the tangents, so only `transpose` is
left to write. A function that is not a class can use
[`NativeFunctionKernel`][cyxla.NativeFunctionKernel]. The params of a call, such as
`factor`, are keyword arguments and become the attributes that the function reads.

A program that calls a primitive holds the call of the native function and nothing else,
so it can be stored and loaded like any program with native calls.

## State

`user_data` is the address of state that the function receives. It has to stay valid for
the life of the process. Values that belong to one call are attributes and are saved in
the program, and state that belongs to the function is created again when a process
imports the library that registers it.

## Saved programs

A program that calls a native function can be stored with `pack_program`. The record
holds the name of each function and the package and version that registered it, as for
the targets of other libraries. The process that loads the program has to import the
library that registers the function first. Loading checks this with
[`check_targets`][cyxla.check_targets], which names each missing function with its
provider, and `import_providers=True` imports the providers itself. The library
registers its functions when it is imported, so the same name means the same function in
both processes.

## Limits

- Only dense arrays of the element types in `native.h` are passed, and a call has at
  most 32 arguments, 32 results and 16 attributes. Attributes are scalars or strings.
- The functions run on the CPU only.
