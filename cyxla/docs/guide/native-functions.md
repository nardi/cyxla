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
from cyxla cimport NativeArray, NativeAttribute, native_attribute_double, native_find_attribute, native_size, register_native

cdef const char* scale(void* user_data, int64_t argument_count,
                       const NativeArray* arguments, int64_t result_count,
                       NativeArray* results, int64_t attribute_count,
                       const NativeAttribute* attributes) noexcept nogil:
    cdef const double* source = <const double*> arguments[0].data
    cdef double* target = <double*> results[0].data
    cdef double factor = native_attribute_double(
        native_find_attribute(attributes, attribute_count, b"factor"))
    cdef int64_t index
    for index in range(native_size(&arguments[0])):
        target[index] = factor * source[index]
    return NULL

register_native(b"mykernels.scale", scale, NULL)
```

`native_size`, `native_find_attribute` and `native_attribute_double` are small inline
helpers for reading the arrays and attributes. The module `downstream_native` in
`tests/downstream` builds this example.

## State

`user_data` is the address of state that the function receives. It has to stay valid for
the life of the process. Values that belong to one call are attributes and are saved in
the program, and state that belongs to the function is created again when a process
imports the library that registers it.

## Limits

- Only dense arrays of the element types in `native.h` are passed, and a call has at
  most 32 arguments, 32 results and 16 attributes. Attributes are scalars or strings.
- The functions run on the CPU only.
- A native function has no derivative. Give it one with `jax.custom_jvp` or
  `jax.custom_vjp` around `native_call`.
