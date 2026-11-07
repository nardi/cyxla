# Custom calls

Some JAX functions do not lower to plain StableHLO operations. They lower to a custom
call, which refers by name to a native function that the program calls. On the CPU,
`jnp.linalg` and `jax.scipy.linalg` work this way with the LAPACK kernels that `jaxlib`
provides, and libraries that define their own primitives, such as a solver, use custom
calls for their own functions in the same way.

A PJRT plugin is a separate build of XLA, and none of these functions are registered in
it. A program only compiles once each of its targets is registered with the plugin by
name. cyxla registers the kernels of `jaxlib` and all other handlers that libraries
register with JAX.

## Linear algebra

Loading the plugin will register the typed FFI handlers of `jaxlib` with it and load the
LAPACK functions of SciPy that they call. Programs with `cholesky`, `solve`, `inv`,
`det`, `slogdet`, `qr`, `eigh`, `svd`, `lstsq`, `pinv`, `eigvals`, `lu`,
`solve_triangular`, `expm`, `schur` and the other functions of `jnp.linalg` and
`jax.scipy.linalg` compile and run as they do in JAX, with the same results.

<!-- name: test_docs_custom_calls -->
```python
import jax
import jax.numpy as jnp
import numpy as np

import cyxla as cx

matrix = np.array([[4.0, 1.0], [1.0, 3.0]])


def solve(coefficients):
    return jnp.linalg.solve(coefficients, jnp.array([1.0, 2.0]))


program = cx.lower(solve, [jax.ShapeDtypeStruct((2, 2), jnp.float32)])
executable, _ = cx.compile_program(program)
(solution,) = executable.run([matrix])
assert np.allclose(matrix @ solution, [1.0, 2.0])
```

The eigenvalues of `eig` and `eigvals` are returned as complex arrays.

`initialize(jax_targets=False)` registers none of these handlers, and handlers of the
older untyped interface are never registered, because the plugin cannot run them.

The registered handlers are the ones in the `registrations` functions of
`jaxlib.lapack`, `jaxlib.cpu_sparse` and, for GPUs, the `gpu_solver`, `gpu_sparse`,
`gpu_linalg` and `gpu_prng` modules of `jaxlib`. The GPU kernels are in
`jax-cuda12-plugin` and the corresponding packages for other CUDA and ROCm versions,
which the CUDA extras install. They are not tested, because CI has no GPU.

## Handlers of other libraries

A library with its own primitives implements a typed FFI handler in C++ and registers it
with `jax.ffi.register_ffi_target`. That call registers the handler with JAX, and cyxla
registers it with the plugin as well. Importing cyxla replaces
`register_custom_call_target` of `jaxlib` with a function that also registers each
handler with the plugin. Handlers that were registered before the import are read from
the registry of JAX when the plugin loads. The order of the imports therefore does not
matter.

A library that does not use JAX can call `register_ffi_target` of cyxla, which takes the
arguments of the JAX function. The handler is a `PyCapsule`, as for JAX, or the address
as an integer.

```python
import ctypes

import jax

import cyxla as cx

library = ctypes.CDLL("libmy_kernels.so")
capsule = jax.ffi.pycapsule(library.my_handler)
jax.ffi.register_ffi_target("my_target", capsule, platform="cpu")
cx.register_ffi_target("my_target", capsule, platform="cpu")
```

With `CYXLA_MIRROR_JAX=0`, the function of `jaxlib` is not replaced. Handlers are then
only registered with the plugin through `register_ffi_target` and from the registry of
JAX when the plugin loads, and `mirror_jax_registrations` replaces the function later.

Handlers of the untyped interface are not registered with the plugin. Each registration
is for one platform, and registrations for a platform other than the one of the plugin
are kept but not used. A handler with stages other than `execute`, such as an
`instantiate` stage for state, is skipped with a warning, because the PJRT interface
accepts one function per handler.

## Custom calls in lowered functions

`jax.export` raises an error for custom calls without a compatibility guarantee, which
includes every target of a third-party library. Lowering succeeds when the targets that
the function may call are passed in `custom_calls`.

```python
def scaled(values):
    spec = jax.ShapeDtypeStruct(values.shape, values.dtype)
    call = jax.ffi.ffi_call("my_target", spec)
    return call(values, factor=3.0)


program = cx.lower(scaled, shapes, custom_calls=["my_target"])
```

A differentiated function can call targets that the original does not, such as the
transposed solve of a linear solver. `custom_calls="detect"` lowers the function once
more, reads the targets from the result and allows all of them, so the list always
matches the function.

```python
program = cx.lower(scaled, shapes, custom_calls="detect")
```

The compiled executable refers to the handler by name, so the cached executable remains
valid when a new version of a library changes its handler. The key of a lowered program
only changes when the caller adds the library to the distributions of `cache_key`.

## Python callbacks

`jax.debug.callback`, `jax.pure_callback` and `jax.experimental.io_callback` call back
into Python. JAX cannot export such a function, so `lower` raises a `HostCallbackError`
that lists each callback with the file and line where it was created. The error checks
of libraries such as Equinox contain callbacks. Equinox omits them if `EQX_ON_ERROR=nan`
is set before it is imported.

`lower(..., host_callbacks=True)` runs the callbacks instead, because cyxla replaces
each callback in the jaxpr of the function by a call of the built-in native function
`cyxla.python_callback`, and the number of the callback is passed in the attribute
`index`. cyxla cannot use the handler of `jaxlib`, because it reads the callbacks from
an execution context that is only populated by the client of `jaxlib`. Calling it
through the PJRT C API crashes the process, so cyxla leaves it unregistered. The native
function takes the GIL and calls the callback, so it uses the same handler as other
[native functions](native-functions.md).

```python
def checked(values):
    def fail(value):
        raise ValueError("negative input")

    spec = jax.ShapeDtypeStruct(values.shape, values.dtype)
    return jax.lax.cond(
        jnp.any(values < 0),
        lambda operand: jax.pure_callback(fail, spec, operand),
        lambda operand: operand,
        values,
    )


program = cx.lower(checked, shapes, host_callbacks=True)
```

An exception in a callback ends the execution with a `RuntimeError` that names the
exception, and the executable remains usable.

- The native function takes the GIL for each callback. `Executable.run` and the
  extension modules of cyxla release it while a program runs, and code that calls
  `execute` with the GIL held would deadlock.
- A program refers to its callbacks by numbers that are only valid in the process that
  lowered it. It cannot be stored, serialized or cached on disk, and the functions that
  would do so raise an error. The callbacks remain registered as long as the program or
  an executable compiled from it exists.
- A callback takes at most 32 arguments and returns at most 32 results, which is the
  limit of native functions.
- Callbacks run on the CPU only, and ordered callbacks (`ordered=True`) are not
  supported, because the runtime cannot pass their tokens.
- Lowering rewrites the jaxpr of the function, which replaces every callback by a call
  of the native function. It finds the callbacks by the name of their primitive and runs
  the `callback` param of each equation, so this depends on names that a future version
  of JAX may change. The tests of cyxla would fail in that case.

## Unsupported calls

- Handlers of the untyped interface (`api_version=0`) fail to compile with an error
  about the API version.
- A program with an unregistered target fails to compile with "No FFI handler registered
  for" and the target name, followed by a reminder to register it.
