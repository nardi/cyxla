# Custom calls

Some JAX functions do not lower to plain StableHLO operations. They lower to a custom
call, which names a native function that the program runs. On the CPU, `jnp.linalg` and
`jax.scipy.linalg` work this way with the LAPACK kernels that jaxlib ships, and
libraries that add primitives, such as a fast solver, do the same with their own
functions.

A PJRT plugin is a separate build of XLA, and it knows none of these functions. The
program compiles only after each target is registered with the plugin by name. cyxla
does that for the kernels of jaxlib, and it gives libraries two ways to do it
themselves.

## Linear algebra

`initialize` registers the typed FFI handlers of jaxlib with the plugin and loads the
LAPACK functions of SciPy that they call. Programs with `cholesky`, `solve`, `inv`,
`det`, `slogdet`, `qr`, `eigh`, `svd`, `lstsq`, `pinv`, `eigvals`, `lu`,
`solve_triangular`, `expm`, `schur` and the other functions of `jnp.linalg` and
`jax.scipy.linalg` compile and run like they do in JAX, with the same results.

<!-- name: test_docs_custom_calls -->
```python
import jax
import jax.numpy as jnp
import numpy as np

import cyxla as cx

cx.enable_x64()
cx.initialize()
matrix = np.array([[4.0, 1.0], [1.0, 3.0]])


def solve(a):
    return jnp.linalg.solve(a, jnp.array([1.0, 2.0]))


program = cx.lower(solve, [jax.ShapeDtypeStruct((2, 2), jnp.float64)])
executable, _ = cx.compile_program(program)
(solution,) = executable.run([matrix], output_sizes=[2])
assert np.allclose(matrix @ solution, [1.0, 2.0])
```

Pass `jax_targets=False` to `initialize` to register none of them. Handlers of the older
untyped interface are never registered, because the plugin cannot run them.

The registered handlers are the ones in the `registrations` functions of
`jaxlib.lapack`, `jaxlib.cpu_sparse` and, for GPUs, the `gpu_solver`, `gpu_sparse`,
`gpu_linalg` and `gpu_prng` modules of jaxlib. The GPU kernels ship in
`jax-cuda12-plugin` and its relatives, which the CUDA extras install. They are untested
here, because CI has no GPU.

## Handlers of other libraries

A library with its own primitives implements a typed FFI handler in C++ and registers it
with `jax.ffi.register_ffi_target`. That call tells JAX about the handler. cyxla needs
the same address, and there are two ways to give it.

`register_ffi_target` takes the arguments of the JAX function, so a library can call
both. The handler is a `PyCapsule`, as JAX takes it, or the address as an integer.

```python
import ctypes

import jax

import cyxla as cx

library = ctypes.CDLL("libmy_kernels.so")
capsule = jax.ffi.pycapsule(library.my_handler)
jax.ffi.register_ffi_target("my_target", capsule, platform="cpu")
cx.register_ffi_target("my_target", capsule, platform="cpu")
```

`mirror_jax_registrations` does it for every library. After one call, whatever is
registered through `jax.ffi.register_ffi_target` also reaches the plugin, so the
libraries work unchanged. Call it before importing them. A library that was imported
earlier is not covered, and neither are handlers of the untyped interface.

```python
cx.mirror_jax_registrations()
import my_library  # Registers its handlers with JAX, and now with the plugin.
```

A handler registers for one platform, and registrations for another platform than the
plugin's wait unused. Handlers that need other stages than `execute`, such as an
`instantiate` stage for state, are refused, because the PJRT interface takes one
function.

## Lowering a function with a custom call

`jax.export` refuses custom calls that it has no compatibility promise for, and that
includes every target of a third-party library. Name the targets that the function may
call, and lowering goes ahead.

```python
def scaled(x):
    call = jax.ffi.ffi_call("my_target", jax.ShapeDtypeStruct(x.shape, x.dtype))
    return call(x, factor=3.0)


program = cx.lower(scaled, shapes, custom_calls=["my_target"])
```

The compiled executable refers to the handler by name. The disk cache therefore stays
valid when a library updates its handler. The key of a *lowered* program only changes
when the caller adds the library to the distributions of `cache_key`.

## What does not run

- Handlers of the untyped interface (`api_version=0`) fail to compile with an error
  about the API version.
- `jax.pure_callback` and `jax.experimental.io_callback` call back into Python, which
  the plugin cannot do. JAX refuses to serialize them.
- A target that nobody registered fails to compile with "No FFI handler registered for"
  and the target name, followed by a reminder of the two ways above.
- Arguments and results must be float64, so functions with complex results convert them
  inside the function.
