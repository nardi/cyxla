# Extension modules

An extension module cimports the runtime and calls `execute`, which runs an executable
from C without the GIL. A solver can then call it from the threads of its callbacks.

## The `execute` function

`execute` takes a `ProgramSpec` for the executable and an array of `Input` structs. Each
input is a host array with a PJRT element type, such as `PJRT_Buffer_Type_F32`, and its
number of elements. On the CPU the plugin reads the arrays in place, so the module
writes the inputs into these arrays and no copy is made. Outputs are also read directly
from the memory of the buffers, and each destination must have room for the number of
bytes in `ProgramSpec.output_bytes`. On other platforms every execution copies its
inputs to the device and its outputs back.

Host arrays are allocated with `aligned_memory`, which takes a size in bytes, and
released with `free_aligned`. They are aligned to 64 bytes, which the plugin needs to
alias them. The pair exists because Windows frees aligned memory with a function other
than `free`. `itemsize` returns the size of one element of a type. The element types and
shapes of the outputs are in `Executable.output_types`, and their sizes are in
`Executable.output_bytes`.

The declarations are in `cyxla/__init__.pxd`, which re-exports them from the private
modules that define them. A module cimports them from `cyxla`, but it must not use the
runtime from two threads at once, because the runtime's state is shared.

```python
import cython
from cython.cimports.cyxla import ABI_VERSION, Input, ProgramSpec, execute

import cyxla

cyxla.check_abi(ABI_VERSION)
```

## Compatibility

`check_abi` rejects a module that was compiled against declarations of another cyxla
version. Cython checks the signatures of shared functions itself, but not the layout of
structs such as `ProgramSpec`. `ABI_VERSION` is incremented whenever a declaration
changes.

A downstream package lists `cyxla` in `build-system.requires`, pinned to a narrow range,
because a new version may change the declarations.

## Build setup

The build needs the PJRT header and the `.pxd` files, and `cyxla.get_include()` returns
their directories. The first is for the C compiler and the second is for Cython. A Meson
build reads them like this:

```meson
cyxla_include = run_command(
  python, '-c', 'import cyxla; print("\\n".join(cyxla.get_include()))', check: true,
).stdout().strip().split('\n')
# Pass cyxla_include[1] to Cython with -I, and cyxla_include[0] to the C compiler.
```

The directory `tests/downstream` in the repository is a complete small package that
builds this way and runs a program.

## Results and errors

`execute` returns -1 when something fails, and 0 otherwise.
`raise_last_error(operation)` raises the last PJRT error as a Python exception, for a
module that checks the result while it holds the GIL. With `zero_copy`, `execute` uses
aliased buffers, and on a plugin that cannot alias host memory it copies instead. An
output whose destination is `NULL` is not copied. After a failure, `error_info()`
returns the number of PJRT errors so far and the last message.

## Processes without JAX

Importing cyxla does not import JAX, and only `lower` does, when it traces a function. A
process that loads serialized executables with `Executable.deserialize` and runs them
therefore starts about half a second faster, and JAX is never imported if no other code
imports it. Pass `jax_targets=False` to `initialize` in such a process unless a program
calls `jnp.linalg` kernels. The handlers of `jaxlib` for them are registered by default,
and that registration takes longer than the rest of the initialization.
