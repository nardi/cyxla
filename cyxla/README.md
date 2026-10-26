# cyxla

cyxla lowers JAX functions to StableHLO, compiles them with a PJRT plugin, and runs them
from Cython extension modules without the GIL. It came out of
[JAX-CIP](https://github.com/nardi/jax-cip), whose SCIP callbacks call JAX functions
thousands of times per second, and it is the base of other packages that do the same.

## Installation

The package and a plugin install from wheels, and nothing downloads at run time.

```bash
pip install cyxla[cpu]
```

Each extra installs the plugin of one platform:

| Extra | Platform | Plugin |
| --- | --- | --- |
| `cpu` | Linux on x86-64 and arm64, macOS on arm64, Windows on x86-64 | [cyxla-pjrt-cpu](../cyxla-pjrt-cpu) |
| `cuda12`, `cuda13` | NVIDIA GPUs on Linux | JAX's `jax-cuda12-pjrt` or `jax-cuda13-pjrt`, with the CUDA libraries |
| `rocm7` | AMD GPUs on Linux | JAX's `jax-rocm7-pjrt`, with ROCm from the system |
| `tpu` | Cloud TPUs | `libtpu` |

jaxlib links its own CPU client statically and ships no plugin library, which is why the
CPU plugin is a separate package.

## Example

<!-- name: test_readme -->
```python
import jax
import jax.numpy as jnp
import numpy as np

import cyxla as cx

cx.enable_x64()
cx.initialize()
program = cx.lower(lambda x: jnp.sin(x) * 2, [jax.ShapeDtypeStruct((3,), jnp.float64)])
executable, statistics = cx.compile_program(program)
(result,) = executable.run([np.array([0.0, 1.0, 2.0])], output_sizes=[3])
```

`Executable.run` is for tests and interactive checks. Extension modules call `execute`
from `cyxla/_runtime.pxd` instead, as the next section shows.

## Extension modules

An extension module cimports the runtime and runs programs with `execute`, which takes
no GIL. On the CPU, inputs alias host arrays that the module fills in place, and outputs
are read from the buffers' memory, so an execution copies nothing beyond the arrays that
the caller passes. On other platforms every execution copies its inputs to the device
and its outputs back.

```python
import cython
from cython.cimports.cyxla._runtime import ABI_VERSION, Input, ProgramSpec, execute

import cyxla

cyxla.check_abi(ABI_VERSION)
```

`check_abi` rejects a module that was compiled against declarations of another cyxla
version. Cython checks the signatures of shared functions itself, but not the layout of
structs such as `ProgramSpec`.

Host arrays for inputs come from `aligned_doubles` and go back with `free_doubles`. The
pair exists because Windows frees aligned memory with a function other than `free`.

The build needs the PJRT header and the `.pxd` files, and `cyxla.get_include()` returns
their directories. A Meson build reads them like this, and `tests/downstream` is a
complete example.

```meson
cyxla_include = run_command(
  python, '-c', 'import cyxla; print("\\n".join(cyxla.get_include()))', check: true,
).stdout().strip().split('\n')
# Pass cyxla_include[1] to Cython with -I, and cyxla_include[0] to the C compiler.
```

Downstream packages list `cyxla` in `build-system.requires` and pin it to a narrow
range, because a new version may change the declarations.

## Plugins

`cx.initialize()` loads one plugin per process. Without arguments it takes the CPU
plugin if one is installed, then CUDA, ROCm and TPU in that order.
`cx.initialize(platform="cuda")` asks for a platform, and
`cx.initialize(Path("libpjrt.so"), platform="cpu")` loads a library by path. `python -m
cyxla` lists what it finds.

Each plugin comes with the options that `PJRT_Client_Create` receives, and `options`
adds to or replaces them:

```python
cx.initialize(platform="cuda", options={"memory_fraction": 0.5})
```

The CPU client runs synchronously on one device, so executions stay in the calling
thread. The GPU clients do not preallocate most of the device memory, as JAX does by
default. Before it loads JAX's CUDA plugin, cyxla loads the CUDA libraries of the
`nvidia` wheels, the same way JAX does.

Other packages can provide a plugin with an entry point in the `cyxla.plugins` group.
The entry point names a module with `PLATFORM`, `CREATE_OPTIONS` and `library_path()`,
as `cyxla_pjrt_cpu` has.

## Disk cache

`compile_program` stores executables in the cache directory under a hash of their
StableHLO, the compile options and the plugin. A later run of the same program loads the
executable instead of compiling it. `store_lowered` and `load_lowered` keep lowered
programs under a key from `cache_key`, which includes the versions of JAX, cyxla and any
distributions that the caller names.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `CYXLA_PJRT_PLUGIN` | Path of a plugin that replaces the installed ones |
| `CYXLA_PLATFORM` | Platform to load when `initialize` gets none |
| `CYXLA_CACHE_DIR` | Directory of the disk cache, `~/.cache/cyxla` if unset |

## Limitations

- Buffers hold float64 values only. TPUs do not support float64, so programs for them
  fail to compile.
- cyxla does not register the custom calls of jaxlib with the plugin. Functions that JAX
  lowers to them, such as most of `jnp.linalg`, fail to compile. Elementwise math,
  reductions and control flow work.
- The GPU and TPU paths are tested up to client creation. CI has no GPU, so execution on
  one is untested.
- On Windows only the CPU plugin exists, since JAX publishes no native GPU plugins
  there.

## Development

```bash
uv sync
uv run pytest
```

`uv sync` builds the CPU plugin package from `../cyxla-pjrt-cpu`, which downloads its
archive once. Development builds trace lines in the Cython module, so its coverage is
measured too. The tests need 100% coverage.

## License

[Hippocratic License 3.0](LICENSE.md). The vendored PJRT header is under XLA's Apache
2.0 license.
