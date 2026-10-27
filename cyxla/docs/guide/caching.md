# Cache and memory

Lowering a function and compiling it dominate the start-up of a program, and neither
changes between runs. cyxla keeps both results on disk.

## The disk cache

The cache directory is `~/.cache/cyxla`, or the one in `CYXLA_CACHE_DIR`. Two kinds of
entries live there.

- `compile_program` stores each executable under a hash of its StableHLO, the compile
  options and the plugin. A later run of the same program loads the executable. An
  executable that the plugin rejects, for instance because the plugin was updated, is
  deleted and compiled again.
- `store_lowered` and `load_lowered` keep lowered programs under a key that the caller
  builds with `cache_key`. A package that lowers many functions uses them to skip
  tracing in later runs.

The key of a lowered program covers the versions of JAX, jaxlib and cyxla. A library
whose code shapes the function names its distribution as well, and then a new version of
it changes the key.

<!-- name: test_docs_caching -->
```python
import jax
import jax.numpy as jnp
import numpy as np

import cyxla as cx

cx.enable_x64()
cx.initialize()
key = cx.cache_key("polynomial", "degree 3", distributions=("numpy",))
cached = cx.load_lowered(key)
if cached is None:
    program = cx.lower(lambda x: x**3 - x, [jax.ShapeDtypeStruct((2,), jnp.float64)])
    cx.store_lowered(key, [program])
    cached = (program,)
(program,) = cached
executable, statistics = cx.compile_program(program)
(result,) = executable.run([np.array([1.0, 2.0])], output_sizes=[2])
```

<!-- name: test_docs_caching -->
```python
assert np.allclose(result, [0.0, 6.0])
```

Several processes may share one directory. Files appear under their name only when they
are complete.

## Memory mappings

The CPU plugin compiles a program to machine code and keeps about 200 memory mappings
per executable. Linux allows 65,530 mappings per process by default, so a process that
holds hundreds of distinct executables at once can reach the limit, and XLA then fails
to compile with "Failed to materialize symbols". Release executables that you no longer
need. A system that keeps many alive can raise `vm.max_map_count` with `sysctl`.
