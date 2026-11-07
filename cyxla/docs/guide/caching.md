# Cache and memory

Lowering a function and compiling it dominate the start-up of a program, and neither
result changes between runs. cyxla stores both results on disk.

## The disk cache

The cache directory is `~/.cache/cyxla`, or the one in `CYXLA_CACHE_DIR`. It contains
two kinds of entries.

- `compile_program` stores each executable under a hash of its StableHLO, the compile
  options and the plugin. When the same program is compiled again, the executable is
  loaded from the cache. An executable that the plugin rejects, for instance because the
  plugin was updated, is deleted and compiled again.
- `store_lowered` and `load_lowered` store and load lowered programs under a key that
  the caller builds with `cache_key`. A package that lowers many functions uses them to
  skip tracing in later runs.

The key of a lowered program includes the versions of JAX, `jaxlib` and cyxla. The
versions of other libraries that the lowered function depends on are included as well
when their distributions are listed in the `distributions` argument of `cache_key`.

<!-- name: test_docs_caching -->
```python
import jax
import jax.numpy as jnp
import numpy as np

import cyxla as cx

key = cx.cache_key("polynomial", "degree 3", distributions=("numpy",))
cached = cx.load_lowered(key)
if cached is None:
    shapes = [jax.ShapeDtypeStruct((2,), jnp.float32)]
    program = cx.lower(lambda values: values**3 - values, shapes)
    cx.store_lowered(key, [program])
    cached = (program,)
(program,) = cached
executable, statistics = cx.compile_program(program)
(result,) = executable.run([np.array([1.0, 2.0])])
```

<!-- name: test_docs_caching -->
```python
assert np.allclose(result, [0.0, 6.0])
```

Several processes may share one directory, because a file is written under a temporary
name and only renamed once it is complete.

## Memory mappings

The CPU plugin compiles a program to machine code and uses about 200 memory mappings per
executable. Linux allows 65,530 mappings per process by default, so a process that has
hundreds of distinct executables at once can reach the limit, and XLA then fails to
compile with "Failed to materialize symbols". Release executables that you no longer
need. On a system where many executables have to stay loaded, `vm.max_map_count` can be
raised with `sysctl`.
