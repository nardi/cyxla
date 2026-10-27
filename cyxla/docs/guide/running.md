# Programs

A program goes through three steps. `lower` turns a JAX function into StableHLO,
`compile_program` turns that into an executable for the loaded plugin, and the
executable runs on host arrays.

## Lowered programs

`lower` takes the function and the shapes of its arguments. It traces the function with
`jax.export`, so everything that JAX traces works, and it returns a
[`LoweredProgram`][cyxla.LoweredProgram] with the portable bytecode and the module as
text.

<!-- name: test_docs_running -->
```python
import jax
import jax.numpy as jnp
import numpy as np

import cyxla as cx


def energy(positions, charges):
    difference = positions[:, None] - positions[None, :]
    distance = jnp.sqrt(jnp.sum(difference**2, axis=-1) + jnp.eye(len(charges)))
    pairs = charges[:, None] * charges[None, :] / distance
    return jnp.sum(jnp.triu(pairs, 1))[None]


cx.enable_x64()
cx.initialize()
shapes = [
    jax.ShapeDtypeStruct((4, 2), jnp.float64),
    jax.ShapeDtypeStruct((4,), jnp.float64),
]
program = cx.lower(energy, shapes)
print(program.text.splitlines()[0])
```

Call `enable_x64` before lowering, since the buffers hold `float64` values. The function
may have unused arguments. They stay in the signature, because callers pass every input
of a program.

Lowering targets the platform of the loaded plugin, or the CPU before any plugin is
loaded. The `platform` argument lowers for another one, for instance to prepare programs
on a machine without a GPU. A function that calls custom calls needs their names in
`custom_calls`, as the [custom calls](custom-calls.md) page explains.

## Compilation

`compile_program` returns the executable and the
[`CompileStatistics`][cyxla.CompileStatistics] of the call. By default it keeps the
executable in the [disk cache](caching.md), so a second process loads it instead of
compiling again.

<!-- name: test_docs_running -->
```python
executable, statistics = cx.compile_program(program, disk_cache=False)
print(statistics.compiled, executable.num_outputs)
```

## Execution from Python

`Executable.run` takes one array per argument and the number of values in each output.
Every array is flattened in row-major order and converted to float64, and the outputs
are returned flat.

<!-- name: test_docs_running -->
```python
rng = np.random.default_rng(0)
positions, charges = rng.normal(size=(4, 2)), rng.uniform(1.0, 2.0, size=4)
(value,) = executable.run([positions, charges], output_sizes=[1])
assert np.allclose(value, energy(positions, charges))
```

A program takes at most 8 arguments and returns at most 4 outputs. `run` copies its
arguments into aligned memory first, so it measures more than an execution does. It is
meant for tests and interactive checks. The `execute` function of an [extension
module](extensions.md) reuses the buffers and takes no GIL.

## Platforms and dtypes

The runtime supports `float64` buffers only. Integer or complex values may appear inside
a program, but its arguments and results have to be `float64`. A function with complex
results, such as the eigenvalues of `jnp.linalg.eig`, converts them to real parts inside
the function. TPUs do not support `float64`, so programs for them fail to compile.
