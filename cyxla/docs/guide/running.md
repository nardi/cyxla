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
shapes = [
    jax.ShapeDtypeStruct((4, 2), jnp.float64),
    jax.ShapeDtypeStruct((4,), jnp.float64),
]
program = cx.lower(energy, shapes)
print(program.text.splitlines()[0])
```

JAX lowers to 32-bit types, and `enable_x64` switches to 64-bit types, which this
example needs for its `float64` arrays. The `input_types` of the program record the
dtype and shape of each argument. The function may have unused arguments. They stay in
the signature, because callers pass every input of a program.

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

`Executable.run` takes one array per argument and returns one array per output, with the
shape and dtype that the program gives. An argument converts to the type of the program
when that loses no more than precision. A `float64` array passes for a `float32`
argument and integers pass for floats, but floats do not pass for integers.

<!-- name: test_docs_running -->
```python
rng = np.random.default_rng(0)
positions, charges = rng.normal(size=(4, 2)), rng.uniform(1.0, 2.0, size=4)
(value,) = executable.run([positions, charges])
assert np.allclose(value, energy(positions, charges))
```

A program takes at most 8 arguments and returns at most 4 outputs. `run` copies its
arguments into aligned memory first, so it measures more than an execution does. It is
meant for tests and interactive checks. The `execute` function of an [extension
module](extensions.md) reuses the buffers and takes no GIL.

## Element types

Arguments and results can be `bool`, `int8` to `int64`, `uint8` to `uint64`, `float16`,
`bfloat16`, `float32`, `float64`, `complex64` and `complex128`, and one program can mix
them.

<!-- name: test_docs_running -->
```python
shapes = [
    jax.ShapeDtypeStruct((3,), jnp.int8),
    jax.ShapeDtypeStruct((3,), jnp.bfloat16),
]
program = cx.lower(lambda counts, weights: (counts * 2, weights * 1j), shapes)
executable, _ = cx.compile_program(program, disk_cache=False)
doubled, rotated = executable.run([[1, 2, 3], [0.5, 1.5, 2.5]])
assert doubled.dtype == np.int8 and rotated.dtype == np.complex64
```

`lower` refuses a function with another dtype, such as the 8-bit floats, 4-bit integers
and PRNG keys, and the error names the argument or result. TPUs do not support
`float64`, so programs for them fail to compile.
