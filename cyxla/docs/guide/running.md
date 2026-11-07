# Programs

`lower` converts a JAX function to StableHLO, `compile_program` compiles the StableHLO
into an executable for the loaded plugin, and the executable then runs on host arrays.

## Lowered programs

`lower` takes the function and the shapes of its arguments. It traces the function with
`jax.export`, so any function that JAX can trace is supported, and it returns a
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


shapes = [
    jax.ShapeDtypeStruct((4, 2), jnp.float32),
    jax.ShapeDtypeStruct((4,), jnp.float32),
]
program = cx.lower(energy, shapes)
print(program.text.splitlines()[0])
```

`lower` traces the function with the current setting of `jax.enable_x64`, as JAX does. A
`float64` shape lowers to `float32` while the setting is off, so 64-bit programs need it
on, either globally or as a context manager around the call. The `input_types` of the
program contain the dtype and shape of each argument. The function may have unused
arguments. They are kept in the signature, because callers pass every input of a
program.

Lowering chooses the platform the way JAX chooses where to run. The arguments may be
arrays, or shapes with a sharding, and the platform of the device they are committed to
takes precedence. Next comes the device of `jax.default_device`, then the loaded plugin,
then the first platform in `jax_platforms` that has a plugin, and last the plugin that
cyxla would load by default. A program runs on one device, and cyxla determines its
platform without starting a JAX backend. With the `platform` argument, the function is
lowered for another platform, for instance to prepare programs on a machine without a
GPU. The platform is stored in the program, and `compile_program` loads the plugin for
it. For a function that contains custom calls, the target names have to be passed in
`custom_calls`, as the [custom calls](custom-calls.md) page explains.

Every operation in a program has a source location, which is the file, line and function
of the Python code that it was traced from. XLA copies these locations into the
executable. `lower(..., debug_info=False)` removes them before compiling, which makes
the serialized executable about 15% smaller for programs with many operations and
shortens the time to load it. The plugin then reports operations by name only, so
profiles and compiler messages no longer contain source locations.

## Compilation

`compile_program` returns the executable and the
[`CompileStatistics`][cyxla.CompileStatistics] of the call. By default the executable is
stored in the [disk cache](caching.md), so a second process loads it instead of
compiling again.

<!-- name: test_docs_running -->
```python
executable, statistics = cx.compile_program(program, disk_cache=False)
print(statistics.compiled, executable.num_outputs)
```

## Execution from Python

`Executable.run` takes one array per argument and returns one array per output, with the
shape and dtype that the program declares. An argument is converted to the argument type
of the program if the conversion only loses precision. A `float64` array is accepted for
a `float32` argument and integers are accepted for floats, but floats are not accepted
for integers.

<!-- name: test_docs_running -->
```python
random_generator = np.random.default_rng(0)
positions = random_generator.normal(size=(4, 2))
charges = random_generator.uniform(1.0, 2.0, size=4)
(value,) = executable.run([positions, charges])
assert np.allclose(value, energy(positions, charges))
```

A program takes at most 8 arguments and returns at most 8 outputs. `run` copies its
arguments into aligned memory first, so a timing of `run` includes this copy and is
longer than the execution itself. It is meant for tests and interactive checks. The
`execute` function of an [extension module](extensions.md) reuses the buffers and does
not acquire the GIL.

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

`lower` raises an error for a function with another dtype, such as the 8-bit floats,
4-bit integers and PRNG keys, and the error names the argument or result. TPUs do not
support `float64`, so programs for them fail to compile.
