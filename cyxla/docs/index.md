# cyxla

cyxla runs JAX functions from native code. It lowers a function to StableHLO, compiles
it with a PJRT plugin, and executes it from a Cython extension module without taking the
GIL.

The use case is a native library that calls back into a function thousands of times per
second. A solver such as SCIP or Ipopt asks for values, gradients and Hessians of a
model, and calling Python for each of them costs more than the evaluation itself. With
cyxla the model is a JAX function, and the solver's callbacks run its compiled
executable directly. [JAX-CIP](https://nardi.github.io/jax-cip/) uses cyxla to do this
for SCIP.

## Installation

cyxla needs Python 3.11 or newer. The extra installs the plugin that runs programs on
your hardware, so nothing downloads at run time.

```bash
pip install cyxla[cpu]
```

The [installation page](installation.md) lists the plugins for GPUs and TPUs.

## A first program

This example lowers a function, compiles it and runs it. cyxla loads the plugin at the
first compilation.

<!-- name: test_docs_index -->
```python
import jax
import jax.numpy as jnp
import numpy as np

import cyxla as cx

shapes = [jax.ShapeDtypeStruct((3,), jnp.float32)]
program = cx.lower(lambda values: jnp.sin(values) * 2, shapes)
executable, statistics = cx.compile_program(program)
(result,) = executable.run([np.array([0.0, 1.0, 2.0])])
```

<!-- name: test_docs_index -->
```python
assert np.allclose(result, 2 * np.sin([0.0, 1.0, 2.0]))
```

`Executable.run` is meant for tests and interactive checks. Extension modules call the
`execute` function instead, which the [extension modules](guide/extensions.md) guide
describes.

## Supported programs

- Any function that JAX lowers to StableHLO runs, including control flow such as
  `lax.scan`, and `jnp.linalg` through the [LAPACK kernels of
  `jaxlib`](guide/custom-calls.md). Libraries with their own primitives can register
  their handlers too.
- Plugins exist for the CPU on Linux, macOS and Windows, and for NVIDIA GPUs, AMD GPUs
  and TPUs through the plugins that JAX publishes. The [plugins](guide/plugins.md) guide
  explains how cyxla picks one.
- Lowered programs and compiled executables are stored in the [disk
  cache](guide/caching.md), so later processes can load them.

## Limits

TPUs do not support `float64`, the untyped custom calls of older libraries do not run on
the plugins, and Python callbacks run on the CPU only. CI has no GPU or TPU, so those
paths are tested only up to creating the client. The [custom
calls](guide/custom-calls.md) page describes what works there.

## License

cyxla is licensed under the [Hippocratic License
3.0](https://firstdonoharm.dev/version/3/0/core.html). The vendored PJRT headers are
licensed under XLA's Apache 2.0 license.
