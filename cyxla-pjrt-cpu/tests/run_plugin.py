"""Runs a program on the installed plugin through cyxla, as the release check of a wheel."""

import jax
import jax.numpy as jnp
import numpy as np

import cyxla as cx

cx.enable_x64()
(plugin,) = [p for p in cx.available_plugins() if p.source == "cyxla-pjrt-cpu"]
info = cx.initialize(plugin)
print(plugin.path, info)
shapes = (jax.ShapeDtypeStruct((3,), jnp.float64),)
program = cx.lower(lambda x: jnp.sin(x) * 2 + jnp.sum(x), shapes)
executable, _ = cx.compile_program(program, disk_cache=False)
values = np.array([0.0, 1.0, 2.0])
(result,) = executable.run([values], [3])
np.testing.assert_allclose(result, np.sin(values) * 2 + values.sum())
assert info.platform == "cpu" and info.aliased_inputs
