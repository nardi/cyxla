# cyxla

This repository holds two Python packages.

- [`cyxla`](cyxla) runs XLA programs from Cython extensions through the PJRT C API. It
  lowers JAX functions to StableHLO and executes them on host buffers without the Python
  interpreter.
- [`cyxla-pjrt-cpu`](cyxla-pjrt-cpu) ships the XLA CPU plugin as a wheel, because jaxlib
  links its CPU client statically. Plugins for GPU and TPU come from the JAX plugin
  wheels.

The [documentation](https://nardi.github.io/cyxla/) covers both packages. Each directory
has its own README, tests and `pyproject.toml`. The workflows in `.github/workflows`
build, test and release both packages.
