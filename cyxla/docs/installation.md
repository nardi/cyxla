# Installation

cyxla comes as wheels for Linux on x86-64 and arm64, macOS on arm64 and Windows on
x86-64. The wheels bundle everything they need, so an installation works offline.

```bash
pip install cyxla[cpu]
```

## Plugins

A plugin is the XLA library that compiles and runs programs for one kind of hardware.
Each extra installs the plugin of one platform.

| Extra | Platform | Plugin |
| --- | --- | --- |
| `cpu` | Linux on x86-64 and arm64, macOS on arm64, Windows on x86-64 | [cyxla-pjrt-cpu](cpu-plugin.md) |
| `cuda12`, `cuda13` | NVIDIA GPUs on Linux | `jax-cuda12-plugin` or `jax-cuda13-plugin`, with the CUDA libraries |
| `rocm7` | AMD GPUs on Linux | `jax-rocm7-pjrt`, with ROCm from the system |
| `tpu` | Cloud TPUs | `libtpu` |

jaxlib links its own CPU client statically and ships no plugin library. That is why the
CPU plugin is a separate package, which holds an XLA build of the same release as the
JAX version in its version number. JAX publishes the GPU and TPU plugins, and cyxla
finds them in the same environment.

The CUDA extras install the plugin package of JAX instead of only its PJRT library,
because the linear algebra kernels of the GPU plugins live there. Windows has the CPU
plugin only, since JAX publishes no native GPU plugins for it.

## Source builds

An sdist builds with Meson, Cython and a C compiler, which is MSVC on Windows.

```bash
git clone https://github.com/nardi/cyxla
cd cyxla/cyxla
uv sync
```

`uv sync` installs the package in editable mode with the development tools, and Meson
rebuilds the extension module when you import it after a change. It also builds the CPU
plugin package from `../cyxla-pjrt-cpu`. That package packs a library that CI built, and
it downloads the library of the matching release unless `CYXLA_PJRT_CPU_LIBRARY` names
one. The [development](development.md) page has the details.

Development builds trace lines of the Cython module for coverage, which slows it down.
Measure timings with a build without tracing:

```bash
uv pip install --no-config --no-deps --no-build-isolation --reinstall -e . \
    -C setup-args=-Dcoverage=false
```
