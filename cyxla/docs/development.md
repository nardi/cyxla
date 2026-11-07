# Development

The repository contains two packages, `cyxla/` for the runtime and `cyxla-pjrt-cpu/` for
the CPU plugin package, and the documentation is in `cyxla/docs`.

## Setup and tests

```bash
cd cyxla
uv sync
uv run pytest
```

`uv sync` builds the extension module in editable mode and installs the plugin package
from `../cyxla-pjrt-cpu`. Set `CYXLA_PJRT_CPU_LIBRARY` to a plugin library first when no
release of that package exists yet. The test suite requires 100% line coverage,
including the Cython module, for which the development build has line tracing enabled.
The code blocks of the README and of these pages that have a `<!-- name: test_... -->`
comment are run as tests too.

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
```

## Layout

- `src/cyxla/_runtime.py` is the Cython module in pure Python mode. `_runtime.pxd`
  declares the runtime, `__init__.pxd` re-exports what extension modules cimport, and
  `_runtime.pyi` types the Python-visible part. `_pjrt.pxd` declares the part of the
  PJRT C API that the runtime uses.
- `_platform.h` contains every operating system call, for loading libraries, aligned
  memory and the clock. Because these calls are in one file, the rest of the runtime is
  portable to Linux, macOS and Windows.
- `_plugins.py` finds and loads plugins, `_lowering.py` lowers functions, `_cache.py`
  implements the disk cache and `_ffi.py` registers custom call handlers.
- `tests/downstream` is a small package that builds against an installed cyxla, and
  `tests/ffi_library` is a C++ handler for the custom call tests.
- `notebooks/colab_gpu_tests.ipynb` runs the checks on a CUDA GPU, which CI does not
  have.

## Conventions

- Bump `ABI_VERSION` in `_runtime.pxd` whenever a declared struct, enum or function
  signature changes.
- Ruff formats and lints the code, and `ty` type-checks it. Avoid inheritance, prefer
  protocols and plain functions, and use `match` statements where they replace chains of
  conditions.
- The Cython code uses the pure Python syntax, and the functions that are visible to
  Python have type stubs.

## Continuous integration

The `cyxla` workflow tests on Linux, macOS and Windows with Python 3.11 and 3.13, checks
combined coverage and lint, and builds wheels and the sdist with cibuildwheel. The tests
take the plugin library from the latest run of the `cyxla-pjrt-cpu` workflow that built
one for their platform.

The `cyxla-pjrt-cpu` workflow builds the plugin on four platforms, creates the wheels,
installs each wheel and runs a program on it. Pushing a tag `cyxla-v<version>` starts a
release that publishes cyxla to PyPI with trusted publishing, and a tag
`cyxla-pjrt-cpu-v<version>` does the same for the plugin package. A workflow that is run
by hand with `release` set makes the same release from `main` and creates the tag
itself, which is useful when tags cannot be pushed.

## Documentation

The site uses [Zensical](https://zensical.org/), and a build from `cyxla/` looks like
this:

```bash
uv sync --only-group docs
uv run --no-sync zensical build --strict
```

The workflow `docs.yml` publishes it to GitHub Pages on every push to `main`.
`mkdocstrings` generates the API reference from the docstrings, so the docstrings are
part of the documentation.
