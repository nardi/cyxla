# Development

The repository holds two packages. `cyxla/` is the runtime, and `cyxla-pjrt-cpu/` packs
the CPU plugin. The documentation sits next to the runtime in `cyxla/docs`.

## Setup and tests

```bash
cd cyxla
uv sync
uv run pytest
```

`uv sync` builds the extension module in editable mode and installs the plugin package
from `../cyxla-pjrt-cpu`. Set `CYXLA_PJRT_CPU_LIBRARY` to a plugin library first when no
release of that package exists yet. The tests need 100% coverage, including the Cython
module, whose lines the development build traces. They also run the code blocks of the
README and of these pages, which carry a `<!-- name: test_... -->` comment.

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
```

## Layout

- `src/cyxla/_runtime.py` is the Cython module in pure Python mode. `_runtime.pxd`
  declares the runtime, and `__init__.pxd` re-exports what extension modules cimport, and `_runtime.pyi` types the Python-visible
  part. `_pjrt.pxd` declares the part of the PJRT C API that the runtime uses.
- `_platform.h` holds every operating system call: loading libraries, aligned memory and
  the clock. The runtime stays portable to Linux, macOS and Windows through it.
- `_plugins.py` finds and loads plugins, `_lowering.py` lowers functions, `_cache.py`
  holds the disk cache and `_ffi.py` registers custom call handlers.
- `tests/downstream` is a small package that builds against an installed cyxla, and
  `tests/ffi_library` is a C++ handler for the custom call tests.
- `notebooks/colab_gpu_tests.ipynb` runs the checks on a CUDA GPU, which CI does not
  have.

## Conventions

- Bump `ABI_VERSION` in `_runtime.pxd` whenever a declared struct, enum or function
  signature changes.
- Code is formatted and linted with Ruff and type-checked with `ty`. Avoid inheritance,
  prefer protocols and plain functions, and use `match` statements where they replace
  chains of conditions.
- The Cython code uses the pure Python syntax, and the functions that Python sees have
  type stubs.

## Continuous integration

The `cyxla` workflow tests on Linux, macOS and Windows with Python 3.11 and 3.13, checks
combined coverage and lint, and builds wheels and the sdist with cibuildwheel. The tests
take the plugin library from the latest run of the `cyxla-pjrt-cpu` workflow that built
one for their platform.

The `cyxla-pjrt-cpu` workflow builds the plugin on four platforms, packs the wheels,
installs each wheel and runs a program on it. A tag `cyxla-v<version>` publishes cyxla
to PyPI, and `cyxla-pjrt-cpu-v<version>` does the same for the plugin package with
trusted publishing. Running either workflow by hand with `release` set does the same
from `main`, and it creates the tag itself, for a session that cannot push tags.

## Documentation

The site uses [Zensical](https://zensical.org/). A build from `cyxla/` looks like this:

```bash
uv sync --only-group docs
uv run --no-sync zensical build --strict
```

The workflow `docs.yml` publishes it to GitHub Pages on every push to `main`.
`mkdocstrings` generates the API reference from the docstrings, so each docstring is
also documentation.
