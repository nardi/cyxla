# The CPU plugin package

`cyxla-pjrt-cpu` packs XLA's CPU plugin as a wheel. cyxla finds it through the
`cyxla.plugins` entry point, so nothing needs configuring.

```bash
pip install cyxla[cpu]
```

## Platforms

The wheels cover the platforms of jaxlib. jaxlib no longer builds for macOS on x86-64,
so neither does this package.

| Platform | Library |
| --- | --- |
| Linux x86-64 | `libpjrt_cpu.so` |
| Linux arm64 | `libpjrt_cpu.so` |
| macOS arm64 | `libpjrt_cpu.dylib` |
| Windows x86-64 | `pjrt_cpu.dll` |

The version of the package is the version of the JAX release whose XLA it contains.

## How it is built

`build_plugin.py` checks out JAX at the release in `plugin.toml`. That release's
`MODULE.bazel` pins the XLA commit, with JAX's patches. JAX's own `build/build.py`
writes the Bazel options that jaxlib uses on the platform, and Bazel then builds the
targets in `bazel/BUILD.bazel` with them. The targets link the same CPU client as XLA's
`pjrt_c_api_cpu_plugin.so`, with the linker options of each system and an entry point
that has default visibility. jaxlib compiles with hidden visibility, which would hide
XLA's own definition of `GetPjrtApi`.

Windows needs extra steps. The build uses the clang-cl and the MSVC of the machine, and
it passes the definitions that XLA's pinned Windows toolchain adds to every compile.
Bazel's repository rules link the LLVM sources with relative symlinks, which Windows
resolves against the wrong path, so the build script replaces them with junctions before
it builds.

A build takes hours without a cache. The workflow reads the public cache of jaxlib and
keeps its own disk cache between runs. The cold Windows build took about three hours.

## Wheels

The build backend in `backend.py` packs a library and needs only the standard library.
It reads the platform and the wheel tag from the library itself, using the newest glibc
symbol version on Linux, the minimum system version on macOS, and the machine type on
Windows.

```bash
uv build --wheel -C library=library/libpjrt_cpu.so
```

`CYXLA_PJRT_CPU_LIBRARY` does the same as the config setting, which helps when another
tool runs the build. Without either, the backend downloads the library of this machine
from the GitHub release of the version and checks it against the hash in `plugin.toml`.

## Releases

A tag `cyxla-pjrt-cpu-v<version>` builds the four libraries, attaches them to a GitHub
release, and publishes the wheels and the sdist to PyPI. After a release, the printed
hashes go into `plugin.toml`, so builds without a library can check the downloaded ones.
