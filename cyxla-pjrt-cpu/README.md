# cyxla-pjrt-cpu

The [XLA](https://github.com/openxla/xla) PJRT CPU plugin as a wheel, for
[cyxla](https://github.com/nardi/cyxla/tree/main/cyxla). jaxlib links its CPU client
statically and ships no plugin library, so cyxla needs this package to run programs on
the CPU.

```bash
pip install cyxla[cpu]
```

cyxla finds the plugin through the `cyxla.plugins` entry point, so nothing needs
configuring and nothing downloads at run time.

## Platforms

The wheels cover the platforms of jaxlib. jaxlib no longer builds for macOS on x86-64,
so neither does this package.

| Platform | Library |
| --- | --- |
| Linux x86-64 | `libpjrt_cpu.so` |
| Linux arm64 | `libpjrt_cpu.so` |
| macOS arm64 | `libpjrt_cpu.dylib` |
| Windows x86-64 | `pjrt_cpu.dll` |

## Build

The plugin is built from XLA the way jaxlib is. `build_plugin.py` checks out JAX at the
release in `plugin.toml`, and that release's `MODULE.bazel` pins the XLA commit, with
JAX's patches. JAX's own `build/build.py` writes the Bazel options that jaxlib uses on
the platform. Bazel then builds the targets in `bazel/BUILD.bazel` with them.

```bash
python build_plugin.py --output library --disk-cache ~/bazel-disk-cache
```

The targets link the same CPU client as XLA's `pjrt_c_api_cpu_plugin.so`, but with the
linker options of each system. `bazel/entry.cc` defines `GetPjrtApi` with default
visibility, because jaxlib's Bazel options hide XLA's own definition and no linker option
can export a hidden symbol. On Windows, `bazel/exports.def` lists the function as well.

Bazel comes from `bazelisk`, which reads the version in JAX's `.bazelversion`. Without a
cache the build takes hours. The builds read jaxlib's public read-only Bazel cache, and
the disk cache keeps earlier outputs. The `cyxla-pjrt-cpu` workflow saves the disk cache
of each platform between runs, including runs that fail or time out.

## Wheels

The build backend in `backend.py` packages a library and needs only the standard
library. It reads the platform and the wheel tag from the library itself, using the
newest glibc symbol version on Linux, the minimum system version on macOS, and the
machine type on Windows.

```bash
uv build --wheel -C library=library/libpjrt_cpu.so
```

`CYXLA_PJRT_CPU_LIBRARY` does the same as the config setting, which helps when another
tool runs the build, such as `uv sync` in cyxla. Without either, the backend downloads
the library of this machine from the GitHub release of the version and checks it against
the hash in `plugin.toml`. Building from the sdist works the same way.

## Releases

A tag `cyxla-pjrt-cpu-v<version>` builds the four libraries, attaches them to a GitHub
release, and publishes the wheels and the sdist to PyPI. After a release, record the
printed hashes in `plugin.toml`, so that builds without a library can check the
downloaded ones.

## License

The plugin is part of XLA, and the package is under XLA's [Apache 2.0](LICENSE) license.
