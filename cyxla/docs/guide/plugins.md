# Plugins

A plugin is a shared library that implements the PJRT C API for one kind of hardware. A
process loads one plugin and creates its client on the first device. The first function
that needs the runtime loads the default plugin, so `initialize` is only for choosing a
plugin or its options, and it has to come before that first use.

## Plugin selection

Without a choice, the default is the CPU plugin if one is installed, then CUDA, ROCm and
TPU in that order. A platform name asks for one, and a path loads a library directly.

```python
from pathlib import Path

import cyxla as cx

cx.initialize(platform="cuda")
```

A path loads that library, and `platform` then names the platform that it implements.

```python
cx.initialize(Path("libpjrt.so"), platform="cpu")
```

`python -m cyxla` lists every plugin that it finds and where it came from. A process
loads one plugin, so later calls return the details of the first.
[`runtime_info`][cyxla.runtime_info] returns the platform, the PJRT API version, and
whether executions avoid copies.

## Options

Each plugin comes with the options that `PJRT_Client_Create` receives, and `options`
adds to or replaces them.

```python
cx.initialize(platform="cuda", options={"memory_fraction": 0.5})
```

The CPU client runs synchronously on one device, so executions stay in the calling
thread. The GPU clients do not preallocate most of the device memory, as JAX does by
default. Before it loads JAX's CUDA plugin, cyxla loads the CUDA libraries of the
`nvidia` wheels, the same way JAX does.

## Plugin packages

Other packages can offer a plugin with an entry point in the `cyxla.plugins` group. The
entry point names a module with `PLATFORM`, `CREATE_OPTIONS` and `library_path()`, as
`cyxla_pjrt_cpu` does. cyxla also finds the plugins of JAX's GPU wheels and the TPU
library without any entry point.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `CYXLA_PJRT_PLUGIN` | Path of a plugin that replaces the installed ones |
| `CYXLA_PLATFORM` | Platform to load when `initialize` gets none |
| `CYXLA_CACHE_DIR` | Directory of the disk cache, `~/.cache/cyxla` if unset |
| `CYXLA_MIRROR_JAX` | Set to `0` to keep cyxla from mirroring the FFI registrations of JAX |
