# Plugins

A plugin is a shared library that implements the PJRT C API for one kind of hardware. A
process loads one plugin and creates its client on the first device. The first function
that needs the runtime loads the default plugin, so `initialize` is only needed to
choose a plugin or set its options, and it has to be called before that first use.

## Plugin selection

If no plugin is chosen, the default is the CPU plugin if one is installed, then CUDA,
ROCm and TPU in that order. A platform name selects the plugin of that platform, and a
path selects a library directly.

```python
from pathlib import Path

import cyxla as cx

cx.initialize(platform="cuda")
```

A path loads that library, and `platform` then names the platform that it implements.

```python
cx.initialize(Path("libpjrt.so"), platform="cpu")
```

A lowered program records its platform, so `compile_program` loads the plugin for it
when no plugin is loaded. The [running programs](running.md) guide describes how the
platform of a lowered program is determined from the placement of the arguments and the
default device of JAX. Compiling a program for another platform than the loaded plugin's
raises an error.

`python -m cyxla` lists every installed plugin and where it was found. A process loads
one plugin, so calls of `initialize` after the first return the details of the plugin
that was loaded first. [`runtime_info`][cyxla.runtime_info] returns the platform, the
PJRT API version, and whether executions use zero-copy buffers.

## Options

Each plugin has default options that are passed to `PJRT_Client_Create`. The entries in
`options` are added to these defaults or replace them.

```python
cx.initialize(platform="cuda", options={"memory_fraction": 0.5})
```

The CPU client runs synchronously on one device, so executions run on the calling
thread. The GPU clients do not preallocate most of the device memory, as JAX does by
default. Before it loads JAX's CUDA plugin, cyxla loads the CUDA libraries of the
`nvidia` wheels, the same way JAX does.

## Plugin packages

Other packages can provide a plugin with an entry point in the `cyxla.plugins` group.
The entry point names a module with `PLATFORM`, `CREATE_OPTIONS` and `library_path()`,
as `cyxla_pjrt_cpu` does. cyxla also finds the plugins of JAX's GPU wheels and the TPU
library without any entry point.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `CYXLA_PJRT_PLUGIN` | Path of a plugin that replaces the installed ones |
| `CYXLA_PLATFORM` | Platform to load when no platform is passed to `initialize` |
| `CYXLA_CACHE_DIR` | Directory of the disk cache, `~/.cache/cyxla` if unset |
| `CYXLA_MIRROR_JAX` | Set to `0` so that cyxla does not mirror the FFI registrations of JAX |
