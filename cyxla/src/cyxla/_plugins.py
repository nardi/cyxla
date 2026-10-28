"""Discovery and loading of PJRT plugins.

Plugins come from three places, which `available_plugins` checks in this order.

1. Distributions with an entry point in the `cyxla.plugins` group, such as
   `cyxla-pjrt-cpu`. The entry point names a module that matches `PluginModule`.
2. The CUDA and ROCm plugins of JAX (`jax-cuda12-pjrt`, `jax-cuda13-pjrt` and
   `jax-rocm7-pjrt`), which install their libraries under the `jax_plugins` namespace.
3. The TPU runtime in `libtpu`.

The environment variable `CYXLA_PJRT_PLUGIN` overrides all of them with a path. None of
these imports JAX's plugin modules, which would register the plugins with JAX itself.
"""

import ctypes
import importlib.metadata
import importlib.util
import os
import re
import threading
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple, Protocol

from cyxla import _ffi, _runtime
from cyxla._types import OptionValue, Plugin, RuntimeInfo

PLUGIN_ENVIRONMENT_VARIABLE = "CYXLA_PJRT_PLUGIN"
"""Environment variable with the path of a plugin that overrides the installed ones."""

PLATFORM_ENVIRONMENT_VARIABLE = "CYXLA_PLATFORM"
"""Environment variable with the platform to load when none is passed."""

ENTRY_POINT_GROUP = "cyxla.plugins"
"""Entry point group of distributions that provide a plugin."""

PLATFORM_PREFERENCE = ("cpu", "cuda", "rocm", "tpu")
"""Platforms in the order that `find_plugin` picks them without a requested platform.

The CPU comes first because its buffers alias host memory, which is what small programs
called from native code benefit from most.
"""

DEFAULT_CREATE_OPTIONS: Mapping[str, Mapping[str, OptionValue]] = {
    "cpu": {"asynchronous": False, "cpu_device_count": 1},
    "cuda": {"platform_name": "cuda", "preallocate": False},
    "rocm": {"platform_name": "ROCM", "preallocate": False},
}
"""Client options of plugins that do not state their own.

The CPU client is synchronous and has one device, so executions run in the calling thread
and no threads start for devices that are never used. The GPU clients do not preallocate
most of the device memory, which JAX does by default.
"""

NVIDIA_LIBRARIES: Mapping[int, tuple[tuple[str, str], ...]] = {
    12: (
        ("cuda_runtime", "libcudart.so.12"),
        ("cuda_nvrtc", "libnvrtc.so.12"),
        ("cublas", "libcublas.so.12"),
        ("cublas", "libcublasLt.so.12"),
        ("nccl", "libnccl.so.2"),
        ("cuda_cupti", "libcupti.so.12"),
        ("cusparse", "libcusparse.so.12"),
        ("cusolver", "libcusolver.so.11"),
        ("cufft", "libcufft.so.11"),
        ("nvshmem", "libnvshmem_host.so.3"),
        ("cudnn", "libcudnn.so.9"),
    ),
    13: (
        ("cu13", "libcudart.so.13"),
        ("cu13", "libnvrtc.so.13"),
        ("cu13", "libcublas.so.13"),
        ("cu13", "libcublasLt.so.13"),
        ("nccl", "libnccl.so.2"),
        ("cu13", "libcupti.so.13"),
        ("cu13", "libcusparse.so.12"),
        ("cu13", "libcusolver.so.12"),
        ("cu13", "libcufft.so.12"),
        ("nvshmem", "libnvshmem_host.so.3"),
        ("cudnn", "libcudnn.so.9"),
    ),
}
"""Libraries of the `nvidia` wheels that the CUDA plugin of each major version loads.

JAX loads them before the plugin in the same way, because the plugin looks them up by soname
and the wheels do not put them on the loader's search path. The lists follow
`jax_plugins.xla_cuda12` of JAX 0.12, with the subdirectory of `nvidia` that holds each.
"""

JAX_PLUGIN_PATTERN = re.compile(r"xla_(?P<platform>cuda|rocm)(?P<major>\d+)")
"""Directory names of JAX's GPU plugins in the `jax_plugins` namespace."""


class PluginModule(Protocol):
    """A module that a `cyxla.plugins` entry point names."""

    PLATFORM: str
    """Platform that the plugin runs programs on."""
    CREATE_OPTIONS: Mapping[str, OptionValue]
    """Options of `PJRT_Client_Create` for this plugin."""

    def library_path(self) -> Path:
        """Returns the path of the plugin library."""
        ...


class PluginIdentity(NamedTuple):
    """What identifies the loaded plugin in cache keys."""

    platform: str
    path: Path
    size: int
    modified_ns: int


loaded_plugin: PluginIdentity | None = None
"""The plugin that `initialize` loaded, or `None` before it runs."""

initialization_lock = threading.RLock()
"""Held while a plugin loads, so threads that start at once load it only once."""


def namespace_directories(name: str) -> tuple[Path, ...]:
    """Returns the directories of a package without importing it, or none if it is missing."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.submodule_search_locations is None:
        return ()
    return tuple(Path(location) for location in spec.submodule_search_locations)


def entry_point_plugins() -> list[Plugin]:
    """Returns the plugins of distributions with a `cyxla.plugins` entry point.

    An entry point that fails to load is skipped with a warning, so one broken distribution
    does not hide the others.
    """
    plugins = []
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            module: PluginModule = entry_point.load()
            plugin = Plugin(
                platform=module.PLATFORM,
                path=Path(module.library_path()),
                source=entry_point.dist.name if entry_point.dist else entry_point.value,
                create_options=dict(module.CREATE_OPTIONS),
            )
        except Exception as error:
            warnings.warn(
                f"Skipping the PJRT plugin of entry point {entry_point.name}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        plugins.append(plugin)
    return plugins


def nvidia_libraries(major: int) -> tuple[Path, ...]:
    """Returns the CUDA libraries of a major version that the `nvidia` wheels installed."""
    directories = namespace_directories("nvidia")
    found = []
    for subdirectory, library in NVIDIA_LIBRARIES.get(major, ()):
        for directory in directories:
            path = directory / subdirectory / "lib" / library
            if path.exists():
                found.append(path)
                break
    return tuple(found)


def jax_gpu_plugins() -> list[Plugin]:
    """Returns the CUDA and ROCm plugins that JAX's `pjrt` wheels installed."""
    plugins = []
    for directory in namespace_directories("jax_plugins"):
        for child in sorted(directory.iterdir(), reverse=True):
            match JAX_PLUGIN_PATTERN.fullmatch(child.name):
                case None:
                    continue
                case found:
                    platform = found["platform"]
                    major = int(found["major"])
            library = child / f"xla_{platform}_plugin.so"
            if not library.exists():
                continue
            plugins.append(
                Plugin(
                    platform=platform,
                    path=library,
                    source=f"jax_plugins.{child.name}",
                    create_options=DEFAULT_CREATE_OPTIONS[platform],
                    preload=nvidia_libraries(major) if platform == "cuda" else (),
                )
            )
    return plugins


def tpu_plugins() -> list[Plugin]:
    """Returns the TPU plugin of `libtpu`, if it is installed."""
    for directory in namespace_directories("libtpu"):
        library = directory / "libtpu.so"
        if library.exists():
            return [Plugin(platform="tpu", path=library, source="libtpu")]
    return []


def available_plugins() -> tuple[Plugin, ...]:
    """Returns every installed plugin, in the order of the module docstring."""
    return (*entry_point_plugins(), *jax_gpu_plugins(), *tpu_plugins())


def find_plugin(platform: str | None = None) -> Plugin:
    """Returns the plugin to load for a platform.

    The path in `CYXLA_PJRT_PLUGIN` wins over installed plugins. Without a platform, the
    one in `CYXLA_PLATFORM` is used, and without that, the first installed platform in
    `PLATFORM_PREFERENCE`.
    """
    wanted = platform or os.environ.get(PLATFORM_ENVIRONMENT_VARIABLE) or None
    override = os.environ.get(PLUGIN_ENVIRONMENT_VARIABLE)
    if override:
        return plugin_at(Path(override), wanted or "cpu", PLUGIN_ENVIRONMENT_VARIABLE)
    plugins = available_plugins()
    if wanted is None:
        for candidate in PLATFORM_PREFERENCE:
            for plugin in plugins:
                if plugin.platform == candidate:
                    return plugin
        if plugins:
            return plugins[0]
        raise RuntimeError(
            "No PJRT plugin is installed. Install one, for example with "
            f"`pip install cyxla[cpu]`, or set {PLUGIN_ENVIRONMENT_VARIABLE}."
        )
    for plugin in plugins:
        if plugin.platform == wanted:
            return plugin
    installed = ", ".join(sorted({plugin.platform for plugin in plugins})) or "none"
    raise RuntimeError(
        f"No PJRT plugin for the platform {wanted} is installed. Installed: {installed}."
    )


def plugin_at(path: Path, platform: str, source: str) -> Plugin:
    """Returns a plugin at a path, with the default options of its platform."""
    return Plugin(
        platform=platform,
        path=path,
        source=source,
        create_options=DEFAULT_CREATE_OPTIONS.get(platform, {}),
    )


def initialize(
    plugin: Plugin | Path | None = None,
    *,
    platform: str | None = None,
    options: Mapping[str, OptionValue] | None = None,
    jax_targets: bool = True,
) -> RuntimeInfo:
    """Loads a PJRT plugin and returns details of its runtime.

    Calling it is optional, as the first function that needs a plugin loads the default one.
    It is for choosing the plugin or its options, and has to come before that first use.

    Without a plugin, `find_plugin` picks one for `platform`. A path is loaded with the
    default options of `platform`, which is `cpu` if not given. `options` add to or replace
    the plugin's client options. A process loads one plugin, so later calls return the
    details of the first.

    With `jax_targets` the FFI handlers that jaxlib ships are registered with the plugin,
    which `jnp.linalg` programs need, and so are the targets registered earlier with
    `register_ffi_target`. Pass `False` to register none.
    """
    with initialization_lock:
        if _runtime.is_initialized():
            return _runtime.runtime_info()
        return load_plugin(plugin, platform, options, jax_targets)


def require_platform(platform: str | None) -> None:
    """Loads the plugin for a platform unless one is loaded, and checks that it runs it.

    Without a platform any plugin will do, so the default one loads. The loaded plugin
    stays for the whole process, so a program for another platform cannot run.
    """
    if platform is None:
        _runtime.ensure_initialized()
        return
    info = initialize(platform=platform)
    if info.platform.lower() != platform.lower():
        raise RuntimeError(
            f"The program was lowered for {platform}, but the loaded plugin runs "
            f"{info.platform}. A process loads one plugin."
        )


def load_plugin(
    plugin: Plugin | Path | None,
    platform: str | None,
    options: Mapping[str, OptionValue] | None,
    jax_targets: bool,
) -> RuntimeInfo:
    """Loads a plugin with the arguments of `initialize`, which holds the lock."""
    global loaded_plugin
    match plugin:
        case None:
            chosen = find_plugin(platform)
        case Path():
            chosen = plugin_at(plugin, platform or "cpu", str(plugin))
        case _:
            chosen = plugin
    for library in chosen.preload:
        ctypes.CDLL(str(library))
    merged = {**chosen.create_options, **(options or {})}
    path = chosen.path.resolve()
    info = _runtime.initialize(str(path), tuple(merged.items()))
    _ffi.apply_targets()
    if jax_targets:
        _ffi.register_jax_targets()
    status = path.stat()
    loaded_plugin = PluginIdentity(
        info.platform, path, status.st_size, status.st_mtime_ns
    )
    return info
