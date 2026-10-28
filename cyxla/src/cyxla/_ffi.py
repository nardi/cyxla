"""FFI handlers, which are the native functions that custom calls in programs run.

A program contains a custom call when a JAX primitive lowers to one. The linear algebra of
`jnp.linalg` does on the CPU, where its LAPACK kernels ship in jaxlib, and third-party
libraries add their own through `jax.ffi`. A PJRT plugin is a separate build of XLA that does
not know any of them, so each handler has to be registered with the plugin by name before a
program compiles. The plugin accepts typed FFI handlers only, and only their `execute` stage.
"""

import contextlib
import ctypes
import importlib
import os
import warnings
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from cyxla import _runtime

COMMAND_BUFFER_COMPATIBLE = 1
"""Handler trait that allows capturing the handler into a GPU command buffer."""

MIRROR_ENVIRONMENT_VARIABLE = "CYXLA_MIRROR_JAX"
"""Environment variable that turns the mirror of JAX's registrations off when it is `0`."""

JAXLIB_MODULES = (
    "jaxlib.lapack",
    "jaxlib.cpu_sparse",
    "jaxlib.gpu_solver",
    "jaxlib.gpu_sparse",
    "jaxlib.gpu_linalg",
    "jaxlib.gpu_prng",
)
"""Modules of jaxlib whose `registrations` function lists the handlers that JAX registers."""

PLATFORM_ALIASES = {"host": "cpu", "gpu": "cuda"}
"""Platform names of XLA that map to the names of the plugins."""

PLATFORM_SPELLINGS = {"cpu": "cpu", "cuda": "CUDA", "rocm": "ROCM"}
"""Spellings of platform names that XLA's FFI registry takes, as JAX passes them."""


class Target(NamedTuple):
    """A handler registered under a name for a platform."""

    name: str
    """Name that the custom call in the program refers to."""
    handler: int
    """Address of the `XLA_FFI_Handler`."""
    platform: str
    """Lower-case platform name, as in `RuntimeInfo.platform`."""
    traits: int
    """Handler traits as a bit set."""


targets: dict[tuple[str, str], Target] = {}
"""Every registered target by platform and name, including those waiting for a plugin."""

applied: set[tuple[str, str]] = set()
"""Keys of the targets that the loaded plugin knows."""

jax_registration: Callable[..., None] | None = None
"""The registration function of jaxlib that `mirror_jax_registrations` replaced."""


def platform_key(platform: str) -> str:
    """Returns the lower-case name that identifies a platform, whichever spelling is given."""
    name = platform.lower()
    return PLATFORM_ALIASES.get(name, name)


def capsule_address(capsule: object) -> int:
    """Returns the pointer inside a `PyCapsule`, whatever the name of the capsule is."""
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.restype = ctypes.c_char_p
    get_name.argtypes = [ctypes.py_object]
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return get_pointer(capsule, get_name(capsule)) or 0


def handler_address(handler: Any) -> int:
    """Returns the address of the `XLA_FFI_Handler` that a registration refers to.

    `handler` may be the address as an `int`, a `PyCapsule` that holds it, or a mapping of
    stage names to either, as `jax.ffi.register_ffi_target` takes. The plugin runs only the
    `execute` stage, so a handler with other stages is refused. The argument is typed `Any`
    because those forms have no common type.
    """
    match handler:
        case int():
            address = handler
        case {"execute": execute, **stages}:
            used = sorted(stage for stage, value in stages.items() if value is not None)
            if used:
                raise ValueError(
                    f"The PJRT plugin runs only the execute stage of a handler, "
                    f"not {', '.join(used)}."
                )
            return handler_address(execute)
        case Mapping():
            raise ValueError("A handler with stages needs an execute stage.")
        case _ if type(handler).__name__ == "PyCapsule":
            address = capsule_address(handler)
        case _:
            raise TypeError(
                "A handler is an address, a PyCapsule or a mapping with an execute stage, "
                f"not {type(handler).__name__}."
            )
    if address == 0:
        raise ValueError("The handler is a null pointer.")
    return address


def apply_targets() -> int:
    """Registers the waiting targets of the loaded plugin's platform and returns their count."""
    if not _runtime.is_initialized():
        return 0
    platform = platform_key(_runtime.runtime_info().platform)
    count = 0
    for key, target in list(targets.items()):
        if target.platform == platform and key not in applied:
            _runtime.register_ffi_handler(
                target.name,
                target.handler,
                PLATFORM_SPELLINGS.get(platform, platform),
                target.traits,
            )
            applied.add(key)
            count += 1
    return count


def register_ffi_target(
    name: str,
    handler: Any,
    platform: str = "cpu",
    *,
    command_buffer_compatible: bool = False,
) -> None:
    """Registers an FFI handler under a name, for programs that call that target.

    The signature follows `jax.ffi.register_ffi_target`, so a library can pass the same
    arguments to both. Registering twice with the same handler does nothing, and with
    another handler raises an error, as the plugin keeps a registration for the whole
    process. A handler for a platform other than the plugin's waits until a plugin for that
    platform loads. The argument is typed `Any` for the reason given at `handler_address`.
    """
    key = (platform_key(platform), name)
    address = handler_address(handler)
    traits = COMMAND_BUFFER_COMPATIBLE if command_buffer_compatible else 0
    existing = targets.get(key)
    if existing is not None:
        if existing.handler != address:
            raise ValueError(
                f"The FFI target {name} on {key[0]} is registered with another handler."
            )
        return
    targets[key] = Target(name, address, key[0], traits)
    apply_targets()


def registry_targets(platform: str) -> list[tuple[str, Any, bool]]:
    """Returns the typed handlers in the registry of JAX for a platform.

    Each entry is a name, a handler and whether the handler is command buffer compatible.
    JAX keeps a registration in one of two places. Until the backend of the platform starts,
    `jaxlib.xla_client` queues it together with its traits. Afterwards it sits in the
    registry of jaxlib, which lists typed handlers as mappings of stages. A bare value would
    be an untyped handler and is skipped. Both places are private to jaxlib, so a version
    of jaxlib without them contributes nothing. The handler is typed `Any` for the reason given at `handler_address`.
    """
    from jaxlib import xla_client

    wanted = platform_key(platform)
    found: dict[str, tuple[Any, bool]] = {}
    queued: dict[str, list[tuple[str, Any, int, Any]]] = getattr(
        xla_client, "_custom_callback", {}
    )
    for queued_platform, entries in list(queued.items()):
        if platform_key(queued_platform) != wanted:
            continue
        for name, handler, api_version, traits in list(entries):
            if api_version >= 1:
                compatible = bool(int(traits) & COMMAND_BUFFER_COMPATIBLE)
                found[name] = (handler, compatible)
    with contextlib.suppress(ImportError, AttributeError):
        registry = importlib.import_module("jaxlib._jax")
        spelling = PLATFORM_SPELLINGS.get(wanted, wanted)
        for name, handler in registry.custom_call_targets(spelling).items():
            if isinstance(handler, Mapping):
                found.setdefault(name, (handler, False))
    return [
        (name, handler, compatible) for name, (handler, compatible) in found.items()
    ]


def jax_targets(platform: str) -> list[tuple[str, Any, bool]]:
    """Returns the typed handlers that JAX knows for a platform.

    These are the handlers in the registry of JAX, such as the ones that libraries
    registered with `jax.ffi.register_ffi_target`, and the ones that the modules of jaxlib
    ship. Each entry is a name, a handler and whether the handler is command buffer
    compatible. The handler is typed `Any` because it is an opaque object of the C API of
    Python.
    """
    wanted = platform_key(platform)
    found: dict[str, tuple[Any, bool]] = {}
    for module_name in JAXLIB_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        registrations: Callable[[], dict[str, list[tuple[str, Any, int]]]] | None = (
            getattr(module, "registrations", None)
        )
        if registrations is None:
            continue
        for module_platform, entries in registrations().items():
            if platform_key(module_platform) != wanted:
                continue
            for name, value, version in entries:
                if version >= 1:
                    found[name] = (value, False)
    found.update(
        {
            name: (handler, compatible)
            for name, handler, compatible in registry_targets(wanted)
        }
    )
    return [
        (name, handler, compatible) for name, (handler, compatible) in found.items()
    ]


def register_jax_targets(platform: str | None = None) -> int:
    """Registers the handlers that JAX knows, and returns how many were new.

    This covers the LAPACK kernels of jaxlib, which make `jnp.linalg` programs compile on
    the CPU, and everything else in the registry of JAX, such as the handlers of libraries
    that registered with `jax.ffi.register_ffi_target` before cyxla was imported. It also
    loads the LAPACK functions of SciPy that the kernels call. Handlers of the older untyped
    interface are left out, as the plugin cannot run them, and so are handlers that the
    plugin cannot take, which raise a warning. A plugin without FFI support gets none.
    """
    _runtime.ensure_initialized()
    if not _runtime.supports_ffi():
        return 0
    chosen = platform_key(platform or _runtime.runtime_info().platform)
    if chosen == "cpu":
        with contextlib.suppress(ImportError):
            importlib.import_module("jaxlib.cpu._lapack").initialize()
    before = len(targets)
    for name, handler, compatible in jax_targets(chosen):
        register_mirrored(name, handler, chosen, compatible)
    return len(targets) - before


def register_mirrored(
    name: str, handler: Any, platform: str, command_buffer_compatible: bool
) -> None:
    """Registers a handler of JAX, with a warning instead of an error if it cannot run.

    The handler is typed `Any` for the reason given at `handler_address`.
    """
    try:
        register_ffi_target(
            name,
            handler,
            platform,
            command_buffer_compatible=command_buffer_compatible,
        )
    except (ValueError, TypeError) as error:
        warnings.warn(
            f"The FFI target {name} is not available to cyxla: {error}",
            stacklevel=3,
        )


def mirror_jax_registrations() -> None:
    """Registers every typed handler that libraries register with JAX from now on.

    A library that supports JAX's FFI calls `jax.ffi.register_ffi_target` when it is
    imported. After this call, such a handler is also registered with the PJRT plugin, so
    the library works in programs that cyxla compiles without changes. Handlers of the
    older untyped interface are not covered. Registrations from before the call are not
    seen here, but loading the plugin registers them from the registry of JAX. Importing
    cyxla calls this unless `CYXLA_MIRROR_JAX` is `0`, and calling it again does nothing.
    """
    global jax_registration
    from jaxlib import xla_client

    if jax_registration is not None:
        return
    jax_registration = xla_client.register_custom_call_target
    original = jax_registration

    def mirrored(
        name: str,
        fn: Any,
        platform: str = "cpu",
        api_version: int = 0,
        traits: xla_client.CustomCallTargetTraits = xla_client.CustomCallTargetTraits.DEFAULT,
    ) -> None:
        original(name, fn, platform, api_version, traits)
        if api_version >= 1:
            register_mirrored(
                name, fn, platform, bool(int(traits) & COMMAND_BUFFER_COMPATIBLE)
            )

    # The module dictionary is written, because the type checker forbids assigning to a
    # function of a module.
    vars(xla_client)["register_custom_call_target"] = mirrored


def mirror_by_default() -> None:
    """Mirrors the registrations of JAX unless the environment turns that off."""
    if os.environ.get(MIRROR_ENVIRONMENT_VARIABLE) != "0":
        mirror_jax_registrations()
