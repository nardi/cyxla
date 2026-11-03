"""Native functions, from Python: registering them and calling them in programs.

A native function is a C function of the type `CyxlaNativeFunction` (see `cyxla/native.h`).
It is registered under a name, and a program calls it with `native_call`. The program holds
the name in the attribute `name` of a call of the target `cyxla_native_function`, so it does
not depend on the process that lowered it. A process that loads such a program only has to
register the same function under the same name, which the library that provides it does when
it is imported.

The handler uses nothing but the call frame, so it is registered with the plugin and with
the runtime of JAX, and `jax.jit` runs the same functions without cyxla.
"""

import ctypes
from collections.abc import Sequence
from typing import Any, NamedTuple

from cyxla import _ffi, _native, _targets

NATIVE_ABI_VERSION = _native.native_abi_version()
"""The version of the native interface that this build of cyxla has."""


class NativeFunctionInfo(NamedTuple):
    """The record that cyxla keeps of a registered native function."""

    name: str
    function_address: int
    user_data_address: int
    provider: str | None
    """The package that registered the function, if it could be told."""


native_functions: dict[str, NativeFunctionInfo] = {}
"""The registered functions by name, which the programs of this process may call."""

handler_function: Any = None
"""The handler as a `ctypes` function, which the capsule for JAX refers to."""

registered_with_plugin = False
"""Whether the handler is registered with the plugin."""

registered_with_jax = False
"""Whether the handler is registered with the runtime of JAX."""


def record_registration(name: str) -> None:
    """Notes the provider of a function that was just registered, called by the native module."""
    existing = native_functions.get(name)
    provider = _ffi.calling_package()
    native_functions[name] = NativeFunctionInfo(
        name,
        0 if existing is None else existing.function_address,
        0 if existing is None else existing.user_data_address,
        provider,
    )


_native.set_registration_hook(record_registration)


def register_handler(with_jax: bool = False) -> None:
    """Registers the handler for native functions with the plugin, and with JAX if asked.

    Registering with JAX imports it, which a process that only loads programs avoids, so
    `native_call` is where that happens. Calling the function again does nothing.
    """
    global handler_function, registered_with_plugin, registered_with_jax
    address = _native.handler_address()
    if handler_function is None:
        handler_function = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(address)
    if not registered_with_plugin:
        _ffi.add_target(_targets.NATIVE_TARGET, address, "cpu", False, "cyxla")
        registered_with_plugin = True
    if with_jax and not registered_with_jax:
        import jax

        jax.ffi.register_ffi_target(
            _targets.NATIVE_TARGET, jax.ffi.pycapsule(handler_function), platform="cpu"
        )
        registered_with_jax = True


def register_native_function(
    name: str,
    function: int,
    user_data: int = 0,
    *,
    abi_version: int = NATIVE_ABI_VERSION,
) -> None:
    """Registers a native function under a name, given by the address of the C function.

    `user_data` is the address of state that the function receives, or 0. The memory has to
    stay valid for the life of the process. The function runs without the GIL on the threads
    of the runtime, so it has to be thread safe, and it should be registered before any
    program calls it, since the table that the handler reads is replaced as a whole. Another
    function under the name of a registered one raises an error, and the same function again
    does nothing. `abi_version` is the `CYXLA_NATIVE_ABI_VERSION` that the function was
    compiled against. Cython code registers with the typed `register_native` of
    `cyxla._native` instead, which checks the signature of the function when it compiles.
    """
    _native.register_native_address(name, function, user_data, abi_version)
    existing = native_functions[name]
    native_functions[name] = existing._replace(
        function_address=function, user_data_address=user_data
    )
    register_handler()


def native_call(
    name: str,
    results: Any,
    *arguments: Any,
    vmap_method: str | None = None,
    has_side_effect: bool = False,
    **attributes: Any,
) -> Any:
    """Calls the native function registered under `name` and returns its results.

    `results` are the `jax.ShapeDtypeStruct`s of the results, as for `jax.ffi.ffi_call`, and
    so are `vmap_method` and `has_side_effect`. The keyword arguments become the attributes
    of the call, which can be numbers or strings, and the function reads them. The call
    works in cyxla and in plain JAX. It has no derivative. A function that needs one is
    wrapped in `jax.custom_jvp` or `jax.custom_vjp`, or made a primitive with
    `native_primitive`. The arguments are typed `Any` because they are arrays of any shape
    and type that the function takes.
    """
    import jax

    register_handler(with_jax=True)
    call = jax.ffi.ffi_call(
        _targets.NATIVE_TARGET,
        results,
        vmap_method=vmap_method,
        has_side_effect=has_side_effect,
    )
    return call(*arguments, name=name, **attributes)


def registered_native_functions() -> Sequence[NativeFunctionInfo]:
    """Returns what is known about the functions registered in this process, by name."""
    return tuple(native_functions[name] for name in sorted(native_functions))
