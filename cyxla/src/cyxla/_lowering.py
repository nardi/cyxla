"""Lowering of JAX functions to StableHLO bytecode that a PJRT plugin compiles."""

import contextlib
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import numpy as np

from cyxla import _plugins, _runtime
from cyxla._types import ArrayType

WIDE_DTYPES = frozenset(
    np.dtype(name) for name in ("float64", "int64", "uint64", "complex128")
)
"""The dtypes that JAX holds as 32-bit types unless its 64-bit mode is on."""

LOWERING_PLATFORMS = frozenset({"cpu", "cuda", "rocm", "tpu"})
"""Platforms that `jax.export` lowers for, which are also the names the plugins report."""


class LoweredProgram(NamedTuple):
    """A program as StableHLO."""

    code: bytes
    """Portable StableHLO bytecode, which the PJRT plugin compiles."""
    text: str
    """The module as text, from which the executable cache key is computed."""
    input_types: tuple[ArrayType, ...] | None = None
    """The type of each argument, or `None` when it is not known."""


def lowering_platform(platform: str | None) -> str:
    """Returns the platform to lower for.

    Without a platform this is the platform of the loaded plugin. Before one is loaded it is
    the platform that `initialize` would load, or `cpu` if no plugin is installed. Lowering
    never loads a plugin itself.
    """
    if platform is None:
        if _runtime.is_initialized():
            platform = _runtime.runtime_info().platform
        else:
            try:
                platform = _plugins.find_plugin().platform
            except RuntimeError:
                platform = "cpu"
    name = platform.lower()
    if name not in LOWERING_PLATFORMS:
        known = ", ".join(sorted(LOWERING_PLATFORMS))
        raise ValueError(f"Cannot lower for the platform {platform}, only for {known}.")
    return name


def lower(
    function: Callable[..., Any],
    shapes: Sequence[jax.ShapeDtypeStruct],
    platform: str | None = None,
    custom_calls: Sequence[str] = (),
) -> LoweredProgram:
    """Lowers a function to StableHLO for a platform, by default the loaded plugin's.

    `function` is typed `Callable[..., Any]` because it may take and return any arrays that
    JAX traces. Unused arguments stay in the signature, since callers pass every input of a
    program. `jax.export` refuses custom calls that it knows no compatibility guarantees
    for, such as the targets of third-party libraries, so `custom_calls` names the targets
    that the function may call. The kernels of `jnp.linalg` need no entry.

    A function with an argument of a 64-bit type (`float64`, `int64`, `uint64` or
    `complex128`) is traced in JAX's 64-bit mode, whatever the setting of the process is,
    because JAX would turn the argument into a 32-bit type otherwise. The setting is
    restored afterwards. A function that creates 64-bit values out of 32-bit arguments
    needs `enable_x64` instead.
    """
    wide = any(is_wide(shape) for shape in shapes)
    try:
        with jax.enable_x64(True) if wide else contextlib.nullcontext():
            exported = jax.export.export(
                jax.jit(function, keep_unused=True),
                platforms=[lowering_platform(platform)],
                disabled_checks=[
                    jax.export.DisabledSafetyCheck.custom_call(name)
                    for name in custom_calls
                ],
            )(*shapes)
    except ValueError as error:
        if "no compatibility guarantees" not in str(error):
            raise
        raise ValueError(
            f"{error}\nName the custom calls that the function may run in the "
            "`custom_calls` argument of `cyxla.lower`."
        ) from error
    input_types = tuple(array_type(aval) for aval in exported.in_avals)
    for index, aval in enumerate(exported.out_avals):
        array_type(aval, f"The result {index}")
    return LoweredProgram(
        code=bytes(exported.mlir_module_serialized),
        text=exported.mlir_module(),
        input_types=input_types,
    )


def is_wide(shape: jax.ShapeDtypeStruct) -> bool:
    """Returns whether an argument has a 64-bit dtype, which is not a key or other extended type."""
    try:
        return np.dtype(shape.dtype) in WIDE_DTYPES
    except TypeError:
        return False


def array_type(aval: Any, subject: str = "An argument") -> ArrayType:
    """Returns the `ArrayType` of an abstract value, or raises if the runtime cannot hold it.

    The abstract value is typed `Any` because `jax.export` exposes it as an internal class.
    """
    try:
        dtype = np.dtype(aval.dtype)
        shape = tuple(int(dimension) for dimension in aval.shape)
    except (TypeError, ValueError, jax.errors.InconclusiveDimensionOperation) as error:
        raise ValueError(
            f"{subject} has the type {aval}, which has no dtype or shape that the runtime "
            "can hold."
        ) from error
    if dtype not in _runtime.CODES:
        supported = ", ".join(sorted(str(known) for known in _runtime.CODES))
        raise ValueError(
            f"{subject} has the dtype {dtype}, but the runtime supports only {supported}."
        )
    return ArrayType(dtype, shape)
