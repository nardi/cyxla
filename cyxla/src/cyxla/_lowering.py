"""Lowering of JAX functions to StableHLO bytecode that a PJRT plugin compiles."""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import numpy as np

from cyxla import _plugins, _runtime
from cyxla._types import ArrayType

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
    platform: str | None = None
    """The platform that the program was lowered for, or `None` when it is not known."""


def jax_platform_name(name: str) -> str:
    """Returns the platform of cyxla for a platform name of JAX.

    JAX calls both CUDA and ROCm devices `gpu`, which maps to the one that has an installed
    plugin, and to `cuda` if both or neither have.
    """
    lowered = name.lower()
    if lowered != "gpu":
        return lowered
    installed = {plugin.platform for plugin in _plugins.available_plugins()}
    return "rocm" if "rocm" in installed and "cuda" not in installed else "cuda"


def placement_platform(shapes: Sequence[Any]) -> str | None:
    """Returns the platform of the devices that the arguments are placed on, if any.

    Arguments are shapes with a sharding, or arrays. An array that JAX has not committed to a
    device goes wherever the computation does, so it says nothing. The arguments are typed
    `Any` because they only need a `sharding` and a `committed` attribute.
    """
    platforms = set()
    for shape in shapes:
        sharding = getattr(shape, "sharding", None)
        if sharding is None or not getattr(shape, "committed", True):
            continue
        devices = sharding.device_set
        if len(devices) > 1:
            raise ValueError(
                "A program runs on one device, but an argument is placed on "
                f"{len(devices)} devices."
            )
        platforms.update(jax_platform_name(device.platform) for device in devices)
    if len(platforms) > 1:
        raise ValueError(
            "The arguments are placed on different platforms: "
            f"{', '.join(sorted(platforms))}."
        )
    return next(iter(platforms), None)


def default_device_platform() -> str | None:
    """Returns the platform of the default device of JAX, if one is set.

    This is what `jax.default_device` sets, to a device or to a platform name. Without it
    JAX picks a backend itself, which cyxla does not start.
    """
    device = getattr(jax.config, "jax_default_device")
    if device is None:
        return None
    return jax_platform_name(device if isinstance(device, str) else device.platform)


def configured_platform() -> str | None:
    """Returns the first platform of `jax_platforms` that has an installed plugin."""
    configured = getattr(jax.config, "jax_platforms") or ""
    for name in filter(None, (part.strip() for part in configured.split(","))):
        candidate = jax_platform_name(name)
        try:
            _plugins.find_plugin(candidate)
        except RuntimeError:
            continue
        return candidate
    return None


def lowering_platform(platform: str | None, shapes: Sequence[Any] = ()) -> str:
    """Returns the platform to lower for.

    An explicit platform wins. Otherwise the choice follows JAX where it can. It is the
    platform of the devices that the arguments are placed on, then of the default device of
    JAX, then of the loaded plugin, then the first platform in `jax_platforms` that has a
    plugin, and last the platform that `initialize` would load, or `cpu` if no plugin is
    installed. Lowering never loads a plugin itself.
    """
    if platform is None:
        platform = placement_platform(shapes) or default_device_platform()
    if platform is None and _runtime.is_initialized():
        platform = _runtime.runtime_info().platform
    if platform is None:
        platform = configured_platform()
    if platform is None:
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
    shapes: Sequence[jax.ShapeDtypeStruct | jax.Array],
    platform: str | None = None,
    custom_calls: Sequence[str] = (),
) -> LoweredProgram:
    """Lowers a function to StableHLO for a platform, which by default follows JAX.

    `shapes` are the arguments as shapes, or as arrays. Their placement and the default
    device of JAX choose the platform as `lowering_platform` describes, unless `platform`
    names one. The program records the platform, and `compile_program` loads the plugin for
    it.

    `function` is typed `Callable[..., Any]` because it may take and return any arrays that
    JAX traces. Unused arguments stay in the signature, since callers pass every input of a
    program. `jax.export` refuses custom calls that it knows no compatibility guarantees
    for, such as the targets of third-party libraries, so `custom_calls` names the targets
    that the function may call. The kernels of `jnp.linalg` need no entry.

    The function is traced with the current setting of `jax.enable_x64`, so 64-bit
    arguments and values need that setting on, as they do in JAX.
    """
    target = lowering_platform(platform, shapes)
    try:
        exported = jax.export.export(
            jax.jit(function, keep_unused=True),
            platforms=[target],
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
        platform=target,
    )


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
