"""Lowering of JAX functions to StableHLO bytecode that a PJRT plugin compiles."""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax

from cyxla import _runtime

LOWERING_PLATFORMS = frozenset({"cpu", "cuda", "rocm", "tpu"})
"""Platforms that `jax.export` lowers for, which are also the names the plugins report."""


class LoweredProgram(NamedTuple):
    """A program as StableHLO."""

    code: bytes
    """Portable StableHLO bytecode, which the PJRT plugin compiles."""
    text: str
    """The module as text, from which the executable cache key is computed."""


def lowering_platform(platform: str | None) -> str:
    """Returns the platform to lower for.

    Without a platform this is the platform of the loaded plugin, or `cpu` before one is
    loaded.
    """
    if platform is None:
        if not _runtime.is_initialized():
            return "cpu"
        platform = _runtime.runtime_info().platform
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
    """
    try:
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
    return LoweredProgram(
        code=bytes(exported.mlir_module_serialized), text=exported.mlir_module()
    )
