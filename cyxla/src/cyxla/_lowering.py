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
) -> LoweredProgram:
    """Lowers a function to StableHLO for a platform, by default the loaded plugin's.

    `function` is typed `Callable[..., Any]` because it may take and return any arrays that
    JAX traces. Unused arguments stay in the signature, since callers pass every input of a
    program.
    """
    exported = jax.export.export(
        jax.jit(function, keep_unused=True), platforms=[lowering_platform(platform)]
    )(*shapes)
    return LoweredProgram(
        code=bytes(exported.mlir_module_serialized), text=exported.mlir_module()
    )
