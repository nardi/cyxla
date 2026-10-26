"""Records shared by the extension module and the Python layer."""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

OptionValue = bool | int | float | str | tuple[int, ...]
"""A value of a `PJRT_Client_Create` option, one of the types of `PJRT_NamedValue`."""


class RuntimeInfo(NamedTuple):
    """Details of the PJRT runtime."""

    api_major_version: int
    api_minor_version: int
    platform: str
    """Platform name that the plugin reports, such as `cpu` or `cuda`."""
    aliased_inputs: bool
    """Whether input buffers share host memory, so filling an input needs no copy."""
    inplace_outputs: bool
    """Whether outputs are read from the buffers' memory instead of copied to the host."""


class ErrorInfo(NamedTuple):
    """PJRT errors recorded since the runtime was initialized."""

    count: int
    last_message: str


class Plugin(NamedTuple):
    """A PJRT plugin library and the options to create its client with."""

    platform: str
    """Platform that the plugin runs programs on, which is also the JAX lowering platform."""
    path: Path
    """Path of the plugin library."""
    source: str
    """Where the plugin was found, such as the distribution that installed it."""
    create_options: Mapping[str, OptionValue] = MappingProxyType({})
    """Options of `PJRT_Client_Create`, which `initialize` merges with the caller's."""
    preload: tuple[Path, ...] = ()
    """Libraries to load before the plugin, so the dynamic loader finds them by soname."""
