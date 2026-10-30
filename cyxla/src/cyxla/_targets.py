"""Which machines can use a serialized executable, and which custom calls a program makes."""

import platform
import re
import sys
from typing import NamedTuple

from cyxla import _plugins, _runtime

CUSTOM_CALL_PATTERN = re.compile(r"custom_call\s+@([\w.$-]+)")
"""A custom call in the text of a StableHLO module, with the name of its target."""


class TargetFingerprint(NamedTuple):
    """What an executable depends on: the plugin that compiled it and the machine it ran on.

    The plugin compiles for the features of its host CPU and checks them when it loads an
    executable. Two machines with the same fingerprint usually accept each other's
    executables, and when one does not, loading fails with an error that the callers of
    `load_program` handle by compiling the StableHLO again.
    """

    platform: str
    """The platform that the plugin runs programs on, such as `cpu`."""
    system: str
    """The operating system, as `sys.platform` names it."""
    machine: str
    """The processor architecture, such as `x86_64` or `arm64`."""
    plugin: str
    """The distribution that provided the plugin with its version, or the file name."""
    plugin_size: int
    """The size of the plugin library in bytes, which tells builds of one version apart."""
    api_version: str
    """The version of the PJRT C API that the plugin implements."""

    @property
    def key(self) -> str:
        """Returns the fingerprint as one string, under which a file stores an executable."""
        return "|".join(str(field) for field in self)


def target_fingerprint() -> TargetFingerprint:
    """Returns the fingerprint of the loaded plugin on this machine.

    It loads the default plugin if none is loaded yet.
    """
    _plugins.require_platform(None)
    identity = _plugins.loaded_plugin
    if identity is None:  # pragma: no cover
        raise RuntimeError("The loaded PJRT plugin is not recorded.")
    info = _runtime.runtime_info()
    name = (
        f"{identity.source} {identity.version}".strip()
        if identity.version
        else identity.path.name
    )
    return TargetFingerprint(
        platform=identity.platform,
        system=sys.platform,
        machine=platform.machine().lower(),
        plugin=name,
        plugin_size=identity.size,
        api_version=f"{info.api_major_version}.{info.api_minor_version}",
    )


def custom_call_targets(text: str) -> tuple[str, ...]:
    """Returns the targets of the custom calls in the text of a StableHLO module, sorted.

    A process that loads programs with custom calls, such as the LAPACK kernels of
    `jnp.linalg`, has to register their handlers first, which `initialize` does unless it
    gets `jax_targets=False`. Programs without custom calls can skip that.
    """
    return tuple(sorted(set(CUSTOM_CALL_PATTERN.findall(text))))
