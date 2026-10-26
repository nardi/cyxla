"""A bridge from JAX to native code through the PJRT C API.

cyxla lowers JAX functions to StableHLO, compiles them with a PJRT plugin, and runs them from
extension modules of other packages without the GIL. Those modules `cimport` the
declarations in `cyxla/_runtime.pxd`, and `get_include` returns the directories that their
build needs.
"""

import importlib.metadata
from pathlib import Path

from cyxla._cache import (
    CACHE_ENVIRONMENT_VARIABLE,
    CompileStatistics,
    cache_directory,
    cache_key,
    compile_options,
    compile_program,
    enable_x64,
    library_identity,
    load_lowered,
    store_lowered,
)
from cyxla._lowering import LoweredProgram, lower
from cyxla._plugins import (
    PLATFORM_ENVIRONMENT_VARIABLE,
    PLUGIN_ENVIRONMENT_VARIABLE,
    PluginModule,
    available_plugins,
    find_plugin,
    initialize,
)
from cyxla._runtime import (
    Executable,
    abi_version,
    error_info,
    is_initialized,
    runtime_info,
)
from cyxla._types import ErrorInfo, OptionValue, Plugin, RuntimeInfo

__version__ = importlib.metadata.version("cyxla")

ABI_VERSION = abi_version()
"""Version of the declarations in `_runtime.pxd` that this installation was compiled with."""


def check_abi(compiled_version: int) -> None:
    """Raises an `ImportError` unless a module was compiled against these declarations.

    An extension module passes the `ABI_VERSION` that it cimported from `cyxla._runtime`
    when it is imported. Cython checks the signatures of shared functions itself, but not
    the layout of structs such as `ProgramSpec`.
    """
    if compiled_version != ABI_VERSION:
        raise ImportError(
            f"The module was compiled against version {compiled_version} of the cyxla "
            f"declarations, but cyxla {__version__} has version {ABI_VERSION}. Rebuild it."
        )


def get_include() -> list[str]:
    """Returns the directories that an extension module needs to build against cyxla.

    The first holds the PJRT C API header, for the C compiler. The second is the parent of
    the package, in which Cython finds `cyxla/_runtime.pxd`.
    """
    package = Path(__file__).parent
    return [str(package / "include"), str(package.parent)]


__all__ = [
    "ABI_VERSION",
    "CACHE_ENVIRONMENT_VARIABLE",
    "PLATFORM_ENVIRONMENT_VARIABLE",
    "PLUGIN_ENVIRONMENT_VARIABLE",
    "CompileStatistics",
    "ErrorInfo",
    "Executable",
    "LoweredProgram",
    "OptionValue",
    "Plugin",
    "PluginModule",
    "RuntimeInfo",
    "available_plugins",
    "cache_directory",
    "cache_key",
    "check_abi",
    "compile_options",
    "compile_program",
    "enable_x64",
    "error_info",
    "find_plugin",
    "get_include",
    "initialize",
    "is_initialized",
    "library_identity",
    "load_lowered",
    "lower",
    "runtime_info",
    "store_lowered",
]
