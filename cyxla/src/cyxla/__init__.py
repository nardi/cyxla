"""A bridge from JAX to native code through the PJRT C API.

cyxla lowers JAX functions to StableHLO, compiles them with a PJRT plugin, and runs them from
extension modules of other packages without the GIL. Those modules `cimport` the
declarations in `cyxla/__init__.pxd`, and `get_include` returns the directories that their
build needs.
"""

import importlib.metadata
from pathlib import Path

from cyxla._bundle import Bundle, BundleError, BundleWriter
from cyxla._cache import (
    CACHE_ENVIRONMENT_VARIABLE,
    CompileStatistics,
    cache_directory,
    cache_key,
    compile_options,
    compile_program,
    library_identity,
    load_lowered,
    store_lowered,
)
from cyxla._codec import decode_record, encode_record
from cyxla._ffi import (
    MIRROR_ENVIRONMENT_VARIABLE,
    mirror_by_default,
    mirror_jax_registrations,
    register_ffi_target,
    register_jax_targets,
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
from cyxla._records import (
    TargetMismatchError,
    load_program,
    pack_program,
    record_custom_calls,
)
from cyxla._runtime import (
    Executable,
    abi_version,
    error_info,
    is_initialized,
    raise_last_error,
    runtime_info,
)
from cyxla._targets import (
    TargetFingerprint,
    custom_call_targets,
    override_target,
    target_fingerprint,
)
from cyxla._trees import (
    LeafInfo,
    check_fingerprint,
    decode_fingerprint,
    encode_fingerprint,
    tree_fingerprint,
)
from cyxla._types import ArrayType, ErrorInfo, OptionValue, Plugin, RuntimeInfo

__version__ = importlib.metadata.version("cyxla")

mirror_by_default()

ABI_VERSION = abi_version()
"""Version of the declarations in `_runtime.pxd` that this installation was compiled with."""


def check_abi(compiled_version: int) -> None:
    """Raises an `ImportError` unless a module was compiled against these declarations.

    An extension module passes the `ABI_VERSION` that it cimported from `cyxla` when it is
    imported. Cython checks the signatures of shared functions itself, but not
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
    the package, in which Cython finds `cyxla/__init__.pxd`.
    """
    package = Path(__file__).parent
    return [str(package / "include"), str(package.parent)]


__all__ = [
    "ABI_VERSION",
    "CACHE_ENVIRONMENT_VARIABLE",
    "MIRROR_ENVIRONMENT_VARIABLE",
    "PLATFORM_ENVIRONMENT_VARIABLE",
    "PLUGIN_ENVIRONMENT_VARIABLE",
    "ArrayType",
    "Bundle",
    "BundleError",
    "BundleWriter",
    "CompileStatistics",
    "ErrorInfo",
    "Executable",
    "LeafInfo",
    "LoweredProgram",
    "OptionValue",
    "Plugin",
    "PluginModule",
    "RuntimeInfo",
    "TargetFingerprint",
    "TargetMismatchError",
    "available_plugins",
    "cache_directory",
    "cache_key",
    "check_abi",
    "check_fingerprint",
    "compile_options",
    "compile_program",
    "custom_call_targets",
    "decode_fingerprint",
    "decode_record",
    "encode_fingerprint",
    "encode_record",
    "error_info",
    "find_plugin",
    "get_include",
    "initialize",
    "is_initialized",
    "library_identity",
    "load_lowered",
    "load_program",
    "lower",
    "mirror_jax_registrations",
    "pack_program",
    "register_ffi_target",
    "record_custom_calls",
    "raise_last_error",
    "register_jax_targets",
    "runtime_info",
    "store_lowered",
    "override_target",
    "target_fingerprint",
    "tree_fingerprint",
]
