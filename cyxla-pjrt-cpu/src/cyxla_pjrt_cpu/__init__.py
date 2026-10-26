"""The XLA PJRT CPU plugin, packaged for cyxla.

cyxla finds this module through the `cyxla.plugins` entry point and reads the three names
below. The module does not import cyxla, so it has no dependencies.
"""

import sys
from pathlib import Path

PLATFORM = "cpu"
"""Platform that the plugin reports, which is also the JAX lowering platform."""

CREATE_OPTIONS: dict[str, bool | int] = {"asynchronous": False, "cpu_device_count": 1}
"""Options of `PJRT_Client_Create`.

A synchronous client runs executions in the calling thread, and one device keeps it from
starting threads for devices that are never used.
"""


def library_path() -> Path:
    """Returns the path of the plugin library in this package."""
    match sys.platform:
        case "darwin":
            name = "libpjrt_cpu.dylib"
        case "win32":
            name = "pjrt_cpu.dll"
        case _:
            name = "libpjrt_cpu.so"
    return Path(__file__).parent / name
