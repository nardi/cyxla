"""Command line diagnostics: `python -m cyxla` lists the installed PJRT plugins."""

import argparse
from collections.abc import Sequence

from cyxla import _plugins


def main(arguments: Sequence[str] | None = None) -> int:
    """Prints the installed plugins and, with `--load`, the runtime of the chosen one."""
    parser = argparse.ArgumentParser(
        prog="python -m cyxla", description="Lists the installed PJRT plugins."
    )
    parser.add_argument(
        "--load",
        nargs="?",
        const="",
        metavar="PLATFORM",
        help="load the plugin for PLATFORM, or the default one, and print its runtime",
    )
    parsed = parser.parse_args(arguments)
    plugins = _plugins.available_plugins()
    if not plugins:
        print("No PJRT plugins are installed.")
    for plugin in plugins:
        print(f"{plugin.platform}\t{plugin.source}\t{plugin.path}")
    if parsed.load is not None:
        info = _plugins.initialize(platform=parsed.load or None)
        print(
            f"Loaded {info.platform}, PJRT C API "
            f"{info.api_major_version}.{info.api_minor_version}, "
            f"aliased inputs: {info.aliased_inputs}, in-place outputs: {info.inplace_outputs}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
