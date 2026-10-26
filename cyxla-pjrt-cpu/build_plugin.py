"""Builds the PJRT CPU plugin of this machine from XLA, with the Bazel options of jaxlib.

The script checks out JAX at the tag in `plugin.toml` and copies `bazel/` into it as the
package `//cyxla_pjrt_cpu`. JAX's `build/build.py` then writes the options that it builds
jaxlib with to `.jax_configure.bazelrc`, and Bazel builds the plugin with them. The library
ends up in the output directory, and the script prints its hash and its wheel tag.

    python build_plugin.py --output dist/library

Bazel comes from `bazelisk` or `bazel` on the path. A build takes hours without a cache, so
`--disk-cache` keeps the outputs of earlier builds and CI saves that directory between runs.
"""

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import backend

PROJECT_DIRECTORY = Path(__file__).parent
"""Directory of this project."""

PACKAGE = "cyxla_pjrt_cpu"
"""Name of the Bazel package that `bazel/` becomes in the checkout of JAX."""


def run(command: Sequence[str], directory: Path, dry_run: bool) -> str:
    """Runs a command in a directory and returns its standard output.

    With `dry_run` the command is only printed, and the output is empty.
    """
    print("+", " ".join(command), flush=True)
    if dry_run:
        return ""
    result = subprocess.run(
        command, cwd=directory, check=True, stdout=subprocess.PIPE, text=True
    )
    return result.stdout


def checkout_jax(settings: dict, directory: Path, dry_run: bool) -> None:
    """Clones JAX at the pinned tag, or checks an existing clone, and verifies the commit."""
    if not (directory / ".git").exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                settings["jax_tag"],
                settings["jax_repository"],
                str(directory),
            ],
            directory.parent,
            dry_run,
        )
    if dry_run:
        return
    commit = run(["git", "rev-parse", "HEAD"], directory, dry_run).strip()
    if commit != settings["jax_commit"]:
        raise RuntimeError(
            f"{directory} is at {commit}, but {settings['jax_tag']} is "
            f"{settings['jax_commit']}."
        )


def bazel_executable() -> str:
    """Returns the Bazel launcher on the path, preferring bazelisk."""
    found = shutil.which("bazelisk") or shutil.which("bazel")
    if found is None:
        raise RuntimeError("Install bazelisk, which reads the Bazel version of JAX.")
    return found


def startup_options(platform_name: str, source: Path, user_root: str) -> list[str]:
    """Returns the Bazel startup options of a platform.

    On Windows, Bazel's output lives under a short path, because the paths of XLA's outputs
    exceed the length limit of Windows otherwise. JAX's CI also passes its Java security
    file there, which resolves names more reliably. Bazel creates real symlinks there only
    on request, and the LLVM build copies files through them.
    """
    if not platform_name.startswith("windows"):
        return []
    security = (source / "ci" / "bazel_dns_windows.security").as_posix()
    return [
        f"--output_user_root={user_root}",
        f"--host_jvm_args=-Djava.security.properties=file:///{security}",
        "--windows_enable_symlinks",
    ]


def main(arguments: Sequence[str] | None = None) -> int:
    """Builds the plugin and copies it to the output directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path, required=True, help="directory for the library"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_DIRECTORY / "build" / "jax",
        help="checkout of JAX to build in, cloned if it does not exist",
    )
    parser.add_argument("--disk-cache", type=Path, help="Bazel disk cache directory")
    parser.add_argument(
        "--user-root",
        default="C:/b",
        help="Bazel output root on Windows, which needs a short path",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands without running them"
    )
    parsed = parser.parse_args(arguments)
    settings = backend.read_toml("plugin.toml")
    platform_name = backend.host_platform()
    entry = settings["platforms"][platform_name]
    source = parsed.source.resolve()
    checkout_jax(settings, source, parsed.dry_run)
    if not parsed.dry_run:
        shutil.copytree(
            PROJECT_DIRECTORY / "bazel", source / PACKAGE, dirs_exist_ok=True
        )

    bazel = bazel_executable() if not parsed.dry_run else "bazel"
    startup = startup_options(platform_name, source, parsed.user_root)
    options = list(entry["bazel_options"])
    if parsed.disk_cache is not None:
        options.append(f"--disk_cache={parsed.disk_cache.resolve().as_posix()}")
    run(
        [
            sys.executable,
            "build/build.py",
            "build",
            "--wheels=jaxlib",
            "--configure_only",
            f"--python_version={settings['python_version']}",
            f"--bazel_path={bazel}",
            *(f"--bazel_startup_options={option}" for option in startup),
            *(f"--bazel_options={option}" for option in options),
        ],
        source,
        parsed.dry_run,
    )
    target = f"//{PACKAGE}:{entry['library']}"
    run([bazel, *startup, "build", target], source, parsed.dry_run)
    files = run(
        [bazel, *startup, "cquery", "--output=files", target], source, parsed.dry_run
    )
    if parsed.dry_run:
        return 0
    built = source / files.strip().splitlines()[-1]
    parsed.output.mkdir(parents=True, exist_ok=True)
    library = parsed.output / entry["library"]
    shutil.copyfile(built, library)
    print(
        f"{library}: sha256 {backend.file_hash(library)}, tag {backend.wheel_tag(library)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
