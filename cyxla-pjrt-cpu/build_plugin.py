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
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
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
    file there, which resolves names more reliably.
    """
    if not platform_name.startswith("windows"):
        return []
    security = (source / "ci" / "bazel_dns_windows.security").as_posix()
    return [
        f"--output_user_root={user_root}",
        f"--host_jvm_args=-Djava.security.properties=file:///{security}",
    ]


def make_junction(target: Path, link: Path) -> None:
    """Creates a directory junction at `link` that points to `target`."""
    import _winapi

    _winapi.CreateJunction(str(target), str(link))


def localize_links(
    external: Path,
    make_directory_link: Callable[[Path, Path], None],
    depth: int = 3,
) -> int:
    """Replaces the symlinks near the top of each fetched repository and returns their count.

    Repository rules link the LLVM sources into place with relative symlinks. Windows
    resolves a relative symlink against the path it was reached by, and the execroot
    reaches repositories through junctions that exist only for the repositories an action
    names. A link such as `llvm/lib` then points into a repository that the execroot does
    not link, and every `cmd.exe` action that reads below it fails. A directory link
    becomes a link with an absolute target, and a file link becomes a copy. The links sit
    within `depth` directories of the repository root, so deeper directories are skipped.
    """
    count = 0
    for repository in external.iterdir():
        if repository.is_dir() and not repository.is_symlink():
            count += localize_directory(repository, make_directory_link, depth)
    return count


def localize_directory(
    directory: Path,
    make_directory_link: Callable[[Path, Path], None],
    depth: int,
) -> int:
    """Replaces the symlinks in a directory and in the directories below it."""
    count = 0
    with os.scandir(directory) as scanner:
        entries = list(scanner)
    for entry in entries:
        path = Path(entry.path)
        if entry.is_symlink():
            target = Path(os.path.realpath(path))
            if target.is_dir():
                path.unlink()
                make_directory_link(target, path)
            else:
                path.unlink()
                shutil.copy2(target, path)
            count += 1
        elif depth > 1 and entry.is_dir(follow_symlinks=False):
            if not os.path.isjunction(path):
                count += localize_directory(path, make_directory_link, depth - 1)
    return count


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
    if platform_name.startswith("windows"):
        run([bazel, *startup, "build", "--nobuild", target], source, parsed.dry_run)
        output_base = run([bazel, *startup, "info", "output_base"], source, parsed.dry_run)
        if not parsed.dry_run:
            external = Path(output_base.strip()) / "external"
            print(f"Localized {localize_links(external, make_junction)} links.")
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
