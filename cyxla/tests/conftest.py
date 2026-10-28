"""Shared fixtures: the PJRT CPU plugin, a fresh disk cache and helpers for subprocesses."""

import ctypes
import os
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import jax
import pytest

import cyxla as cx

HEADER_DIRECTORY = Path(cx.get_include()[0])
"""Directory from which the PJRT C API header is included."""

ENVIRONMENT_VARIABLES = (
    cx.PLUGIN_ENVIRONMENT_VARIABLE,
    cx.PLATFORM_ENVIRONMENT_VARIABLE,
)
"""Environment variables that change which plugin loads, cleared for the tests."""


@pytest.fixture(scope="session", autouse=True)
def runtime(tmp_path_factory: pytest.TempPathFactory) -> Iterator[cx.RuntimeInfo]:
    """Loads the installed CPU plugin, with a fresh disk cache."""
    previous = {name: os.environ.pop(name, None) for name in ENVIRONMENT_VARIABLES}
    previous_cache = os.environ.get(cx.CACHE_ENVIRONMENT_VARIABLE)
    os.environ[cx.CACHE_ENVIRONMENT_VARIABLE] = str(tmp_path_factory.mktemp("cache"))
    jax.config.update("jax_enable_x64", True)
    yield cx.initialize(platform="cpu")
    for name, value in previous.items():
        if value is not None:
            os.environ[name] = value
    if previous_cache is None:
        del os.environ[cx.CACHE_ENVIRONMENT_VARIABLE]
    else:
        os.environ[cx.CACHE_ENVIRONMENT_VARIABLE] = previous_cache


@pytest.fixture
def cache_directory() -> Path:
    """Returns the disk cache directory of the session."""
    return Path(os.environ[cx.CACHE_ENVIRONMENT_VARIABLE])


def run_python(
    code: str, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Runs code in a fresh interpreter without a loaded plugin.

    The variables that select a plugin are cleared, and `environment` adds to the rest.
    """
    variables = {
        key: value
        for key, value in os.environ.items()
        if key not in ENVIRONMENT_VARIABLES
    }
    variables |= environment or {}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=variables,
        check=False,
    )


class SampleLibraries(NamedTuple):
    """Shared libraries that the tests load."""

    not_plugin: Path
    """A library without `GetPjrtApi`."""
    old_plugin: Path
    """A plugin of a PJRT C API version that is too old."""
    preloaded: Path
    """A library with a function `preloaded` that returns 1."""


def without_coverage() -> dict[str, str]:
    """Returns the environment without the variables that start coverage in subprocesses."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COV_CORE_", "COVERAGE_"))
    }


def meson_executable() -> str:
    """Returns the Meson of this environment."""
    return shutil.which("meson", path=str(Path(sys.executable).parent)) or "meson"


@pytest.fixture(scope="session")
def sample_libraries(tmp_path_factory: pytest.TempPathFactory) -> SampleLibraries:
    """Builds the libraries in `tests/libraries` with Meson and returns their paths."""
    build = tmp_path_factory.mktemp("libraries")
    source = Path(__file__).parent / "libraries"
    for command in (
        [
            meson_executable(),
            "setup",
            str(build),
            str(source),
            "--vsenv",
            f"-Dpjrt_include={HEADER_DIRECTORY}",
        ],
        [meson_executable(), "compile", "-C", str(build)],
    ):
        result = subprocess.run(
            command, capture_output=True, text=True, env=without_coverage(), check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def library(name: str) -> Path:
        (path,) = [
            path
            for path in build.iterdir()
            if name in path.name and path.suffix in (".so", ".dylib", ".dll")
        ]
        return path

    return SampleLibraries(*(library(name) for name in SampleLibraries._fields))


@pytest.fixture(scope="session")
def ffi_library_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Builds the library in `tests/ffi_library` with Meson and returns its path."""
    build = tmp_path_factory.mktemp("ffi_library")
    source = Path(__file__).parent / "ffi_library"
    for command in (
        [
            meson_executable(),
            "setup",
            str(build),
            str(source),
            "--vsenv",
            "--buildtype=release",
            f"-Dffi_include={jax.ffi.include_dir()}",
        ],
        [meson_executable(), "compile", "-C", str(build)],
    ):
        result = subprocess.run(
            command, capture_output=True, text=True, env=without_coverage(), check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
    (path,) = [
        path
        for path in build.iterdir()
        if "scale" in path.name and path.suffix in (".so", ".dylib", ".dll")
    ]
    return path


@pytest.fixture(scope="session")
def ffi_library(ffi_library_path: Path) -> ctypes.CDLL:
    """Returns the library of `ffi_library_path` loaded."""
    library = ctypes.CDLL(str(ffi_library_path))
    library.scale_handler.restype = ctypes.c_void_p
    return library
