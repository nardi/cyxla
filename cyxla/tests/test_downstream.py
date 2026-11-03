"""Tests of an extension module of another package that builds against cyxla."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from conftest import meson_executable, without_coverage

import cyxla as cx

DOWNSTREAM_DIRECTORY = Path(__file__).parent / "downstream"
"""Meson project of the downstream extension module."""

DOWNSTREAM_TEST = """
import sys

import jax
import jax.numpy as jnp
import cyxla as cx

sys.path.insert(0, sys.argv[1])
import downstream_square

assert downstream_square.COMPILED_ABI_VERSION == cx.ABI_VERSION
program = cx.lower(lambda x: x**2 + 1, (jax.ShapeDtypeStruct((3,), jnp.float32),))
executable, _ = cx.compile_program(program, disk_cache=False)
assert executable.output_bytes == (12,)
for zero_copy in (True, False):
    result = downstream_square.run(executable, [1.0, 2.0, 3.0], zero_copy)
    assert result == [2.0, 5.0, 10.0], result
"""
"""Imports the built module, whose directory is the first argument, and runs a program with it."""


DOWNSTREAM_NATIVE_TEST = """
import sys

import jax
import jax.numpy as jnp
import numpy as np
import cyxla as cx

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, sys.argv[1])
import downstream_native

info = {item.name: item for item in cx.registered_native_functions()}
assert info["downstream.scale"].provider == "__main__" or info["downstream.scale"].provider is None, info

def function(x):
    return cx.native_call(
        "downstream.scale", jax.ShapeDtypeStruct(x.shape, x.dtype), x, factor=2.5
    )

program = cx.lower(function, (jax.ShapeDtypeStruct((3,), jnp.float64),))
executable, _ = cx.compile_program(program, disk_cache=False)
(result,) = executable.run([np.array([1.0, 2.0, 3.0])])
assert result.tolist() == [2.5, 5.0, 7.5], result
assert np.asarray(jax.jit(function)(np.array([1.0, 2.0, 3.0]))).tolist() == [2.5, 5.0, 7.5]
"""
"""Imports the Cython module that registers a native function, and calls it from a program."""


def test_get_include() -> None:
    """Checks that the include directories hold the header and the declarations."""
    header, declarations = map(Path, cx.get_include())
    assert (header / "xla" / "pjrt" / "c" / "pjrt_c_api.h").exists()
    assert (declarations / "cyxla" / "_runtime.pxd").exists()


def test_check_abi() -> None:
    """Checks that a module compiled against other declarations is rejected."""
    cx.check_abi(cx.ABI_VERSION)
    with pytest.raises(ImportError, match="Rebuild it"):
        cx.check_abi(cx.ABI_VERSION + 1)


def test_downstream_module_runs_programs(tmp_path: Path) -> None:
    """Checks that a module built with `get_include` executes programs.

    The build and the run happen outside coverage. The coverage plugin of Cython would
    attribute lines of the downstream module to the cyxla declarations it cimports, which
    conflicts with the records of cyxla itself.
    """
    environment = {
        key: value
        for key, value in without_coverage().items()
        if key not in (cx.PLUGIN_ENVIRONMENT_VARIABLE, cx.PLATFORM_ENVIRONMENT_VARIABLE)
    }
    meson = meson_executable()
    for command in (
        # A release build, because a debug build of an extension links the debug library
        # of Python, which Windows installations of Python do not include.
        [
            meson,
            "setup",
            "--vsenv",
            "--buildtype=release",
            str(tmp_path),
            str(DOWNSTREAM_DIRECTORY),
        ],
        [meson, "compile", "-C", str(tmp_path)],
        [sys.executable, "-c", textwrap.dedent(DOWNSTREAM_TEST), str(tmp_path)],
        [sys.executable, "-c", textwrap.dedent(DOWNSTREAM_NATIVE_TEST), str(tmp_path)],
    ):
        result = subprocess.run(
            command, capture_output=True, text=True, env=environment, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr
