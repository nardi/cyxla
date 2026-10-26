"""Tests of the PJRT runtime: plugin loading, client options, executables and errors."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import SampleLibraries, run_python

import cyxla as cx
from cyxla import _runtime
from cyxla._runtime import Executable


def test_runtime_avoids_copies(runtime: cx.RuntimeInfo) -> None:
    """Checks that the CPU plugin aliases inputs and exposes outputs in host memory."""
    assert runtime.platform == "cpu"
    assert runtime.api_major_version == 0
    assert runtime.api_minor_version >= _runtime.MINIMUM_API_MINOR_VERSION
    assert runtime.aliased_inputs and runtime.inplace_outputs
    assert cx.initialize() == runtime
    assert _runtime.initialize("ignored") == runtime
    assert cx.is_initialized()
    assert cx.runtime_info() == runtime


def test_executable_round_trip() -> None:
    """Checks that an executable survives serialization."""
    program = cx.lower(
        lambda x: (jnp.sin(x), x), (jax.ShapeDtypeStruct((3,), jnp.float64),)
    )
    executable = Executable.compile(program.code, cx.compile_options())
    assert executable.num_outputs == 2
    serialized = executable.serialize()
    assert Executable.deserialize(serialized).num_outputs == 2


def test_errors_are_recorded() -> None:
    """Checks that rejected bytes raise and that the error is recorded."""
    count = cx.error_info().count
    with pytest.raises(RuntimeError, match="PJRT_Executable_DeserializeAndLoad"):
        Executable.deserialize(b"not an executable")
    with pytest.raises(RuntimeError, match="PJRT_Client_Compile"):
        Executable.compile(b"not bytecode", cx.compile_options())
    info = cx.error_info()
    assert info.count == count + 2
    assert info.last_message


def test_uninitialized_runtime() -> None:
    """Checks the errors of using the runtime before a plugin is loaded."""
    result = run_python(
        """
        import pytest
        import cyxla as cx
        from cyxla import _cache, _lowering

        assert not cx.is_initialized()
        assert _lowering.lowering_platform(None) == "cpu"
        for call in (
            cx.runtime_info,
            lambda: cx.Executable.compile(b"", b""),
            lambda: cx.Executable.deserialize(b""),
        ):
            with pytest.raises(RuntimeError, match="not initialized"):
                call()
        with pytest.raises(RuntimeError, match="No PJRT plugin is loaded"):
            _cache.executable_key(cx.LoweredProgram(b"", ""))
        """
    )
    assert result.returncode == 0, result.stderr


def test_bad_plugins(sample_libraries: SampleLibraries) -> None:
    """Checks the errors of a missing plugin, a library that is not one, and an old one."""
    not_plugin = sample_libraries.not_plugin
    old_plugin = sample_libraries.old_plugin
    result = run_python(
        f"""
        from pathlib import Path

        import pytest
        import cyxla as cx

        with pytest.raises(OSError, match="Cannot load"):
            cx.initialize(Path("/nonexistent/libpjrt.so"))
        with pytest.raises(OSError, match="does not export GetPjrtApi"):
            cx.initialize(Path({str(not_plugin)!r}))
        with pytest.raises(RuntimeError, match="too old"):
            cx.initialize(Path({str(old_plugin)!r}))
        """
    )
    assert result.returncode == 0, result.stderr


def test_client_options() -> None:
    """Checks that every option type reaches the plugin and that rejected options raise.

    The CPU plugin rejects the first unknown name, so the options of other types are
    checked by the error that the last of them causes. A failed initialization leaves the
    runtime unloaded, so loading again succeeds.
    """
    result = run_python(
        """
        import pytest
        import cyxla as cx

        with pytest.raises(TypeError, match="unsupported type list"):
            cx.initialize(platform="cpu", options={"listed": [1]})
        with pytest.raises(RuntimeError, match="expected type index"):
            cx.initialize(platform="cpu", options={"cpu_device_count": "one"})
        for name, value in (("ratio", 0.5), ("ids", (1, 2)), ("empty", ()), ("tag", "x")):
            with pytest.raises(RuntimeError, match=f"Unexpected option name.*{name}"):
                cx.initialize(platform="cpu", options={name: value})
        assert not cx.is_initialized()
        info = cx.initialize(platform="cpu", options={"cpu_device_count": 2})
        assert info.platform == "cpu" and info.aliased_inputs
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("zero_copy", [True, False])
def test_run(zero_copy: bool) -> None:
    """Checks that programs run with aliased and with copied buffers."""
    shapes = (jax.ShapeDtypeStruct((3,), jnp.float64),) * 2
    program = cx.lower(lambda x, y: (x * y, jnp.sum(x)), shapes)
    executable = Executable.compile(program.code, cx.compile_options())
    product, total = executable.run(
        [np.arange(3.0), np.full(3, 2.0)], [3, 1], zero_copy=zero_copy
    )
    np.testing.assert_array_equal(product, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(total, [3.0])


def test_run_errors() -> None:
    """Checks the errors of wrong argument counts and of arguments of the wrong size."""
    program = cx.lower(jnp.sin, (jax.ShapeDtypeStruct((3,), jnp.float64),))
    executable = Executable.compile(program.code, cx.compile_options())
    with pytest.raises(ValueError, match="at most 8 arguments and has 1 outputs"):
        executable.run([np.zeros(3)] * 9, [3])
    with pytest.raises(RuntimeError, match="PJRT_LoadedExecutable_Execute"):
        executable.run([np.zeros(2)], [3], zero_copy=False)


def test_clock() -> None:
    """Checks that the clock of extension modules moves forward."""
    first = _runtime.clock()
    assert _runtime.clock() >= first > 0
