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
    assert executable.output_types == (cx.ArrayType(np.dtype(np.float64), (3,)),) * 2
    assert executable.input_types is None
    serialized = executable.serialize()
    restored = Executable.deserialize(serialized)
    assert restored.output_types == executable.output_types


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


def test_first_use_loads_the_default_plugin() -> None:
    """Checks that the first function that needs a plugin loads it, but lowering does not."""
    result = run_python(
        """
        import jax, jax.numpy as jnp, numpy as np
        import cyxla as cx

        assert not cx.is_initialized()
        program = cx.lower(jnp.sin, (jax.ShapeDtypeStruct((2,), jnp.float32),))
        assert not cx.is_initialized()
        executable, _ = cx.compile_program(program, disk_cache=False)
        assert cx.is_initialized()
        assert cx.runtime_info().platform == "cpu"
        (result,) = executable.run([np.array([0.0, 1.0])])
        assert result.dtype == np.float32 and np.allclose(result, np.sin([0.0, 1.0]))
        """
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "call",
    [
        "cx.runtime_info()",
        "cx.Executable.compile(b'', b'')",
        "cx.Executable.deserialize(b'')",
        "cx.register_jax_targets()",
    ],
)
def test_each_entry_point_loads_the_plugin(call: str) -> None:
    """Checks that every function that needs the runtime loads it when it is not loaded."""
    result = run_python(
        f"""
        import cyxla as cx

        try:
            {call}
        except RuntimeError as error:
            assert "PJRT_" in str(error), error
        assert cx.is_initialized()
        """
    )
    assert result.returncode == 0, result.stderr


def test_initialize_after_first_use_keeps_the_first_plugin() -> None:
    """Checks that a plugin chosen after the first use does not replace the loaded one."""
    result = run_python(
        """
        import cyxla as cx

        first = cx.runtime_info()
        assert cx.initialize(platform="cuda") == first
        """
    )
    assert result.returncode == 0, result.stderr


def test_first_use_without_a_plugin() -> None:
    """Checks the error of the first use when no plugin is installed."""
    result = run_python(
        """
        import pytest
        import cyxla as cx
        from cyxla import _plugins

        _plugins.available_plugins = lambda: ()
        with pytest.raises(RuntimeError, match="No PJRT plugin is installed"):
            cx.runtime_info()
        assert not cx.is_initialized()
        """
    )
    assert result.returncode == 0, result.stderr


def test_threads_load_the_plugin_once() -> None:
    """Checks that threads that need the runtime at the same moment share one plugin."""
    result = run_python(
        """
        import threading
        import cyxla as cx

        barrier = threading.Barrier(4)
        results = []

        def use() -> None:
            barrier.wait()
            results.append(cx.runtime_info())

        threads = [threading.Thread(target=use) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 4 and len(set(results)) == 1
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
    executable, _ = cx.compile_program(program, disk_cache=False)
    product, total = executable.run(
        [np.arange(3.0), np.full(3, 2.0)], zero_copy=zero_copy
    )
    np.testing.assert_array_equal(product, [0.0, 2.0, 4.0])
    np.testing.assert_array_equal(total, 3.0)
    assert total.shape == ()
    assert executable.output_bytes == (24, 8)


DTYPES = sorted(_runtime.DTYPES.values(), key=str)
"""Every dtype that buffers can hold."""


def sample_values(dtype: np.dtype) -> np.ndarray:
    """Returns a 2 by 3 array of a dtype, with values that are exact in it."""
    values = np.arange(6).reshape(2, 3) % 2 if dtype.kind == "b" else np.arange(6) + 1
    match dtype.kind:
        case "c":
            return (values.reshape(2, 3) * (1 + 2j)).astype(dtype)
        case _:
            return values.reshape(2, 3).astype(dtype)


@pytest.mark.parametrize("zero_copy", [True, False])
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_run_keeps_every_dtype(dtype: np.dtype, zero_copy: bool) -> None:
    """Checks that an array of each supported dtype goes through a program unchanged."""
    shapes = (jax.ShapeDtypeStruct((2, 3), dtype),)
    program = cx.lower(lambda x: (x, x), shapes)
    executable, _ = cx.compile_program(program, disk_cache=False)
    values = sample_values(dtype)
    first, second = executable.run([values], zero_copy=zero_copy)
    for result in (first, second):
        assert result.dtype == dtype and result.shape == (2, 3)
        np.testing.assert_array_equal(result, values)
    assert executable.output_bytes == (values.nbytes,) * 2


def test_run_computes_in_each_dtype() -> None:
    """Checks arithmetic on integers, booleans, complex numbers and half precision."""
    cases = {
        np.dtype(np.int8): lambda x: x * 3,
        np.dtype(np.uint16): lambda x: x + 7,
        np.dtype(np.bool_): lambda x: jnp.logical_not(x),
        np.dtype(np.float16): lambda x: x / 2,
        np.dtype(jnp.bfloat16): lambda x: x * x,
        np.dtype(np.complex64): lambda x: x * 1j,
    }
    for dtype, function in cases.items():
        shapes = (jax.ShapeDtypeStruct((2, 3), dtype),)
        executable, _ = cx.compile_program(cx.lower(function, shapes), disk_cache=False)
        values = sample_values(dtype)
        (result,) = executable.run([values])
        expected = np.asarray(jax.jit(function)(values))
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)


def test_run_converts_arguments_to_program_types() -> None:
    """Checks which arguments convert to the type that a program takes."""
    shapes = (
        jax.ShapeDtypeStruct((2,), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    program = cx.lower(lambda x, n: (x * n, n), shapes)
    executable, _ = cx.compile_program(program, disk_cache=False)
    product, count = executable.run([[1.5, 2.5], 3])
    assert product.dtype == np.float32 and count.dtype == np.int32
    np.testing.assert_array_equal(product, [4.5, 7.5])
    with pytest.raises(TypeError, match="Argument 1 has the dtype float64"):
        executable.run([np.zeros(2), np.float64(1.5)])
    with pytest.raises(TypeError, match="Argument 0 has the dtype complex128"):
        executable.run([np.zeros(2, dtype=np.complex128), 1])
    with pytest.raises(ValueError, match="Argument 0 has 3 elements, but the program"):
        executable.run([np.zeros(3), 1])
    with pytest.raises(ValueError, match="takes 2 arguments, not 1"):
        executable.run([np.zeros(2)])


def test_run_errors() -> None:
    """Checks the errors of too many arguments, of unsupported dtypes and of bad sizes."""
    program = cx.lower(jnp.sin, (jax.ShapeDtypeStruct((3,), jnp.float64),))
    executable = Executable.compile(program.code, cx.compile_options())
    with pytest.raises(ValueError, match="at most 8 arguments and returns at most 4"):
        executable.run([np.zeros(3)] * 9)
    with pytest.raises(TypeError, match="Argument 0 has the dtype <U1"):
        executable.run([np.array(["a"])])
    with pytest.raises(RuntimeError, match="PJRT_LoadedExecutable_Execute"):
        executable.run([np.zeros(2)], zero_copy=False)


def test_too_many_outputs() -> None:
    """Checks that a program with more outputs than the runtime returns is refused."""
    shapes = (jax.ShapeDtypeStruct((2,), jnp.float64),)
    program = cx.lower(lambda x: (x,) * 5, shapes)
    executable, _ = cx.compile_program(program, disk_cache=False)
    assert executable.num_outputs == 5
    with pytest.raises(ValueError, match="returns at most 4 outputs, not 1 and 5"):
        executable.run([np.zeros(2)])


def test_unsupported_output_type() -> None:
    """Checks that an executable with an output of an unsupported type is refused."""
    exported = jax.export.export(
        jax.jit(lambda x: x.astype(jnp.float8_e4m3fn)), platforms=["cpu"]
    )(jax.ShapeDtypeStruct((2,), jnp.float32))
    with pytest.raises(ValueError, match="Output 0 has the PJRT element type"):
        Executable.compile(bytes(exported.mlir_module_serialized), cx.compile_options())


def test_itemsize() -> None:
    """Checks the element sizes of the supported types and the size of other types."""
    assert [_runtime.itemsize(code) for code in sorted(_runtime.DTYPES)] == [
        dtype.itemsize for _, dtype in sorted(_runtime.DTYPES.items())
    ]
    assert [_runtime.itemsize(code) for code in (0, 16, 99)] == [0, 0, 0]


def test_default_dtypes_without_x64() -> None:
    """Checks that programs run in 32 bits when JAX is not in 64-bit mode."""
    result = run_python(
        """
        import jax, jax.numpy as jnp, numpy as np
        import cyxla as cx

        program = cx.lower(
            lambda x, n: (x * n, jnp.arange(3)), (jax.ShapeDtypeStruct((2,), jnp.float32), jax.ShapeDtypeStruct((), jnp.int32))
        )
        executable, _ = cx.compile_program(program, disk_cache=False)
        scaled, steps = executable.run([np.array([0.5, 1.5]), 2])
        assert scaled.dtype == np.float32 and steps.dtype == np.int32
        assert scaled.tolist() == [1.0, 3.0] and steps.tolist() == [0, 1, 2]
        """
    )
    assert result.returncode == 0, result.stderr


def test_clock() -> None:
    """Checks that the clock of extension modules moves forward."""
    first = _runtime.clock()
    assert _runtime.clock() >= first > 0
