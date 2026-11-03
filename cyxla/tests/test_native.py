"""Tests of native functions: C functions that programs call by name without the GIL."""

import ctypes
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cyxla as cx

SHAPES = (jax.ShapeDtypeStruct((3,), jnp.float64),)
"""The argument of the lowered test functions."""

POINT = np.array([1.0, 2.0, 3.0])
"""A point to run the programs on."""


def address(library: ctypes.CDLL, name: str) -> int:
    """Returns the address of an exported function."""
    value = ctypes.cast(getattr(library, name), ctypes.c_void_p).value
    assert value is not None
    return value


@pytest.fixture(scope="module")
def natives(native_library_path: Path) -> SimpleNamespace:
    """Registers the functions of the test library once, and returns what the tests need."""
    library = ctypes.CDLL(str(native_library_path))
    offset = ctypes.c_double(10.0)
    calls = ctypes.c_int64(0)
    for name, function, state in (
        ("test.scale", "scale", 0),
        ("test.scale_offset", "scale", ctypes.addressof(offset)),
        ("test.pair", "sum_and_difference", 0),
        ("test.label_length", "label_length", 0),
        ("test.fails", "always_fails", 0),
        ("test.square", "square", 0),
        ("test.square_backward", "square_backward", 0),
        ("test.sleep", "sleep_and_count", ctypes.addressof(calls)),
        ("test.record", "record_argument", 0),
    ):
        cx.register_native_function(name, address(library, function), state)
    return SimpleNamespace(library=library, offset=offset, calls=calls)


def spec(shape: tuple[int, ...] = (3,)) -> jax.ShapeDtypeStruct:
    """Returns the structure of a float64 result."""
    return jax.ShapeDtypeStruct(shape, jnp.float64)


def run(function: Any, shapes: Any = SHAPES, point: Any = POINT) -> list[np.ndarray]:
    """Lowers a function, compiles it in cyxla and runs it on a point."""
    program = cx.lower(function, shapes)
    executable, _ = cx.compile_program(program, disk_cache=False)
    return executable.run([point])


def test_registration_rules(natives: SimpleNamespace) -> None:
    """Checks the rules for names, the same function twice and the interface version."""
    scale = address(natives.library, "scale")
    cx.register_native_function("test.scale", scale)
    with pytest.raises(ValueError, match="registered with another function"):
        cx.register_native_function("test.scale", address(natives.library, "square"))
    with pytest.raises(ValueError, match="version 99 of the native interface"):
        cx.register_native_function("test.new", scale, abi_version=99)
    info = {item.name: item for item in cx.registered_native_functions()}
    assert info["test.scale"].function_address == scale
    assert info["test.scale_offset"].user_data_address == ctypes.addressof(
        natives.offset
    )
    assert info["test.scale"].provider is not None


def test_call_in_a_program(natives: SimpleNamespace) -> None:
    """Checks that a program calls the function with its attribute, and needs no allowlist."""

    def function(x: jax.Array) -> jax.Array:
        return cx.native_call("test.scale", spec(), x, factor=3.0) + 1.0

    (result,) = run(function)
    np.testing.assert_array_equal(result, 3.0 * POINT + 1.0)


def test_state_of_a_function(natives: SimpleNamespace) -> None:
    """Checks that the function receives the pointer that it was registered with."""
    (result,) = run(
        lambda x: cx.native_call("test.scale_offset", spec(), x, factor=2.0)
    )
    np.testing.assert_array_equal(result, 2.0 * POINT + 10.0)
    natives.offset.value = 20.0
    (result,) = run(
        lambda x: cx.native_call("test.scale_offset", spec(), x, factor=2.0)
    )
    np.testing.assert_array_equal(result, 2.0 * POINT + 20.0)
    natives.offset.value = 10.0


def test_two_results_and_a_string_attribute(natives: SimpleNamespace) -> None:
    """Checks several arguments and results, and an attribute that is a string."""

    def function(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        total, difference = cx.native_call("test.pair", (spec(), spec()), x, 2.0 * x)
        length = cx.native_call("test.label_length", spec((1,)), label="hello world")
        return total, difference, length

    total, difference, length = run(function)
    np.testing.assert_array_equal(total, 3.0 * POINT)
    np.testing.assert_array_equal(difference, -POINT)
    np.testing.assert_array_equal(length, [11.0])


def test_same_results_in_plain_jax(natives: SimpleNamespace) -> None:
    """Checks that jax.jit runs the same function through jaxlib, and agrees with cyxla."""

    def function(x: jax.Array) -> jax.Array:
        return cx.native_call("test.scale", spec(), x, factor=3.0) + 1.0

    expected = np.asarray(jax.jit(function)(POINT))
    (actual,) = run(function)
    np.testing.assert_array_equal(actual, expected)


def test_empty_arrays(natives: SimpleNamespace) -> None:
    """Checks a call whose arguments and results have no elements."""
    shapes = (jax.ShapeDtypeStruct((0,), jnp.float64),)
    (result,) = run(
        lambda x: cx.native_call("test.scale", spec((0,)), x, factor=2.0),
        shapes,
        np.zeros(0),
    )
    assert result.shape == (0,)


def test_error_message_ends_the_run(natives: SimpleNamespace) -> None:
    """Checks that a returned message is the error of the run, and the program survives."""

    def function(x: jax.Array) -> jax.Array:
        failing = cx.native_call("test.fails", spec(), x, has_side_effect=True)
        return failing + x

    executable, _ = cx.compile_program(cx.lower(function, SHAPES), disk_cache=False)
    with pytest.raises(RuntimeError, match="the native function failed on purpose"):
        executable.run([POINT])
    with pytest.raises(RuntimeError, match="failed on purpose"):
        executable.run([POINT])


def test_missing_function_and_attribute_errors(natives: SimpleNamespace) -> None:
    """Checks the messages of a call to an unknown name and of a call missing an attribute."""
    executable, _ = cx.compile_program(
        cx.lower(lambda x: cx.native_call("test.missing", spec(), x), SHAPES),
        disk_cache=False,
    )
    with pytest.raises(
        RuntimeError,
        match="No native function is registered under the name test.missing",
    ):
        executable.run([POINT])
    executable, _ = cx.compile_program(
        cx.lower(lambda x: cx.native_call("test.scale", spec(), x), SHAPES),
        disk_cache=False,
    )
    with pytest.raises(RuntimeError, match="needs the float64 attribute factor"):
        executable.run([POINT])


def test_functions_run_without_the_gil(natives: SimpleNamespace) -> None:
    """Checks that Python code keeps running while eight executions sleep in C.

    The function sleeps for 0.3 seconds. If it held the GIL, the main thread could not run
    during that time, and the longest pause between two of its turns would be about as long.
    The check does not time the executions against each other, since the runtime has as many
    threads as the machine has processors and a runner of CI has two.
    """
    executable, _ = cx.compile_program(
        cx.lower(
            lambda x: cx.native_call("test.sleep", spec(), x, seconds=0.3), SHAPES
        ),
        disk_cache=False,
    )
    executable.run([POINT])
    before = natives.calls.value
    threads = [
        threading.Thread(target=lambda: executable.run([POINT])) for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    longest_pause = 0.0
    previous = time.perf_counter()
    while any(thread.is_alive() for thread in threads):
        now = time.perf_counter()
        longest_pause = max(longest_pause, now - previous)
        previous = now
    for thread in threads:
        thread.join()
    assert natives.calls.value - before == 8
    assert longest_pause < 0.15


def test_argument_data_is_the_runtime_buffer(natives: SimpleNamespace) -> None:
    """Checks that the function sees the element type and a pointer, with no copy by cyxla."""
    run(lambda x: cx.native_call("test.record", spec(), x))
    dtype = ctypes.c_int32.in_dll(natives.library, "last_argument_dtype").value
    assert dtype == 12
    pointer = ctypes.c_void_p.in_dll(natives.library, "last_argument_data").value
    assert pointer is not None and pointer % 8 == 0


def test_calls_that_the_handler_refuses(natives: SimpleNamespace) -> None:
    """Checks the messages for a malformed call, which the Python interface cannot make."""
    target = "cyxla_native_function"
    many: dict[str, Any] = {f"attribute_{n}": 1 for n in range(20)}
    long_name: dict[str, Any] = {"n" * 80: 1}
    cases = {
        "needs the attribute name": lambda x: jax.ffi.ffi_call(target, spec())(x),
        "attribute name must be a string": lambda x: jax.ffi.ffi_call(target, spec())(
            x, name=np.int32(5)
        ),
        "too many or too long attributes": lambda x: cx.native_call(
            "test.scale",
            spec(),
            x,
            factor=1.0,
            **many,
        ),
        "too long attributes": lambda x: cx.native_call(
            "test.scale", spec(), x, factor=1.0, **long_name
        ),
        "scalar and string attributes only": lambda x: cx.native_call(
            "test.scale", spec(), x, factor=1.0, values=np.arange(3)
        ),
        "too many arguments or results": lambda x: cx.native_call(
            "test.scale", spec(), *([x] * 40), factor=1.0
        ),
    }
    for message, function in cases.items():
        executable, _ = cx.compile_program(cx.lower(function, SHAPES), disk_cache=False)
        with pytest.raises(RuntimeError, match=message):
            executable.run([POINT])
