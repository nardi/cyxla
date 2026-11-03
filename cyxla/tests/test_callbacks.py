"""Tests of Python callbacks inside programs, which the handler of cyxla runs."""

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import io_callback

import cyxla as cx
from cyxla import _callbacks, _ffi, _native_functions, _targets

SHAPES = (jax.ShapeDtypeStruct((3,), jnp.float64),)
"""The argument of the lowered test functions."""

POINT = np.array([1.0, 2.0, 3.0])
"""A point to run the programs on."""


def compiled(
    function: Any, shapes: Any = SHAPES, **options: Any
) -> tuple[cx.Executable, cx.LoweredProgram]:
    """Lowers a function with callbacks, compiles it, and returns both."""
    program = cx.lower(function, shapes, host_callbacks=True, **options)
    executable, _ = cx.compile_program(program, disk_cache=False)
    return executable, program


def test_debug_callback_receives_the_values() -> None:
    """Checks that a callback that only observes gets the values of the arguments."""
    seen: list[np.ndarray] = []

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda value: seen.append(np.asarray(value)), x)
        return x * 2.0

    executable, _ = compiled(function)
    (result,) = executable.run([POINT])
    np.testing.assert_array_equal(result, 2.0 * POINT)
    (value,) = seen
    assert value.dtype == np.float64
    np.testing.assert_array_equal(value, POINT)


def test_pure_callback_returns_values_to_the_program() -> None:
    """Checks that the results of a callback flow into the rest of the program."""

    def function(x: jax.Array) -> jax.Array:
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        return jax.pure_callback(lambda v: np.asarray(v) * 3.0, spec, x) + 1.0

    executable, _ = compiled(function)
    np.testing.assert_array_equal(executable.run([POINT])[0], 3.0 * POINT + 1.0)


def test_callbacks_in_loops_and_with_several_dtypes() -> None:
    """Checks a callback in a loop, and arguments and results of other element types."""
    counted: list[float] = []
    integers: list[np.ndarray] = []

    def function(x: jax.Array) -> jax.Array:
        def body(index: Any, value: jax.Array) -> jax.Array:
            jax.debug.callback(lambda i: counted.append(float(i)), index)
            return value + 1.0

        jax.debug.callback(integers.append, jnp.arange(3, dtype=jnp.int32) > 0)
        return jax.lax.fori_loop(0, 4, body, x)

    executable, _ = compiled(function)
    np.testing.assert_array_equal(executable.run([POINT])[0], POINT + 4.0)
    assert counted == [0.0, 1.0, 2.0, 3.0]
    np.testing.assert_array_equal(integers[0], [False, True, True])


def test_callback_without_elements() -> None:
    """Checks that an empty array reaches the callback."""
    shapes = (jax.ShapeDtypeStruct((0,), jnp.float64),)
    sizes: list[tuple[int, ...]] = []

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda value: sizes.append(np.shape(value)), x)
        return x

    executable, _ = compiled(function, shapes)
    assert executable.run([np.zeros(0)])[0].shape == (0,)
    assert sizes == [(0,)]


def test_raising_callback_fails_the_run_and_the_program_survives() -> None:
    """Checks that an exception becomes an error that names it, and that the next run works."""

    def check(value: np.ndarray) -> None:
        if value[0] < 0:
            raise ValueError("negative input")

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(check, x)
        return x * 2.0

    executable, _ = compiled(function)
    with pytest.raises(RuntimeError, match="ValueError: negative input"):
        executable.run([-POINT])
    np.testing.assert_array_equal(executable.run([POINT])[0], 2.0 * POINT)


def test_error_check_in_a_branch_like_equinox_makes() -> None:
    """Checks a callback that raises inside `lax.cond`, which only runs when the branch does."""

    def raises(value: np.ndarray) -> np.ndarray:
        raise RuntimeError("the check failed")

    def function(x: jax.Array) -> jax.Array:
        def fail(value: jax.Array) -> jax.Array:
            spec = jax.ShapeDtypeStruct(value.shape, value.dtype)
            return jax.pure_callback(raises, spec, value)

        return jax.lax.cond(jnp.any(x < 0), fail, lambda value: value, x)

    executable, _ = compiled(function)
    np.testing.assert_array_equal(executable.run([POINT])[0], POINT)
    with pytest.raises(RuntimeError, match="RuntimeError: the check failed"):
        executable.run([-POINT])
    np.testing.assert_array_equal(executable.run([POINT])[0], POINT)


def test_wrong_result_shape_is_reported() -> None:
    """Checks that JAX's own check of the results of a callback reaches the caller."""

    def function(x: jax.Array) -> jax.Array:
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        return jax.pure_callback(lambda v: np.zeros(2), spec, x)

    executable, _ = compiled(function)
    with pytest.raises(RuntimeError, match="Incorrect output shape"):
        executable.run([POINT])


def test_threads_run_one_program_at_once() -> None:
    """Checks that callbacks from several executions at once all arrive, without deadlock."""
    seen: list[float] = []

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda v: seen.append(float(v[0])), x)
        return x

    executable, _ = compiled(function)
    with ThreadPoolExecutor(8) as pool:
        list(
            pool.map(
                lambda number: executable.run([np.array([float(number), 0.0, 0.0])]),
                range(32),
            )
        )
    assert sorted(seen) == [float(number) for number in range(32)]


def test_programs_get_their_own_callbacks() -> None:
    """Checks that two programs with a callback at the same position call different ones."""
    log: list[str] = []

    def make(name: str) -> Any:
        def function(x: jax.Array) -> jax.Array:
            jax.debug.callback(lambda v: log.append(name), x)
            return x

        return function

    first, _ = compiled(make("first"))
    second, _ = compiled(make("second"))
    second.run([POINT])
    first.run([POINT])
    assert log == ["second", "first"]


def test_callbacks_leave_the_registry_with_their_program() -> None:
    """Checks that collecting a program and its executable removes its callbacks."""
    before = set(_callbacks.callbacks)

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda v: None, x)
        return x

    executable, program = compiled(function)
    (key,) = set(_callbacks.callbacks) - before
    assert program.callbacks is not None and program.callbacks.keys == [key]
    del executable, program
    gc.collect()
    assert key not in _callbacks.callbacks


def test_released_callback_is_an_error() -> None:
    """Checks the error for a program whose callback is gone from the registry."""

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda v: None, x)
        return x

    executable, program = compiled(function)
    assert program.callbacks is not None
    key = program.callbacks.keys[0]
    saved = _callbacks.callbacks.pop(key)
    try:
        with pytest.raises(
            RuntimeError, match="LookupError: The callback .* was released"
        ):
            executable.run([POINT])
    finally:
        _callbacks.callbacks[key] = saved


def test_callback_with_the_wrong_number_of_results() -> None:
    """Checks the handler's own check, which JAX's wrapper makes unreachable in practice."""

    def function(x: jax.Array) -> jax.Array:
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        return jax.pure_callback(lambda v: v, spec, x)

    executable, program = compiled(function)
    assert program.callbacks is not None
    key = program.callbacks.keys[0]
    saved = _callbacks.callbacks[key]
    _callbacks.callbacks[key] = lambda *arguments: ()
    try:
        with pytest.raises(RuntimeError, match="returned 0 results"):
            executable.run([POINT])
    finally:
        _callbacks.callbacks[key] = saved


def test_element_types_of_the_api_that_callbacks_cannot_take() -> None:
    """Checks the error for an argument of a type that the handler has no NumPy type for."""

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda value: None, x.astype(jnp.float8_e4m3fn))
        return x

    executable, _ = compiled(function)
    with pytest.raises(
        RuntimeError, match="cannot take elements of the FFI element type"
    ):
        executable.run([POINT])


def test_call_without_an_index_is_an_error() -> None:
    """Checks the answer of the native function to a call that lacks the attribute."""

    def function(x: jax.Array) -> jax.Array:
        return cx.native_call(
            _callbacks.PYTHON_CALLBACK, jax.ShapeDtypeStruct(x.shape, x.dtype), x
        )

    executable, _ = compiled(function)
    with pytest.raises(RuntimeError, match="no attribute index"):
        executable.run([POINT])


def test_callback_with_too_many_arguments_is_an_error() -> None:
    """Checks that the limit of the native function handler applies to callbacks."""

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda *values: None, *[x + index for index in range(33)])
        return x

    executable, _ = compiled(function)
    with pytest.raises(RuntimeError, match="too many arguments or results"):
        executable.run([POINT])


def test_ordered_callbacks_are_refused() -> None:
    """Checks that tokens, which the runtime cannot pass, are reported when lowering."""

    def function(x: jax.Array) -> jax.Array:
        io_callback(lambda v: None, None, x, ordered=True)
        return x

    with pytest.raises(NotImplementedError, match="ordered callbacks"):
        cx.lower(function, SHAPES, host_callbacks=True)


def test_other_platforms_are_refused() -> None:
    """Checks that callbacks are for the CPU only."""
    with pytest.raises(NotImplementedError, match="CPU only, not on cuda"):
        cx.lower(lambda x: x, SHAPES, platform="cuda", host_callbacks=True)


def test_callback_flag_without_callbacks_gives_an_ordinary_program() -> None:
    """Checks that a function without callbacks lowers to a program that can be stored."""
    executable, program = compiled(lambda x: x + 1.0)
    assert program.callbacks is None
    np.testing.assert_array_equal(executable.run([POINT])[0], POINT + 1.0)
    cx.pack_program(cx.BundleWriter("test.program", 1), "plain", executable)


def test_program_without_debug_info_keeps_its_callbacks() -> None:
    """Checks that the locations can be removed from a program with callbacks."""
    seen: list[float] = []

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda v: seen.append(float(v[0])), x)
        return x

    executable, program = compiled(function, debug_info=False)
    assert "loc(" not in program.text
    executable.run([POINT])
    assert seen == [1.0]


def test_programs_with_callbacks_are_not_stored(tmp_path: Path) -> None:
    """Checks that storing, serializing and the disk cache all refuse or skip such a program."""

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda v: None, x)
        return x

    program = cx.lower(function, SHAPES, host_callbacks=True)
    cache = cx.cache_directory()
    before = set(cache.glob("*")) if cache.exists() else set()
    executable, _ = cx.compile_program(program, disk_cache=True)
    after = set(cache.glob("*")) if cache.exists() else set()
    assert after == before
    with pytest.raises(ValueError, match="cannot be serialized"):
        executable.serialize()
    with pytest.raises(ValueError, match="cannot be stored"):
        cx.pack_program(cx.BundleWriter("test.program", 1), "callbacks", executable)
    with pytest.raises(ValueError, match="cannot be stored"):
        cx.store_lowered("key", [program])


def test_lowering_without_the_flag_points_at_it() -> None:
    """Checks that the error for a callback names the flag that allows it."""

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda v: None, x)
        return x

    with pytest.raises(cx.HostCallbackError, match="host_callbacks=True"):
        cx.lower(function, SHAPES)


def test_the_handler_is_registered_once() -> None:
    """Checks that registering again does nothing, and that the target has a provider."""
    _native_functions.register_handler()
    _native_functions.register_handler()
    assert cx.target_provider(_targets.NATIVE_TARGET) == "cyxla"
    assert ("cpu", _targets.NATIVE_TARGET) in _ffi.targets


def lowered_run(function: Any, shapes: Any = SHAPES) -> list[np.ndarray]:
    """Lowers a function with callbacks, runs it on `POINT` and returns the results."""
    executable, _ = compiled(function, shapes)
    return executable.run([POINT])


def test_callbacks_inside_control_flow_and_transformations() -> None:
    """Checks that callbacks are found in scans, while loops, nested jits and remat."""
    seen: list[str] = []

    def note(label: str) -> Any:
        return lambda value: seen.append(label)

    def scanned(x: jax.Array) -> jax.Array:
        def body(carry: jax.Array, _: Any) -> tuple[jax.Array, None]:
            jax.debug.callback(note("scan"), carry)
            return carry + 1.0, None

        return jax.lax.scan(body, x, None, length=3)[0]

    def looped(x: jax.Array) -> jax.Array:
        def body(state: tuple[int, jax.Array]) -> tuple[int, jax.Array]:
            jax.debug.callback(note("while"), state[1])
            return state[0] + 1, state[1] * 2.0

        return jax.lax.while_loop(lambda state: state[0] < 2, body, (0, x))[1]

    @jax.jit
    def inner(x: jax.Array) -> jax.Array:
        jax.debug.callback(note("jit"), x)
        return x + 1.0

    def nested(x: jax.Array) -> jax.Array:
        return inner(inner(x))

    @jax.checkpoint
    def remembered(x: jax.Array) -> jax.Array:
        jax.debug.callback(note("remat"), x)
        return jnp.sin(x)

    def checkpointed(x: jax.Array) -> jax.Array:
        return remembered(x)

    for function, label, expected in (
        (scanned, "scan", POINT + 3.0),
        (looped, "while", POINT * 4.0),
        (nested, "jit", POINT + 2.0),
        (checkpointed, "remat", np.sin(POINT)),
    ):
        seen.clear()
        (result,) = lowered_run(function)
        np.testing.assert_allclose(result, expected)
        assert seen == [label] * (
            3 if label == "scan" else 2 if label in ("while", "jit") else 1
        )


def test_callbacks_inside_custom_derivative_rules() -> None:
    """Checks that custom_jvp and custom_vjp functions keep working after the rewrite."""
    seen: list[float] = []

    @jax.custom_jvp
    def with_jvp(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda v: seen.append(float(v[0])), x)
        return x**2

    @with_jvp.defjvp
    def with_jvp_rule(primals: Any, tangents: Any) -> Any:
        (x,), (t,) = primals, tangents
        return with_jvp(x), 2.0 * x * t

    @jax.custom_vjp
    def with_vjp(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda v: seen.append(float(v[0]) + 100.0), x)
        return x**3

    def forward(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return with_vjp(x), x

    def backward(x: jax.Array, cotangent: jax.Array) -> tuple[jax.Array]:
        return (3.0 * x**2 * cotangent,)

    with_vjp.defvjp(forward, backward)

    (value,) = lowered_run(with_jvp)
    np.testing.assert_allclose(value, POINT**2)
    (derivative,) = lowered_run(jax.jacfwd(lambda x: jnp.sum(with_jvp(x))))
    np.testing.assert_allclose(derivative, 2.0 * POINT)
    (gradient,) = lowered_run(jax.grad(lambda x: jnp.sum(with_vjp(x))))
    np.testing.assert_allclose(gradient, 3.0 * POINT**2)
    assert seen[0] == 1.0 and 101.0 in seen


def test_callbacks_under_vmap() -> None:
    """Checks a pure callback with a vmap method, batched before lowering."""

    def function(x: jax.Array) -> jax.Array:
        spec = jax.ShapeDtypeStruct((), x.dtype)
        return jax.vmap(
            lambda value: jax.pure_callback(
                lambda v: np.asarray(v) * 2.0, spec, value, vmap_method="expand_dims"
            )
        )(x)

    (result,) = lowered_run(function)
    np.testing.assert_allclose(result, 2.0 * POINT)


def test_wrong_result_dtype_is_reported() -> None:
    """Checks that a result of another dtype than declared is an error."""

    def function(x: jax.Array) -> jax.Array:
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        return jax.pure_callback(lambda v: np.zeros(3, dtype=np.float32), spec, x)

    executable, _ = compiled(function)
    with pytest.raises(RuntimeError, match="Incorrect output dtype"):
        executable.run([POINT])


def test_repeated_tracing_registers_each_callback_once() -> None:
    """Checks that a function that is traced several times keeps one key per callback."""

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(lambda v: None, x)
        return x

    _, program = compiled(function, custom_calls="detect")
    assert program.callbacks is not None and len(program.callbacks.keys) == 1


def test_open_jaxpr_that_gains_constants_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks the error for an open jaxpr, which has no place for constants."""
    closed = jax.make_jaxpr(lambda x: x + np.arange(3.0))(np.zeros(3))
    assert closed.consts
    monkeypatch.setattr(_callbacks, "rebuild_closed", lambda jaxpr, handle: closed)
    open_jaxpr = jax.make_jaxpr(lambda x: x + 1.0)(np.zeros(3)).jaxpr
    with pytest.raises(NotImplementedError, match="constants cannot be rewritten"):
        _callbacks.rebuild(open_jaxpr, _callbacks.CallbackHandle())


def test_values_kept_after_the_call_are_not_overwritten() -> None:
    """Checks that what a callback keeps stays valid, whatever the runtime does with its buffer.

    JAX aliases the NumPy arrays that it gets, so this fails if the handler passes views of
    the buffers of the runtime and not copies.
    """
    kept: list[Any] = []
    kept_numpy: list[np.ndarray] = []

    def keep(value: Any) -> None:
        kept.append(value)
        kept_numpy.append(np.asarray(value))

    def function(x: jax.Array) -> jax.Array:
        jax.debug.callback(keep, x)
        return x

    executable, _ = compiled(function)
    executable.run([POINT])
    for number in range(1, 20):
        executable.run([POINT * (10.0 + number)])
    np.testing.assert_array_equal(np.asarray(kept[0]), POINT)
    np.testing.assert_array_equal(kept_numpy[0], POINT)
