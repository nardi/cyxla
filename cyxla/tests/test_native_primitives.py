"""Tests of primitives for native functions, and of derivatives around native calls."""

import ctypes
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.interpreters import ad

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


def spec(shape: tuple[int, ...] = (3,)) -> jax.ShapeDtypeStruct:
    """Returns the structure of a float64 result."""
    return jax.ShapeDtypeStruct(shape, jnp.float64)


class ScaleKernel:
    """A kernel for the function that multiplies by a factor, which is linear.

    It stands for a Cython `cdef class`: the name and the address are attributes, the
    function is registered with the primitive, and the rules are methods.
    """

    name = "test.primitive_scale"
    linear = True
    user_data_address = 0

    def __init__(self, library: ctypes.CDLL) -> None:
        self.function_address = address(library, "scale")
        self.scale: Callable[..., Any] | None = None

    def result_avals(self, x: Any, *, factor: float) -> Sequence[Any]:
        """The result has the type of the argument."""
        return (x,)

    def transpose(self, cotangents: Any, x: Any, *, factor: float) -> list[Any]:
        """The function is symmetric, so its transpose is itself."""
        (cotangent,) = cotangents
        assert self.scale is not None
        if type(cotangent) is ad.Zero:
            return [ad.Zero(x.aval)]
        return [self.scale(cotangent, factor=factor)]

    def batch(
        self, arguments: Any, dimensions: Any, *, factor: float
    ) -> tuple[Any, Any]:
        """The function works on any shape, so a batch dimension stays where it is."""
        assert self.scale is not None
        return [self.scale(arguments[0], factor=factor)], [dimensions[0]]


class SquareKernel:
    """A kernel for squaring, whose derivative is written with `jnp` operations."""

    name = "test.primitive_square"
    user_data_address = 0

    def __init__(self, library: ctypes.CDLL) -> None:
        self.function_address = address(library, "square")
        self.square: Callable[..., Any] | None = None

    def result_avals(self, x: Any) -> Sequence[Any]:
        """The result has the type of the argument."""
        return (x,)

    def jvp(self, primals: Any, tangents: Any) -> tuple[list[Any], list[Any]]:
        """Differentiates with `jnp`, so the derivative can be differentiated again."""
        assert self.square is not None
        (x,), (tangent,) = primals, tangents
        return [self.square(x)], [2.0 * x * tangent]


@pytest.fixture(scope="module")
def primitives(native_library_path: Path) -> SimpleNamespace:
    """Registers the functions of the test library and builds the primitives."""
    library = ctypes.CDLL(str(native_library_path))
    scale_kernel = ScaleKernel(library)
    scale_kernel.scale = cx.native_primitive(scale_kernel)
    square_kernel = SquareKernel(library)
    square_kernel.square = cx.native_primitive(square_kernel)
    for name, function in (
        ("test.scale_named", "scale"),
        ("test.square_named", "square"),
        ("test.square_backward_named", "square_backward"),
    ):
        cx.register_native_function(name, address(library, function))
    return SimpleNamespace(scale=scale_kernel.scale, square=square_kernel.square)


def in_cyxla(function: Callable[..., Any]) -> np.ndarray:
    """Lowers a function of one array, compiles it in cyxla and runs it on `POINT`."""
    executable, _ = cx.compile_program(cx.lower(function, SHAPES), disk_cache=False)
    return executable.run([POINT])[0]


def test_primitive_works_in_jit_and_in_cyxla(primitives: SimpleNamespace) -> None:
    """Checks the primitive under jit in plain JAX and in a program of cyxla."""

    def function(x: jax.Array) -> jax.Array:
        return primitives.scale(x, factor=3.0) + 1.0

    expected = 3.0 * POINT + 1.0
    np.testing.assert_array_equal(jax.jit(function)(POINT), expected)
    np.testing.assert_array_equal(in_cyxla(function), expected)


def test_linear_primitive_has_forward_and_reverse_derivatives(
    primitives: SimpleNamespace,
) -> None:
    """Checks jvp, linearize and grad (which needs the transpose rule) of a linear kernel."""

    def loss(x: jax.Array) -> jax.Array:
        return jnp.sum(primitives.scale(x, factor=3.0) ** 2)

    gradient = 18.0 * POINT
    primal, tangent = jax.jvp(loss, (POINT,), (np.ones(3),))
    np.testing.assert_allclose(primal, 9.0 * np.sum(POINT**2))
    np.testing.assert_allclose(tangent, np.sum(gradient))
    _, linear = jax.linearize(loss, POINT)
    np.testing.assert_allclose(linear(np.ones(3)), np.sum(gradient))
    np.testing.assert_allclose(jax.grad(loss)(POINT), gradient)
    np.testing.assert_allclose(in_cyxla(jax.grad(loss)), gradient)


def test_linear_primitive_batches(primitives: SimpleNamespace) -> None:
    """Checks vmap over a native function, in JAX and in a program of cyxla."""
    batch = np.arange(6.0).reshape(2, 3)

    def function(x: jax.Array) -> jax.Array:
        return jax.vmap(lambda row: primitives.scale(row, factor=2.0))(x)

    np.testing.assert_array_equal(jax.jit(function)(batch), 2.0 * batch)
    program = cx.lower(function, (jax.ShapeDtypeStruct((2, 3), jnp.float64),))
    executable, _ = cx.compile_program(program, disk_cache=False)
    np.testing.assert_array_equal(executable.run([batch])[0], 2.0 * batch)


def test_nonlinear_primitive_has_second_derivatives(
    primitives: SimpleNamespace,
) -> None:
    """Checks a jvp rule written with jnp operations, which differentiates again."""

    def loss(x: jax.Array) -> jax.Array:
        return jnp.sum(primitives.square(x))

    np.testing.assert_allclose(jax.grad(loss)(POINT), 2.0 * POINT)
    np.testing.assert_allclose(jax.hessian(loss)(POINT), 2.0 * np.eye(3))
    np.testing.assert_allclose(in_cyxla(jax.grad(loss)), 2.0 * POINT)
    hessian = cx.compile_program(cx.lower(jax.hessian(loss), SHAPES), disk_cache=False)[
        0
    ].run([POINT])[0]
    np.testing.assert_allclose(hessian, 2.0 * np.eye(3))


def test_custom_vjp_around_native_calls(primitives: SimpleNamespace) -> None:
    """Checks a function and its backward pass that are both native functions."""

    @jax.custom_vjp
    def square(x: jax.Array) -> jax.Array:
        return cx.native_call("test.square_named", spec(), x)

    def forward(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return square(x), x

    def backward(x: jax.Array, cotangent: jax.Array) -> tuple[jax.Array]:
        return (cx.native_call("test.square_backward_named", spec(), x, cotangent),)

    square.defvjp(forward, backward)

    def loss(x: jax.Array) -> jax.Array:
        return jnp.sum(square(x))

    np.testing.assert_allclose(jax.grad(loss)(POINT), 2.0 * POINT)
    np.testing.assert_allclose(in_cyxla(jax.grad(loss)), 2.0 * POINT)


def test_custom_jvp_around_a_native_call(primitives: SimpleNamespace) -> None:
    """Checks forward mode through a `custom_jvp` whose tangent calls the same function."""

    @jax.custom_jvp
    def scale(x: jax.Array) -> jax.Array:
        return cx.native_call("test.scale_named", spec(), x, factor=5.0)

    @scale.defjvp
    def scale_jvp(primals: Any, tangents: Any) -> tuple[Any, Any]:
        (x,), (tangent,) = primals, tangents
        return scale(x), cx.native_call("test.scale_named", spec(), tangent, factor=5.0)

    primal, tangent = jax.jvp(scale, (POINT,), (np.ones(3),))
    np.testing.assert_array_equal(primal, 5.0 * POINT)
    np.testing.assert_array_equal(tangent, 5.0 * np.ones(3))
    np.testing.assert_array_equal(
        in_cyxla(lambda x: jax.jvp(scale, (x,), (jnp.ones_like(x),))[1]),
        5.0 * np.ones(3),
    )


def test_programs_with_primitives_record_their_functions(
    primitives: SimpleNamespace,
) -> None:
    """Checks that a program with a primitive stores like any program with native calls."""
    program = cx.lower(lambda x: primitives.scale(x, factor=2.0), SHAPES)
    executable, _ = cx.compile_program(program, disk_cache=False)
    record = cx.pack_program(
        cx.BundleWriter("test.program", 1), "primitive", executable
    )
    assert set(record["native_functions"]) == {"test.primitive_scale"}


def test_function_kernel_without_a_class(native_library_path: Path) -> None:
    """Checks the function form, with the rules as functions, and a kernel without rules."""
    library = ctypes.CDLL(str(native_library_path))
    plain = cx.native_primitive(
        cx.NativeFunctionKernel(
            name="test.plain_square",
            function_address=address(library, "square"),
            result_avals=lambda x: (x,),
        )
    )
    np.testing.assert_array_equal(jax.jit(plain)(POINT), POINT**2)
    with pytest.raises(Exception, match="[Dd]ifferentiation rule"):
        jax.grad(lambda x: jnp.sum(plain(x)))(POINT)
