"""Python callbacks inside programs, run through the native function handler of cyxla.

JAX lowers `jax.debug.callback`, `jax.pure_callback` and `jax.experimental.io_callback` to a
custom call of a handler in jaxlib. That handler reads its callbacks from the execution
context of the client of jaxlib, which is empty when a program runs through the PJRT C API.
Calling it crashes the process, so cyxla does not register it. Programs call the built-in
native function `PYTHON_CALLBACK` instead, with the callback number in the attribute
`index`. The native function takes the GIL and calls `dispatch`, which finds the callable in
a registry by that number, calls it with copies of the arguments and copies the results into
the result buffers. An exception becomes the error of the execution.

The GIL is free while a program runs, because `Executable.run` and the extension modules
that call `execute` release it. A caller that holds the GIL during an execution would
deadlock.
"""

import contextlib
import ctypes
import functools
import itertools
import logging
import weakref
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from cyxla import _native, _native_functions

PYTHON_CALLBACK = "cyxla.python_callback"
"""The name of the native function that programs with callbacks call."""

Callback = Callable[..., Sequence[ArrayLike]]
"""A callback as JAX wraps it: it takes arrays and returns a sequence of arrays."""

callbacks: dict[int, Callback] = {}
"""The callbacks that programs may call, by the integer that the custom call carries."""

identifiers = itertools.count()
"""Source of the keys of `callbacks`, which are unique in the process."""


@functools.cache
def element_types() -> dict[int, np.dtype]:
    """Returns the NumPy type for each element type of the FFI API that callbacks can take."""
    import ml_dtypes

    return {
        1: np.dtype(np.bool_),
        2: np.dtype(np.int8),
        3: np.dtype(np.int16),
        4: np.dtype(np.int32),
        5: np.dtype(np.int64),
        6: np.dtype(np.uint8),
        7: np.dtype(np.uint16),
        8: np.dtype(np.uint32),
        9: np.dtype(np.uint64),
        10: np.dtype(np.float16),
        11: np.dtype(np.float32),
        12: np.dtype(np.float64),
        15: np.dtype(np.complex64),
        16: np.dtype(ml_dtypes.bfloat16),
        18: np.dtype(np.complex128),
    }


ArraySpecification = tuple[int, tuple[int, ...], int, int]
"""The element type code, the shape, the element count and the address of an array."""


def buffer_array(specification: ArraySpecification, writable: bool) -> np.ndarray:
    """Returns an array that views the memory of an argument or a result.

    The array views memory of the runtime, so the caller copies an argument. A buffer
    without elements has no memory to view.
    """
    code, shape, size, address = specification
    dtypes = element_types()
    if code not in dtypes:
        raise TypeError(
            f"A callback cannot take elements of the FFI element type {code}."
        )
    dtype = dtypes[code]
    if size == 0:
        return np.zeros(shape, dtype)
    memory = (ctypes.c_char * (size * dtype.itemsize)).from_address(address)
    view = np.frombuffer(memoryview(memory), dtype=dtype).reshape(shape)
    view.flags.writeable = writable
    return view


def dispatch(
    index: int,
    argument_specifications: Sequence[ArraySpecification],
    result_specifications: Sequence[ArraySpecification],
) -> None:
    """Calls the callback with a number and copies its results into the result buffers.

    The native function calls this with the GIL held. An exception ends the execution, and
    the native function reports its type and message.
    """
    if index not in callbacks:
        raise LookupError(
            f"The callback {index} was released, because the program that calls it was "
            "collected."
        )
    # The arguments are copied. A view of the buffers would be valid during the call only,
    # and `jax.device_put` aliases a NumPy array that it receives instead of copying it. A
    # callback that keeps a value would then see the memory of the runtime change under it.
    arguments = [
        buffer_array(specification, False).copy()
        for specification in argument_specifications
    ]
    values = callbacks[index](*arguments)
    if len(values) != len(result_specifications):
        raise RuntimeError(
            f"The callback returned {len(values)} results, but the program expects "
            f"{len(result_specifications)}."
        )
    for position, value in enumerate(values):
        destination = buffer_array(result_specifications[position], True)
        result = np.asarray(value)
        if result.shape != destination.shape:
            raise RuntimeError(
                f"Incorrect output shape for return value #{position}: Expected: "
                f"{destination.shape}, Actual: {result.shape}"
            )
        if result.dtype != destination.dtype:
            raise RuntimeError(
                f"Incorrect output dtype for return value #{position}: Expected: "
                f"{destination.dtype}, Actual: {result.dtype}"
            )
        destination[...] = result


_native.set_python_dispatch(dispatch)


def release(keys: Sequence[int]) -> None:
    """Removes callbacks from the registry."""
    for key in keys:
        callbacks.pop(key, None)


class CallbackHandle:
    """Keeps the callbacks of a program registered for as long as the program exists.

    The program holds the handle, and so does every executable that was compiled from it.
    Collecting the last of them removes the callbacks from the registry.
    """

    def __init__(self) -> None:
        self.keys: list[int] = []
        """The keys in `callbacks` that the program calls."""
        self.by_equation: dict[int, int] = {}
        """The key of the callback of each equation, by the identity of the equation."""
        self.finalizer = weakref.finalize(self, release, self.keys)

    def register(self, equation: Any, callback: Callback) -> int:
        """Returns the key of the callback of an equation, registering it the first time.

        A function that is traced more than once meets the same equation each time, and
        is assigned the same key. The equation is typed `Any` because it is a `JaxprEqn`.
        """
        if id(equation) not in self.by_equation:
            key = next(identifiers)
            callbacks[key] = callback
            self.keys.append(key)
            self.by_equation[id(equation)] = key
        return self.by_equation[id(equation)]


def array_structure(aval: Any) -> Any:
    """Returns the `jax.ShapeDtypeStruct` of an abstract value, which is typed `Any`."""
    import jax

    return jax.ShapeDtypeStruct(
        aval.shape, aval.dtype, weak_type=bool(getattr(aval, "weak_type", False))
    )


def is_callback(equation: Any) -> bool:
    """Returns whether an equation of a jaxpr calls back into Python.

    JAX names these primitives `debug_callback`, `pure_callback` and `io_callback`. The
    equation is typed `Any` because it is a `JaxprEqn`.
    """
    return bool(equation.primitive.name.endswith("callback"))


def host_function(equation: Any) -> Callback:
    """Returns the function that runs the callback of an equation on the host.

    The `callback` param of the three callback primitives takes the flat arguments and
    returns the flat results. The function does what JAX does around it: it places the
    arguments on the CPU, runs the callback there, logs an exception and converts the
    results to NumPy arrays. The `impl` rule of `pure_callback` cannot be used, since it
    runs the primitive through the runtime of JAX again. The equation is typed `Any` because
    it is a `JaxprEqn`.
    """
    import jax

    callback = equation.params["callback"]
    name = equation.primitive.name

    def run(*arguments: Any) -> list[np.ndarray]:
        cpu = jax.local_devices(backend="cpu")[0]
        try:
            with jax.default_device(cpu):
                result = callback(*jax.device_put(arguments, cpu))
        except BaseException:
            logging.getLogger(__name__).exception("%s failed", name)
            raise
        return [np.asarray(item) for item in jax.tree.leaves(result)]

    return run


def emit_callback(
    equation: Any, arguments: list[Any], handle: CallbackHandle
) -> list[Any]:
    """Binds the custom call of cyxla in place of an equation that calls into Python.

    The callable is `host_function` of the equation. The equation and the arguments are
    typed `Any` because they are a `JaxprEqn` and traced values.
    """
    import jax

    if any("Ordered" in type(effect).__name__ for effect in equation.effects):
        raise NotImplementedError(
            "The function has ordered callbacks (`ordered=True`), whose tokens the "
            "runtime of cyxla cannot pass. Use unordered callbacks."
        )
    key = handle.register(equation, host_function(equation))
    structures = [array_structure(variable.aval) for variable in equation.outvars]
    results = _native_functions.native_call(
        PYTHON_CALLBACK,
        structures,
        *arguments,
        has_side_effect=bool(equation.effects),
        index=np.uint64(key),
    )
    return list(jax.tree.leaves(results))


def contains_jaxpr(value: Any) -> bool:
    """Returns whether a param of an equation is a jaxpr or a tuple or list of them."""
    from jax.extend import core

    if isinstance(value, core.ClosedJaxpr | core.Jaxpr):
        return True
    if type(value) in (tuple, list):
        return any(contains_jaxpr(item) for item in value)
    return False


def rebuild(value: Any, handle: CallbackHandle) -> Any:
    """Returns a param of an equation with the callbacks of its jaxprs replaced.

    Params without a jaxpr are returned as they are. A jaxpr is rebuilt by tracing the
    interpreter over it, so that its effects come out right. An open jaxpr must have no
    constants. The value is typed `Any` because a param can be any object.
    """
    from jax.extend import core

    if not contains_jaxpr(value):
        return value
    if isinstance(value, core.ClosedJaxpr):
        return rebuild_closed(value, handle)
    if isinstance(value, core.Jaxpr):
        closed = rebuild_closed(core.ClosedJaxpr(value, ()), handle)
        if closed.consts:
            raise NotImplementedError("A jaxpr with constants cannot be rewritten.")
        return closed.jaxpr
    return type(value)(rebuild(item, handle) for item in value)


def rebuild_closed(closed: Any, handle: CallbackHandle) -> Any:
    """Returns a closed jaxpr whose callbacks are calls of the handler of cyxla.

    The jaxpr is typed `Any` because it is a `ClosedJaxpr`.
    """
    import jax

    structures = [array_structure(variable.aval) for variable in closed.jaxpr.invars]

    def run(*arguments: Any) -> list[Any]:
        return interpret(closed.jaxpr, closed.consts, arguments, handle)

    return jax.make_jaxpr(run)(*structures)


def interpret(
    jaxpr: Any,
    constants: Sequence[Any],
    arguments: Sequence[Any],
    handle: CallbackHandle,
) -> list[Any]:
    """Evaluates a jaxpr with its Python callbacks replaced by the handler of cyxla.

    The function works like `jax.core.eval_jaxpr`. It runs inside a trace, so the result is
    a program that calls the handler. Equations with jaxprs in their params are bound again
    with rewritten jaxprs, and the params are rewritten before JAX derives the bind params,
    because the rules of `custom_jvp_call` and `custom_vjp_call` take the function from
    them. The jaxpr is typed `Any` because it is a `Jaxpr`, and the values because they are
    traced.
    """
    from jax.extend import core

    environment: dict[Any, Any] = {}

    def read(atom: Any) -> Any:
        return atom.val if isinstance(atom, core.Literal) else environment[atom]

    environment.update(zip(jaxpr.constvars, constants, strict=True))
    environment.update(zip(jaxpr.invars, arguments, strict=True))
    for equation in jaxpr.eqns:
        inputs = [read(atom) for atom in equation.invars]
        if is_callback(equation):
            outputs = emit_callback(equation, inputs, handle)
        else:
            params = {
                name: rebuild(value, handle) for name, value in equation.params.items()
            }
            bind_params = equation.primitive.get_bind_params(params)
            # The context holds the mesh and similar settings that the equation was traced
            # in.
            context = getattr(getattr(equation, "ctx", None), "manager", None)
            with context if context is not None else contextlib.nullcontext():
                result = equation.primitive.bind(*inputs, **bind_params)
            outputs = list(result) if equation.primitive.multiple_results else [result]
        environment.update(zip(equation.outvars, outputs, strict=True))
    return [read(atom) for atom in jaxpr.outvars]


def rewrite_callbacks(closed: Any, handle: CallbackHandle) -> Callable[..., list[Any]]:
    """Returns a function that runs a jaxpr with its Python callbacks sent to the handler.

    Registering the handler for native functions is part of it. The jaxpr is typed `Any`
    because it is a `ClosedJaxpr`. This module imports nothing private from JAX. It finds
    the callbacks by the name of their primitive and runs the `callback` param of the
    equation.
    """
    _native_functions.register_handler()

    def run(*arguments: Any) -> list[Any]:
        return interpret(closed.jaxpr, closed.consts, arguments, handle)

    return run
