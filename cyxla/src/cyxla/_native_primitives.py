"""JAX primitives that call native functions, with derivative rules supplied by the caller.

`native_call` has no derivative, as `jax.ffi.ffi_call` has none. `native_primitive` wraps a
native function in a primitive of its own, so the function works under `jit`, `vmap`,
`grad`, `jvp` and `linearize`. A kernel is any object that satisfies the protocol
`NativeKernel`, such as a Cython `cdef class` that holds the state of the function. The
rules of differentiation and batching are methods of the kernel that JAX calls with the
params of the primitive.

A program that calls the primitive holds nothing but the call of the native function, so
it can be stored and loaded as any program that calls a native function.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from cyxla import _native_functions


class NativeKernel(Protocol):
    """The members of a native function that `native_primitive` needs.

    The members below `result_avals` are optional. A kernel that lacks one has no such rule.
    The kernel is kept alive for the life of the process, so a `user_data_address` that
    points into the kernel stays valid.
    """

    name: str
    """The name under which the function is registered, such as `mylib.scale`."""
    function_address: int
    """The address of the C function, of the type `CyxlaNativeFunction`."""
    user_data_address: int
    """The address of state that the function receives, or 0 for none."""

    def result_avals(self, *argument_avals: Any, **params: Any) -> Sequence[Any]:
        """Returns the abstract values of the results for the abstract values of the arguments.

        An abstract value has a `shape` and a `dtype`. The params are the keyword arguments
        of the call, which become the attributes of the call of the function.
        """
        ...


@dataclasses.dataclass(frozen=True)
class NativeFunctionKernel:
    """A kernel made from the pieces of a function that has no class of its own.

    Every field besides the address and the name is optional. The rules take the arguments
    that the methods of `NativeKernel` take, without `self`: `jvp(primals, tangents,
    **params)` returns the primal results and the tangents of the results,
    `transpose(cotangents, *arguments, **params)` returns a cotangent for each argument,
    and `batch(arguments, dimensions, **params)` returns the results and their batch
    dimensions. `linear` says that the function is linear in all its arguments, which gives
    it the rule of differentiation of applying itself to the tangents.
    """

    name: str
    function_address: int
    result_avals: Callable[..., Sequence[Any]]
    user_data_address: int = 0
    linear: bool = False
    jvp: Callable[..., Any] | None = None
    transpose: Callable[..., Any] | None = None
    batch: Callable[..., Any] | None = None


kernels: dict[str, Any] = {}
"""The kernels that were turned into primitives, by name, which keeps their state alive."""


def native_primitive(kernel: Any) -> Callable[..., Any]:
    """Returns a function that binds a primitive for a native function.

    The kernel satisfies `NativeKernel` and is typed `Any` because the optional rules are
    found with `getattr`. The returned function takes arrays and, as keyword arguments, the
    params of the call, which have to be numbers or strings. It returns one array, or a
    tuple of arrays for a function with several results. The function is registered under
    `kernel.name` and the primitive is new for every call of `native_primitive`.
    """
    import jax
    from jax.extend import core
    from jax.interpreters import ad, batching, mlir

    _native_functions.register_native_function(
        kernel.name, kernel.function_address, kernel.user_data_address
    )
    kernels[kernel.name] = kernel
    primitive = core.Primitive(f"cyxla_native_{kernel.name}")
    primitive.multiple_results = True

    def call(*arguments: Any, **params: Any) -> list[Any]:
        avals = [jax.typeof(argument) for argument in arguments]
        structures = [
            jax.ShapeDtypeStruct(aval.shape, aval.dtype)
            for aval in kernel.result_avals(*avals, **params)
        ]
        return list(
            jax.tree.leaves(
                _native_functions.native_call(
                    kernel.name, structures, *arguments, **params
                )
            )
        )

    def abstract_eval(*avals: Any, **params: Any) -> list[Any]:
        return list(kernel.result_avals(*avals, **params))

    primitive.def_abstract_eval(abstract_eval)
    primitive.def_impl(call)
    mlir.register_lowering(primitive, mlir.lower_fun(call, multiple_results=True))

    def bind(*arguments: Any, **params: Any) -> Any:
        results = primitive.bind(*arguments, **params)
        return results[0] if len(results) == 1 else tuple(results)

    jvp = getattr(kernel, "jvp", None)
    if getattr(kernel, "linear", False):

        def linear_jvp(primals: Any, tangents: Any, **params: Any) -> tuple[Any, Any]:
            tangents = [ad.instantiate_zeros(tangent) for tangent in tangents]
            return primitive.bind(*primals, **params), primitive.bind(
                *tangents, **params
            )

        ad.primitive_jvps[primitive] = linear_jvp
    elif jvp is not None:

        def user_jvp(primals: Any, tangents: Any, **params: Any) -> tuple[Any, Any]:
            primal_out, tangent_out = jvp(primals, tangents, **params)
            return list(primal_out), list(tangent_out)

        ad.primitive_jvps[primitive] = user_jvp
    transpose = getattr(kernel, "transpose", None)
    if transpose is not None:
        ad.primitive_transposes[primitive] = transpose
    batch = getattr(kernel, "batch", None)
    if batch is not None:

        def batcher(arguments: Any, dimensions: Any, **params: Any) -> tuple[Any, Any]:
            results, result_dimensions = batch(arguments, dimensions, **params)
            return list(results), list(result_dimensions)

        batching.primitive_batchers[primitive] = batcher
    return bind
