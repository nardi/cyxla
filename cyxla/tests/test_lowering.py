"""Tests of lowering for the loaded platform and for others."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import run_python

import cyxla as cx
from cyxla._lowering import array_type, is_wide, lowering_platform

SHAPES = (jax.ShapeDtypeStruct((3,), jnp.float64),)
"""Arguments of the lowered test functions."""


def test_lowering_platform() -> None:
    """Checks the default platform and the validation of names."""
    assert lowering_platform(None) == "cpu"
    assert lowering_platform("CUDA") == "cuda"
    with pytest.raises(ValueError, match="Cannot lower for the platform metal"):
        lowering_platform("metal")


def test_lowering_keeps_unused_arguments() -> None:
    """Checks that an unused argument stays in the signature of the program."""
    program = cx.lower(lambda x, y: jnp.sin(x), (*SHAPES, *SHAPES))
    assert program.text.count("tensor<3xf64>") >= 3


def test_lowering_for_other_platforms() -> None:
    """Checks that programs lower for GPUs without one being present."""
    for platform in ("cuda", "rocm"):
        program = cx.lower(jnp.sin, SHAPES, platform=platform)
        assert "stablehlo.sine" in program.text and program.code


def test_lowering_records_argument_types() -> None:
    """Checks that a program keeps the dtype and shape of each argument."""
    shapes = (
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    program = cx.lower(lambda x, n: x * n, shapes)
    assert program.input_types == (
        cx.ArrayType(np.dtype(np.float32), (2, 3)),
        cx.ArrayType(np.dtype(np.int32), ()),
    )
    assert program.input_types[0].size == 6
    assert program.input_types[0].nbytes == 24


def test_lowering_refuses_unsupported_types() -> None:
    """Checks the errors for dtypes and shapes that the runtime cannot hold."""
    with pytest.raises(ValueError, match="The result 0 has the dtype float8_e4m3fn"):
        cx.lower(lambda x: x.astype(jnp.float8_e4m3fn), SHAPES)
    with pytest.raises(ValueError, match="An argument has the dtype int4"):
        cx.lower(lambda x: x, (jax.ShapeDtypeStruct((2,), jnp.int4),))
    (symbolic,) = jax.export.symbolic_shape("n")
    with pytest.raises(ValueError, match="no dtype or shape"):
        array_type(jax.ShapeDtypeStruct((symbolic,), jnp.float64))


def test_lowering_before_initialization() -> None:
    """Checks that lowering picks the default platform without loading a plugin."""
    result = run_python(
        """
        import jax, jax.numpy as jnp
        import cyxla as cx
        from cyxla import _lowering, _plugins

        assert _lowering.lowering_platform(None) == "cpu"
        _plugins.available_plugins = lambda: ()
        assert _lowering.lowering_platform(None) == "cpu"
        program = cx.lower(jnp.sin, (jax.ShapeDtypeStruct((2,), jnp.float32),))
        assert program.input_types[0].dtype == jnp.float32
        assert not cx.is_initialized()
        """
    )
    assert result.returncode == 0, result.stderr


def test_wide_arguments_lower_in_64_bit_mode() -> None:
    """Checks that 64-bit arguments keep their dtype and that the setting is restored."""
    result = run_python(
        """
        import jax, jax.numpy as jnp
        import cyxla as cx

        assert not jax.config.jax_enable_x64
        for dtype in (jnp.float64, jnp.int64, jnp.uint64, jnp.complex128):
            program = cx.lower(lambda x: x, (jax.ShapeDtypeStruct((2,), dtype),))
            assert program.input_types[0].dtype == dtype, program.input_types
            assert not jax.config.jax_enable_x64
        program = cx.lower(lambda x: x, (jax.ShapeDtypeStruct((2,), jnp.complex64),))
        assert program.input_types[0].dtype == jnp.complex64
        """
    )
    assert result.returncode == 0, result.stderr


def test_is_wide() -> None:
    """Checks which dtypes switch lowering to 64-bit mode, including ones that are not NumPy's."""
    assert is_wide(jax.ShapeDtypeStruct((1,), jnp.float64))
    assert not is_wide(jax.ShapeDtypeStruct((1,), jnp.float32))
    assert not is_wide(jax.ShapeDtypeStruct((1,), jnp.complex64))
    assert not is_wide(jax.ShapeDtypeStruct((), jax.random.key(0).dtype))
