"""Tests of lowering for the loaded platform and for others."""

import jax
import jax.numpy as jnp
import pytest

import cyxla as cx
from cyxla._lowering import lowering_platform

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
