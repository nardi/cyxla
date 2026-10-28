"""Tests of lowering for the loaded platform and for others."""

from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import run_python

import cyxla as cx
from cyxla import _plugins
from cyxla._lowering import (
    array_type,
    jax_platform_name,
    lowering_platform,
    placement_platform,
)

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


def test_64_bit_shapes_follow_the_setting_of_jax() -> None:
    """Checks that a `float64` shape lowers to 64 bits only while JAX's 64-bit mode is on."""
    result = run_python(
        """
        import warnings

        import jax, jax.numpy as jnp
        import cyxla as cx

        shapes = (jax.ShapeDtypeStruct((2,), jnp.float64),)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert cx.lower(lambda x: x, shapes).input_types[0].dtype == jnp.float32
        with jax.enable_x64(True):
            assert cx.lower(lambda x: x, shapes).input_types[0].dtype == jnp.float64
        """
    )
    assert result.returncode == 0, result.stderr


class FakeDevice(NamedTuple):
    """A device of JAX, which only has a platform and an index."""

    platform: str
    index: int


def placed(*platforms: str, committed: bool = True) -> SimpleNamespace:
    """Returns an argument that is placed on devices of the given platforms."""
    sharding = SimpleNamespace(
        device_set={FakeDevice(name, index) for index, name in enumerate(platforms)}
    )
    return SimpleNamespace(sharding=sharding, committed=committed)


def test_placement_chooses_the_platform() -> None:
    """Checks that the devices of committed arguments choose the platform to lower for."""
    assert placement_platform([]) is None
    assert placement_platform([SimpleNamespace(sharding=None)]) is None
    assert placement_platform([placed("cpu"), placed("cpu")]) == "cpu"
    assert placement_platform([placed("cpu", committed=False)]) is None
    assert placement_platform([placed("tpu"), placed("cpu", committed=False)]) == "tpu"
    assert lowering_platform(None, [placed("tpu")]) == "tpu"
    assert lowering_platform("cpu", [placed("tpu")]) == "cpu"
    with pytest.raises(ValueError, match="different platforms: cpu, tpu"):
        placement_platform([placed("cpu"), placed("tpu")])
    with pytest.raises(ValueError, match="placed on 2 devices"):
        placement_platform([placed("cpu", "cpu")])


def test_jax_names_cuda_and_rocm_devices_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that the platform `gpu` of JAX maps to the GPU plugin that is installed."""
    assert jax_platform_name("CPU") == "cpu"
    assert jax_platform_name("cuda") == "cuda"
    assert jax_platform_name("gpu") == "cuda"
    rocm = cx.Plugin("rocm", Path("libpjrt_rocm.so"), "test")
    cuda = rocm._replace(platform="cuda")
    monkeypatch.setattr(_plugins, "available_plugins", lambda: (rocm,))
    assert jax_platform_name("gpu") == "rocm"
    monkeypatch.setattr(_plugins, "available_plugins", lambda: (rocm, cuda))
    assert jax_platform_name("gpu") == "cuda"


def test_lowering_follows_arrays_and_the_default_device() -> None:
    """Checks lowering for committed arrays, shapes with a sharding and the default device."""
    device = jax.devices("cpu")[0]
    sharding = jax.sharding.SingleDeviceSharding(device)
    shape = jax.ShapeDtypeStruct((3,), jnp.float32, sharding=sharding)
    array = jax.device_put(jnp.arange(3.0), device)
    for argument in (shape, array):
        program = cx.lower(lambda x: x + 1, [argument])
        assert program.platform == "cpu"
    with jax.default_device(device):
        assert lowering_platform(None) == "cpu"
    with jax.default_device("cpu"):
        assert lowering_platform(None) == "cpu"
    assert lowering_platform(None, [placed("tpu")]) == "tpu"


def test_lowering_platform_before_initialization() -> None:
    """Checks the platforms that `jax_platforms` and the loaded plugin lead to."""
    code = """
        import jax
        import cyxla as cx
        from cyxla import _lowering

        assert _lowering.lowering_platform(None) == {expected!r}
        if {load!r}:
            cx.runtime_info()
            assert _lowering.lowering_platform(None) == "cpu"
        """
    cases = [
        ("cuda,cpu", False, "cpu"),
        ("cuda", False, "cpu"),
        ("tpu, cpu", False, "cpu"),
        ("cuda", True, "cpu"),
    ]
    for platforms, load, expected in cases:
        result = run_python(
            code.format(expected=expected, load=load),
            {"JAX_PLATFORMS": platforms},
        )
        assert result.returncode == 0, (platforms, result.stderr)


def test_programs_choose_their_plugin() -> None:
    """Checks that compiling loads the plugin of the platform that the program was lowered for."""
    result = run_python(
        """
        import jax, jax.numpy as jnp
        import pytest
        import cyxla as cx

        shapes = (jax.ShapeDtypeStruct((2,), jnp.float32),)
        other = cx.lower(jnp.sin, shapes, platform="cuda")
        assert other.platform == "cuda"
        with pytest.raises(RuntimeError, match="No PJRT plugin for the platform cuda"):
            cx.compile_program(other)
        assert not cx.is_initialized()
        program = cx.lower(jnp.sin, shapes, platform="cpu")
        cx.compile_program(program, disk_cache=False)
        assert cx.runtime_info().platform == "cpu"
        with pytest.raises(RuntimeError, match="lowered for cuda, but the loaded plugin runs cpu"):
            cx.compile_program(other)
        """
    )
    assert result.returncode == 0, result.stderr
