"""Tests of programs in bundles, and of the executable features they rely on."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cyxla as cx


def compiled_square() -> cx.Executable:
    """Returns the executable of `x ** 2 + 1` for three float64 values."""
    program = cx.lower(lambda x: x**2 + 1, (jax.ShapeDtypeStruct((3,), jnp.float64),))
    executable, _ = cx.compile_program(program, disk_cache=False)
    return executable


def packed(executable: cx.Executable) -> tuple[cx.Bundle, dict[str, Any]]:
    """Returns a bundle that holds an executable, and the record of it."""
    writer = cx.BundleWriter("test.program", 1)
    record = cx.pack_program(writer, "square", executable)
    data = writer.to_bytes({"record": record})
    bundle = cx.Bundle(data, "test.program", 1)
    return bundle, bundle.meta["record"]


def test_loaded_executable_runs_like_the_original() -> None:
    """Checks that the serialized executable of this machine is used, not recompiled."""
    bundle, record = packed(compiled_square())
    executable, statistics = cx.load_program(bundle, record)
    assert (statistics.loaded, statistics.compiled) == (1, 0)
    (result,) = executable.run([np.array([1.0, 2.0, 3.0])])
    np.testing.assert_allclose(result, [2.0, 5.0, 10.0])
    assert executable.input_types == compiled_square().input_types
    assert executable.program is not None
    assert executable.program.platform == "cpu"
    assert executable.output_bytes == (24,)


def test_loaded_executable_can_be_stored_again() -> None:
    """Checks that a loaded executable has the source that storing needs."""
    bundle, record = packed(compiled_square())
    executable, _ = cx.load_program(bundle, record)
    again, again_record = packed(executable)
    second, _ = cx.load_program(again, again_record)
    (result,) = second.run([np.array([0.0, 0.0, 0.0])])
    np.testing.assert_allclose(result, 1.0)


def test_record_lists_custom_calls() -> None:
    """Checks that the targets of custom calls are recorded and collected."""
    program = cx.lower(
        lambda matrix: jnp.linalg.inv(matrix),
        (jax.ShapeDtypeStruct((2, 2), jnp.float64),),
    )
    executable, _ = cx.compile_program(program, disk_cache=False)
    _, with_calls = packed(executable)
    _, without = packed(compiled_square())
    assert with_calls["custom_calls"]
    assert without["custom_calls"] == []
    assert cx.record_custom_calls([with_calls, without]) == tuple(
        sorted(with_calls["custom_calls"])
    )
    assert cx.record_custom_calls([without]) == ()


def test_other_target_recompiles_the_source() -> None:
    """Checks the fallback for a bundle written on another machine."""
    bundle, record = packed(compiled_square())
    record = {
        **record,
        "executables": {
            "elsewhere": record["executables"][cx.target_fingerprint().key]
        },
    }
    executable, statistics = cx.load_program(bundle, record)
    assert (statistics.loaded, statistics.compiled) == (0, 1)
    (result,) = executable.run([np.array([1.0, 2.0, 3.0])])
    np.testing.assert_allclose(result, [2.0, 5.0, 10.0])
    with pytest.raises(cx.TargetMismatchError, match="no executable for this machine"):
        cx.load_program(bundle, record, on_mismatch="raise")


def test_rejected_executable_recompiles_the_source() -> None:
    """Checks that an executable the plugin cannot load falls back to the source."""
    writer = cx.BundleWriter("test.program", 1)
    record = cx.pack_program(writer, "square", compiled_square())
    key = cx.target_fingerprint().key
    writer.blobs[record["executables"][key]] = b"not an executable"
    bundle = cx.Bundle(writer.to_bytes({}), "test.program", 1)
    executable, statistics = cx.load_program(bundle, record)
    assert statistics.compiled == 1
    assert executable.num_outputs == 1
    with pytest.raises(cx.TargetMismatchError, match="cannot be loaded"):
        cx.load_program(bundle, record, on_mismatch="raise")


def test_storing_needs_the_source_program() -> None:
    """Checks the error for an executable that does not know its source."""
    executable = compiled_square()
    executable.program = None
    with pytest.raises(ValueError, match="no lowered program"):
        cx.pack_program(cx.BundleWriter("test.program", 1), "square", executable)


def test_compile_program_records_the_source() -> None:
    """Checks that both compiling and loading from the cache set the program."""
    program = cx.lower(lambda x: x - 1, (jax.ShapeDtypeStruct((2,), jnp.float64),))
    first, first_statistics = cx.compile_program(program)
    second, second_statistics = cx.compile_program(program)
    assert first.program == program == second.program
    assert second_statistics.loaded == 1 or first_statistics.loaded == 1


def test_warm_up_runs_on_zeros() -> None:
    """Checks that an executable warms up, and that one without types cannot."""
    compiled_square().warm_up()
    bare = cx.Executable.deserialize(compiled_square().serialize())
    with pytest.raises(ValueError, match="types of the arguments are not known"):
        bare.warm_up()


def test_deserialize_takes_buffers_on_several_threads() -> None:
    """Checks that views of bytes work, that empty input fails, and that threads can load."""
    data = compiled_square().serialize()
    for buffer in (data, bytearray(data), memoryview(data)[:]):
        assert cx.Executable.deserialize(buffer).num_outputs == 1
    with pytest.raises(ValueError, match="empty"):
        cx.Executable.deserialize(b"")
    with pytest.raises(RuntimeError):
        cx.Executable.deserialize(b"garbage")
    with ThreadPoolExecutor(4) as pool:
        loaded = list(pool.map(cx.Executable.deserialize, [data] * 8))
    assert all(item.num_outputs == 1 for item in loaded)
