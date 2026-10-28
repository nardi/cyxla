"""Tests of FFI handlers: the kernels of jnp.linalg and the targets of other libraries."""

import ctypes
import importlib
import itertools
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import run_python
from jaxlib import xla_client

import cyxla as cx
from cyxla import _ffi, _runtime

MATRIX = np.random.default_rng(0).normal(size=(4, 4))
"""A fixed square matrix that is almost surely invertible."""

SYMMETRIC = MATRIX @ MATRIX.T + 4 * np.eye(4)
"""A fixed symmetric positive definite matrix."""

COUNTER = itertools.count()
"""Numbers for target names, because the plugin keeps a registration for the whole process."""

LINEAR_ALGEBRA: dict[str, tuple[Callable[[Any], Any], np.ndarray]] = {
    "cholesky": (jnp.linalg.cholesky, SYMMETRIC),
    "solve": (lambda a: jnp.linalg.solve(a, jnp.arange(4.0)), MATRIX),
    "inv": (jnp.linalg.inv, MATRIX),
    "det": (lambda a: jnp.linalg.det(a)[None], MATRIX),
    "slogdet": (lambda a: jnp.linalg.slogdet(a)[1][None], MATRIX),
    "qr": (lambda a: jnp.abs(jnp.linalg.qr(a)[1]), MATRIX),
    "eigh": (lambda a: jnp.linalg.eigh(a)[0], SYMMETRIC),
    "svd": (lambda a: jnp.linalg.svd(a, compute_uv=False), MATRIX),
    "pinv": (jnp.linalg.pinv, MATRIX),
    "lstsq": (lambda a: jnp.linalg.lstsq(a, jnp.arange(4.0))[0], MATRIX),
    "eigvals": (lambda a: jnp.sort(jnp.abs(jnp.linalg.eigvals(a))), MATRIX),
    "lu": (lambda a: jax.scipy.linalg.lu(a)[2], MATRIX),
    "triangular": (
        lambda a: jax.scipy.linalg.solve_triangular(
            jnp.tril(a) + 5 * jnp.eye(4), jnp.arange(4.0), lower=True
        ),
        MATRIX,
    ),
    "expm": (lambda a: jax.scipy.linalg.expm(0.1 * a), MATRIX),
    "schur": (lambda a: jnp.abs(jax.scipy.linalg.schur(a)[0]), MATRIX),
}
"""Functions of jnp.linalg and jax.scipy.linalg, each with the matrix it runs on."""


def run_program(function: Callable[..., Any], argument: np.ndarray) -> np.ndarray:
    """Lowers, compiles and runs a function of one array through the plugin."""
    program = cx.lower(function, (jax.ShapeDtypeStruct(argument.shape, jnp.float64),))
    executable, _ = cx.compile_program(program, disk_cache=False)
    expected = np.asarray(function(argument))
    (result,) = executable.run([np.ascontiguousarray(argument)])
    assert result.shape == expected.shape
    return result


def test_complex_results_match_jax() -> None:
    """Checks that the eigenvalues of a general matrix come back as complex numbers."""
    function = jnp.linalg.eigvals
    result = run_program(function, MATRIX)
    assert result.dtype == np.complex128
    np.testing.assert_allclose(result, jax.jit(function)(MATRIX), atol=1e-10)


@pytest.mark.parametrize("name", LINEAR_ALGEBRA)
def test_linear_algebra_matches_jax(name: str) -> None:
    """Checks that programs with LAPACK custom calls run on the plugin like in JAX."""
    function, argument = LINEAR_ALGEBRA[name]
    expected = np.asarray(jax.jit(function)(argument))
    np.testing.assert_allclose(run_program(function, argument), expected, atol=1e-10)


def test_jax_targets_are_registered() -> None:
    """Checks that initialization registered the typed handlers of jaxlib for the CPU."""
    names = {name for platform, name in _ffi.targets if platform == "cpu"}
    assert {"lapack_dpotrf_ffi", "lapack_dgetrf_ffi", "lapack_dgesdd_ffi"} <= names
    assert _runtime.supports_ffi()
    assert cx.register_jax_targets() == 0


def test_cached_executables_run_in_a_fresh_process() -> None:
    """Checks that a cached executable with LAPACK calls runs without lowering again.

    The second process never traces the function, which is what loads the LAPACK functions
    of SciPy for JAX. The initialization of cyxla has to do it.
    """
    key = cx.cache_key("cholesky test")
    program = cx.lower(
        jnp.linalg.cholesky, (jax.ShapeDtypeStruct((4, 4), jnp.float64),)
    )
    cx.store_lowered(key, [program])
    code = f"""
        import numpy as np
        import cyxla as cx

        cx.initialize(platform="cpu")
        (program,) = cx.load_lowered({key!r})
        executable, statistics = cx.compile_program(program)
        matrix = np.array({SYMMETRIC.tolist()!r})
        (result,) = executable.run([matrix])
        assert np.allclose(result, np.linalg.cholesky(matrix))
        print(statistics.loaded, statistics.compiled)
        """
    cx.compile_program(program)
    result = run_python(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["1", "0"]


def test_programs_need_registered_targets_without_jax_targets() -> None:
    """Checks that turning off the jaxlib handlers leaves LAPACK programs uncompilable."""
    result = run_python(
        """
        import jax, jax.numpy as jnp
        import cyxla as cx

        cx.enable_x64()
        cx.initialize(platform="cpu", jax_targets=False)
        program = cx.lower(jnp.linalg.cholesky, (jax.ShapeDtypeStruct((3, 3), jnp.float64),))
        try:
            cx.compile_program(program, disk_cache=False)
        except RuntimeError as error:
            assert "No FFI handler registered for lapack_dpotrf_ffi" in str(error)
            assert "cyxla.register_ffi_target" in str(error)
        else:
            raise AssertionError("The program compiled.")
        """
    )
    assert result.returncode == 0, result.stderr


def scale_program(target: str) -> cx.LoweredProgram:
    """Lowers a function that calls a target that multiplies a vector by three, plus one."""

    def function(x: jax.Array) -> jax.Array:
        call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(x.shape, x.dtype))
        return call(x, factor=np.float64(3.0)) + 1.0

    return cx.lower(
        function, (jax.ShapeDtypeStruct((4,), jnp.float64),), custom_calls=[target]
    )


def run_scale(program: cx.LoweredProgram) -> np.ndarray:
    """Compiles and runs a program of `scale_program` on 0, 1, 2 and 3."""
    executable, _ = cx.compile_program(program, disk_cache=False)
    (result,) = executable.run([np.arange(4.0)])
    return result


def test_lowering_names_custom_calls() -> None:
    """Checks that lowering a call to an unknown target says how to allow it."""

    def function(x: jax.Array) -> jax.Array:
        return jax.ffi.ffi_call(
            "unknown_target", jax.ShapeDtypeStruct(x.shape, x.dtype)
        )(x)

    with pytest.raises(ValueError, match="`custom_calls` argument of `cyxla.lower`"):
        cx.lower(function, (jax.ShapeDtypeStruct((4,), jnp.float64),))


def test_registered_target_runs(ffi_library: ctypes.CDLL) -> None:
    """Checks that a handler of another library works once it is registered."""
    name = f"cyxla_test_scale_{next(COUNTER)}"
    handler = ffi_library.scale_handler()
    cx.register_ffi_target(name, handler)
    cx.register_ffi_target(name, handler, platform="Host")
    np.testing.assert_array_equal(run_scale(scale_program(name)), [1.0, 4.0, 7.0, 10.0])
    with pytest.raises(ValueError, match="registered with another handler"):
        cx.register_ffi_target(name, handler + 8)


def test_unregistered_target_explains_registration() -> None:
    """Checks the error for a typed target that nobody registered."""
    program = scale_program(f"cyxla_test_missing_{next(COUNTER)}")
    with pytest.raises(
        RuntimeError, match="No FFI handler registered for cyxla_test_missing"
    ):
        cx.compile_program(program, disk_cache=False)


def test_untyped_targets_are_rejected_by_the_plugin() -> None:
    """Checks that a custom call of the older untyped interface fails to compile."""

    def function(x: jax.Array) -> jax.Array:
        call = jax.ffi.ffi_call(
            "cyxla_test_untyped",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            custom_call_api_version=0,
        )
        return call(x)

    program = cx.lower(
        function,
        (jax.ShapeDtypeStruct((4,), jnp.float64),),
        custom_calls=["cyxla_test_untyped"],
    )
    with pytest.raises(RuntimeError, match="API version"):
        cx.compile_program(program, disk_cache=False)


def test_mirror_is_installed_when_cyxla_is_imported() -> None:
    """Checks that importing cyxla mirrors JAX's registrations and that a second call is a no-op."""
    mirrored = xla_client.register_custom_call_target
    assert _ffi.jax_registration is not None and mirrored is not _ffi.jax_registration
    cx.mirror_jax_registrations()
    assert xla_client.register_custom_call_target is mirrored


def test_mirror_can_be_turned_off() -> None:
    """Checks that the environment variable stops cyxla from replacing the function of JAX."""
    code = """
        from jaxlib import xla_client
        original = xla_client.register_custom_call_target
        import cyxla as cx
        from cyxla import _ffi

        assert xla_client.register_custom_call_target is original
        assert _ffi.jax_registration is None
        """
    assert run_python(code, {cx.MIRROR_ENVIRONMENT_VARIABLE: "0"}).returncode == 0
    mirrored = code.replace("is original", "is not original").replace(
        "is None", "is not None"
    )
    assert run_python(mirrored).returncode == 0


def test_mirror_registers_what_libraries_register(ffi_library: ctypes.CDLL) -> None:
    """Checks that `jax.ffi.register_ffi_target` also reaches the plugin."""
    name = f"cyxla_test_mirror_{next(COUNTER)}"
    capsule = jax.ffi.pycapsule(ctypes.CFUNCTYPE(None)(ffi_library.scale_handler()))
    jax.ffi.register_ffi_target(name, capsule, platform="cpu")
    np.testing.assert_array_equal(run_scale(scale_program(name)), [1.0, 4.0, 7.0, 10.0])
    traits = xla_client.CustomCallTargetTraits.COMMAND_BUFFER_COMPATIBLE
    xla_client.register_custom_call_target(
        f"cyxla_test_traits_{next(COUNTER)}", capsule, "cuda", 1, traits
    )
    assert any(
        target.traits & _ffi.COMMAND_BUFFER_COMPATIBLE
        for target in _ffi.targets.values()
    )


def test_mirror_skips_what_the_plugin_cannot_run(ffi_library: ctypes.CDLL) -> None:
    """Checks that untyped handlers and handlers with several stages are not mirrored."""
    capsule = jax.ffi.pycapsule(ctypes.CFUNCTYPE(None)(ffi_library.scale_handler()))
    untyped = f"cyxla_test_untyped_{next(COUNTER)}"
    xla_client.register_custom_call_target(untyped, capsule, "cpu", 0)
    assert ("cpu", untyped) not in _ffi.targets
    staged = f"cyxla_test_staged_{next(COUNTER)}"
    with pytest.warns(UserWarning, match="runs only the execute stage"):
        jax.ffi.register_ffi_target(
            staged, {"execute": capsule, "prepare": capsule}, platform="cpu"
        )
    assert ("cpu", staged) not in _ffi.targets


def test_handler_forms(ffi_library: ctypes.CDLL) -> None:
    """Checks the forms in which a handler may be given."""
    address = ffi_library.scale_handler()
    capsule = jax.ffi.pycapsule(ctypes.CFUNCTYPE(None)(address))
    assert _ffi.handler_address(address) == address
    assert _ffi.handler_address(capsule) == address
    assert _ffi.handler_address({"execute": capsule, "prepare": None}) == address
    with pytest.raises(ValueError, match="only the execute stage"):
        _ffi.handler_address({"execute": capsule, "initialize": capsule})
    with pytest.raises(ValueError, match="needs an execute stage"):
        _ffi.handler_address({"prepare": capsule})
    with pytest.raises(ValueError, match="null pointer"):
        _ffi.handler_address(0)
    with pytest.raises(TypeError, match="not str"):
        _ffi.handler_address("handler")


def test_targets_wait_for_their_platform() -> None:
    """Checks that a target for another platform is kept and not given to this plugin."""
    name = f"cyxla_test_waiting_{next(COUNTER)}"
    cx.register_ffi_target(name, 4096, platform="gpu", command_buffer_compatible=True)
    assert ("cuda", name) in _ffi.targets
    assert ("cuda", name) not in _ffi.applied
    assert _ffi.platform_key("Host") == "cpu"
    assert _ffi.platform_key("ROCM") == "rocm"
    assert _ffi.apply_targets() == 0


def test_registration_without_a_runtime_or_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that registering is a no-op while no plugin can take handlers."""
    with monkeypatch.context() as patch:
        patch.setattr(_runtime, "is_initialized", lambda: False)
        assert _ffi.apply_targets() == 0
    with monkeypatch.context() as patch:
        patch.setattr(_runtime, "supports_ffi", lambda: False)
        assert cx.register_jax_targets() == 0


def test_jax_targets_skip_missing_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that modules which are missing or have no registrations are skipped."""
    monkeypatch.setattr(
        _ffi, "JAXLIB_MODULES", ("jaxlib.no_such_module", "jax", "jaxlib.lapack")
    )
    names = {name for name, *_ in _ffi.jax_targets("cpu")}
    assert "lapack_dpotrf_ffi" in names
    assert not {name for name, *_ in _ffi.jax_targets("rocm")}


def test_jax_targets_of_another_platform_wait() -> None:
    """Checks that registering the handlers of jaxlib for a GPU platform keeps them waiting."""
    names = {name for name, *_ in _ffi.jax_targets("cuda")}
    assert cx.register_jax_targets("cuda") >= 0
    assert all(("cuda", name) in _ffi.targets for name in names)
    assert not any(("cuda", name) in _ffi.applied for name in names)


def test_lowering_passes_other_errors_on() -> None:
    """Checks that errors of the function itself are not mistaken for custom call errors."""

    def function(x: jax.Array) -> jax.Array:
        raise ValueError("The function is broken.")

    with pytest.raises(ValueError, match="The function is broken."):
        cx.lower(function, (jax.ShapeDtypeStruct((4,), jnp.float64),))


def test_registry_lists_typed_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks what the registry of JAX contributes: live entries, queued ones and traits."""
    live = {name for name, *_ in _ffi.registry_targets("cpu")}
    assert {"lapack_dpotrf_ffi", "lapack_dgetrf_ffi"} <= live
    traits = xla_client.CustomCallTargetTraits.COMMAND_BUFFER_COMPATIBLE
    default = xla_client.CustomCallTargetTraits.DEFAULT
    queued = {
        "CUDA": [
            ("queued_fast", 4096, 1, traits),
            ("queued_slow", 8192, 1, default),
            ("queued_untyped", 12288, 0, default),
        ],
        "Host": [("queued_host", 16384, 1, default)],
    }
    monkeypatch.setattr(xla_client, "_custom_callback", queued)
    assert _ffi.registry_targets("gpu") == [
        ("queued_fast", 4096, True),
        ("queued_slow", 8192, False),
    ]
    assert ("queued_host", 16384, False) in _ffi.registry_targets("cpu")
    assert _ffi.registry_targets("rocm") == []


def test_registry_skips_bare_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that only handlers listed as mappings of stages are taken from the registry."""
    registry = importlib.import_module("jaxlib._jax")
    monkeypatch.setattr(xla_client, "_custom_callback", {})
    monkeypatch.setattr(
        registry,
        "custom_call_targets",
        lambda spelling: {"bare": 4096, "staged": {"execute": 8192}},
    )
    assert _ffi.registry_targets("cpu") == [("staged", {"execute": 8192}, False)]


def test_registry_without_private_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that a jaxlib without the private registry contributes nothing."""
    registry = importlib.import_module("jaxlib._jax")
    monkeypatch.delattr(xla_client, "_custom_callback")
    monkeypatch.delattr(registry, "custom_call_targets")
    assert _ffi.registry_targets("cpu") == []


def test_jax_targets_skip_untyped_registrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that a module of jaxlib lists untyped handlers that cyxla leaves out."""
    module = types.SimpleNamespace(
        registrations=lambda: {
            "cpu": [("fake_typed", 4096, 1), ("fake_untyped", 8192, 0)]
        }
    )
    monkeypatch.setitem(sys.modules, "fake_registrations", module)
    monkeypatch.setattr(_ffi, "JAXLIB_MODULES", ("fake_registrations",))
    names = {name for name, *_ in _ffi.jax_targets("cpu")}
    assert "fake_typed" in names and "fake_untyped" not in names


@pytest.mark.parametrize("backend_started", [False, True])
def test_handlers_registered_before_cyxla_are_found(
    ffi_library_path: Path, backend_started: bool
) -> None:
    """Checks that a handler registered with JAX before cyxla is imported reaches the plugin.

    Before the JAX backend starts the registration is queued, and afterwards it goes to the
    registry of jaxlib, so both are tried.
    """
    result = run_python(
        f"""
        import ctypes

        import jax
        import jax.numpy as jnp
        import numpy as np

        if {backend_started!r}:
            jnp.zeros(1).block_until_ready()
        library = ctypes.CDLL({str(ffi_library_path)!r})
        library.scale_handler.restype = ctypes.c_void_p
        capsule = jax.ffi.pycapsule(ctypes.CFUNCTYPE(None)(library.scale_handler()))
        jax.ffi.register_ffi_target("early_scale", capsule, platform="cpu")

        import cyxla as cx

        def function(x):
            call = jax.ffi.ffi_call("early_scale", jax.ShapeDtypeStruct(x.shape, x.dtype))
            return call(x, factor=np.float64(3.0)) + 1.0

        program = cx.lower(
            function, (jax.ShapeDtypeStruct((4,), jnp.float64),), custom_calls=["early_scale"]
        )
        executable, _ = cx.compile_program(program, disk_cache=False)
        (result,) = executable.run([np.arange(4.0)])
        assert result.tolist() == [1.0, 4.0, 7.0, 10.0], result
        """
    )
    assert result.returncode == 0, result.stderr


def test_handlers_the_plugin_cannot_take_raise_a_warning(
    ffi_library_path: Path,
) -> None:
    """Checks that a handler of JAX with several stages is skipped with a warning at load."""
    result = run_python(
        f"""
        import ctypes
        import warnings

        import jax

        library = ctypes.CDLL({str(ffi_library_path)!r})
        library.scale_handler.restype = ctypes.c_void_p
        capsule = jax.ffi.pycapsule(ctypes.CFUNCTYPE(None)(library.scale_handler()))
        jax.ffi.register_ffi_target(
            "early_staged", {{"execute": capsule, "prepare": capsule}}, platform="cpu"
        )

        import cyxla as cx

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cx.runtime_info()
        messages = [str(warning.message) for warning in caught]
        assert any("early_staged" in message and "execute stage" in message for message in messages), messages
        """
    )
    assert result.returncode == 0, result.stderr
