"""Tests of the providers of FFI targets, and of the check that stored programs get."""

import ctypes
import importlib.metadata
import itertools
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import cyxla as cx
from cyxla import _ffi, _records

COUNTER = itertools.count()
"""Numbers for target names, because the plugin keeps a registration for the whole process."""

REGISTRATION_SOURCE = """
def register(cx, name, handler):
    cx.register_ffi_target(name, handler)
"""
"""A function that registers a target, which runs as if it were part of a package."""


def register_as(package: str, name: str, handler: int) -> None:
    """Registers a target from code whose module belongs to `package`."""
    namespace: dict[str, Any] = {"__name__": f"{package}.registry"}
    exec(REGISTRATION_SOURCE, namespace)
    namespace["register"](cx, name, handler)


def stored_record(name: str) -> dict[str, Any]:
    """Lowers and compiles a program that calls a registered target and returns its record."""

    def function(x: jax.Array) -> jax.Array:
        call = jax.ffi.ffi_call(name, jax.ShapeDtypeStruct(x.shape, x.dtype))
        return call(x, factor=np.float64(3.0)) + 1.0

    program = cx.lower(
        function, (jax.ShapeDtypeStruct((4,), jnp.float64),), custom_calls=[name]
    )
    executable, _ = cx.compile_program(program, disk_cache=False)
    return cx.pack_program(cx.BundleWriter("test.program", 1), "scale", executable)


def new_name(prefix: str) -> str:
    """Returns a target name that no earlier test used."""
    return f"cyxla_test_{prefix}_{next(COUNTER)}"


def test_provider_is_the_package_of_the_caller(ffi_library: ctypes.CDLL) -> None:
    """Checks that the package of the registering code is recorded with its version."""
    name = new_name("provider")
    register_as("pytest", name, ffi_library.scale_handler())
    assert cx.target_provider(name) == "pytest"
    assert cx.target_provider(new_name("unknown")) is None
    record = stored_record(name)
    assert record["registered_targets"] == {
        name: {"provider": "pytest", "version": importlib.metadata.version("pytest")}
    }


def test_provider_of_a_library_that_registers_through_jax(
    ffi_library: ctypes.CDLL,
) -> None:
    """Checks that the mirror of JAX's registrations finds the library behind the call."""
    name = new_name("through_jax")
    capsule = jax.ffi.pycapsule(ctypes.CFUNCTYPE(None)(ffi_library.scale_handler()))
    namespace: dict[str, Any] = {"__name__": "pytest.registry"}
    exec(
        "def register(jax, name, capsule):\n"
        '    jax.ffi.register_ffi_target(name, capsule, platform="cpu")\n',
        namespace,
    )
    namespace["register"](jax, name, capsule)
    assert cx.target_provider(name) == "pytest"


def test_registration_from_a_script_has_no_provider(ffi_library: ctypes.CDLL) -> None:
    """Checks that code in `__main__` does not name a package."""
    name = new_name("script")
    register_as("__main__", name, ffi_library.scale_handler())
    assert cx.target_provider(name) is None
    assert stored_record(name)["registered_targets"] == {
        name: {"provider": None, "version": None}
    }


def test_package_without_a_distribution_has_no_version(
    ffi_library: ctypes.CDLL,
) -> None:
    """Checks that a provider that no installed distribution ships has no version."""
    name = new_name("loose")
    register_as("loose_module", name, ffi_library.scale_handler())
    assert stored_record(name)["registered_targets"][name] == {
        "provider": "loose_module",
        "version": None,
    }


def test_registration_without_a_package_in_the_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that a stack made of plumbing only names no provider."""
    packages = set()
    frame = sys._getframe()
    while frame is not None:
        packages.add(str(frame.f_globals.get("__name__", "")).partition(".")[0])
        frame = frame.f_back
    monkeypatch.setattr(_ffi, "PASSTHROUGH_PACKAGES", frozenset(packages))
    assert _ffi.calling_package() is None


def test_targets_that_the_plugin_has_built_in_are_not_recorded() -> None:
    """Checks that only targets in the registry are recorded, and only with a platform."""
    assert _records.registered_targets([new_name("builtin")], "cpu") == {}
    assert _records.registered_targets([new_name("builtin")], None) == {}


def test_records_without_targets_need_none() -> None:
    """Checks that a record from before targets were recorded passes the check."""
    old = {"platform": "cpu", "custom_calls": ["something"]}
    assert cx.missing_targets([old]) == ()
    cx.check_targets([old])


def test_missing_targets_are_reported_once(ffi_library: ctypes.CDLL) -> None:
    """Checks that registered targets pass and that missing ones come back sorted."""
    known = new_name("known")
    register_as("pytest", known, ffi_library.scale_handler())
    absent = new_name("absent")
    other = new_name("other")
    origin = {"provider": "somewhere", "version": "1.2"}
    records = [
        {"platform": "cpu", "registered_targets": {known: origin, other: origin}},
        {"platform": "cpu", "registered_targets": {absent: origin, other: origin}},
    ]
    assert cx.missing_targets(records) == (
        cx.MissingTarget(absent, "somewhere", "1.2"),
        cx.MissingTarget(other, "somewhere", "1.2"),
    )


def test_check_names_each_missing_target() -> None:
    """Checks the three forms of the error, by what is known about the provider."""
    unknown = new_name("unknown")
    versionless = new_name("versionless")
    full = new_name("full")
    record = {
        "platform": "cpu",
        "registered_targets": {
            unknown: {"provider": None, "version": None},
            versionless: {"provider": "loose_module", "version": None},
            full: {"provider": "library", "version": "2.0"},
        },
    }
    with pytest.raises(cx.MissingTargetError) as caught:
        cx.check_targets([record])
    message = str(caught.value)
    assert f"{unknown} (FFI target, the provider is not known)" in message
    assert f"{versionless} (FFI target, provided by loose_module)" in message
    assert f"{full} (FFI target, provided by library 2.0)" in message
    assert "Import the library" in message


def test_check_imports_the_provider_when_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ffi_library: ctypes.CDLL
) -> None:
    """Checks that importing the provider registers its targets, but only on request."""
    name = new_name("late")
    module = f"cyxla_test_late_provider_{next(COUNTER)}"
    (tmp_path / f"{module}.py").write_text(
        "import os\n"
        "import cyxla\n"
        'cyxla.register_ffi_target(os.environ["LATE_TARGET"], int(os.environ["LATE_HANDLER"]))\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("LATE_TARGET", name)
    monkeypatch.setenv("LATE_HANDLER", str(ffi_library.scale_handler()))
    record = {
        "platform": "cpu",
        "registered_targets": {name: {"provider": module, "version": "1.0"}},
    }
    try:
        with pytest.raises(cx.MissingTargetError, match=f"provided by {module} 1.0"):
            cx.check_targets([record])
        assert module not in sys.modules
        cx.check_targets([record], import_providers=True)
        assert cx.target_provider(name) == module
        assert cx.missing_targets([record]) == ()
    finally:
        sys.modules.pop(module, None)


def test_check_reports_a_provider_that_cannot_be_imported() -> None:
    """Checks the error for a provider that is not installed."""
    record = {
        "platform": "cpu",
        "registered_targets": {
            new_name("gone"): {"provider": "cyxla_test_not_installed", "version": "1"}
        },
    }
    with pytest.raises(
        cx.MissingTargetError, match="cyxla_test_not_installed.*cannot be"
    ):
        cx.check_targets([record], import_providers=True)


def test_check_still_fails_when_the_import_registers_nothing() -> None:
    """Checks that importing a provider that does not register the target is not enough."""
    record = {
        "platform": "cpu",
        "registered_targets": {
            new_name("silent"): {"provider": "json", "version": None}
        },
    }
    with pytest.raises(cx.MissingTargetError, match="provided by json"):
        cx.check_targets([record], import_providers=True)
