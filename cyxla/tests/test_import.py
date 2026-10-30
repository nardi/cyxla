"""Tests that the runtime loads without importing JAX, which only lowering needs."""

from conftest import run_python


def test_import_and_run_without_jax() -> None:
    """Checks that a process can load the plugin and an executable without JAX."""
    result = run_python(
        """
        import sys

        import cyxla as cx

        cx.initialize(platform="cpu", jax_targets=False)
        assert cx.is_initialized()
        assert "jax" not in sys.modules, "importing cyxla imported jax"
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lowering_imports_jax_when_it_is_used() -> None:
    """Checks that lowering still works in a process that has not imported JAX yet."""
    result = run_python(
        """
        import sys

        import cyxla as cx

        assert "jax" not in sys.modules
        program = cx.lower(lambda x: x + 1, [__import__("jax").ShapeDtypeStruct((2,), "float32")])
        assert program.platform == "cpu"
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
