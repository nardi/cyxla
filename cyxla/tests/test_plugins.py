"""Tests of plugin discovery and of loading plugins by path, by record and by default."""

import importlib.metadata
from pathlib import Path

import pytest
from conftest import SampleLibraries, run_python

import cyxla as cx
from cyxla import _plugins
from cyxla._plugins import (
    available_plugins,
    entry_point_plugins,
    find_plugin,
    jax_gpu_plugins,
    namespace_directories,
    nvidia_libraries,
    tpu_plugins,
)

CPU = cx.Plugin(platform="cpu", path=Path("cpu.so"), source="test")
"""A CPU plugin that is never loaded."""

CUDA = cx.Plugin(platform="cuda", path=Path("cuda.so"), source="test")
"""A CUDA plugin that is never loaded."""

TPU = cx.Plugin(platform="tpu", path=Path("tpu.so"), source="test")
"""A TPU plugin that is never loaded."""


class FakeModule:
    """A plugin module as an entry point names it."""

    PLATFORM = "fake"
    CREATE_OPTIONS = {"answer": 42}

    @staticmethod
    def library_path() -> Path:
        """Returns a path that need not exist."""
        return Path("/plugins/libfake.so")


class FakeEntryPoint:
    """An entry point that loads a fixed object, or raises if there is none."""

    def __init__(self, name: str, loaded: object | None) -> None:
        self.name = name
        self.value = f"{name}_module"
        self.dist = None
        self.loaded = loaded

    def load(self) -> object:
        """Returns the loaded object or raises an `ImportError`."""
        if self.loaded is None:
            raise ImportError("broken")
        return self.loaded


def touch(path: Path) -> Path:
    """Creates an empty file and its directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_the_cpu_plugin_is_installed() -> None:
    """Checks that the plugin of `cyxla-pjrt-cpu` is found through its entry point."""
    (plugin,) = [p for p in available_plugins() if p.source == "cyxla-pjrt-cpu"]
    assert plugin.platform == "cpu" and plugin.path.exists()
    assert plugin.create_options == {"asynchronous": False, "cpu_device_count": 1}
    assert find_plugin() == plugin


def test_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that entry points become plugins and that broken ones are skipped."""
    points = [FakeEntryPoint("fake", FakeModule), FakeEntryPoint("broken", None)]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: points)
    with pytest.warns(RuntimeWarning, match="entry point broken: broken"):
        (plugin,) = entry_point_plugins()
    assert plugin == cx.Plugin(
        platform="fake",
        path=Path("/plugins/libfake.so"),
        source="fake_module",
        create_options={"answer": 42},
    )


def test_namespace_directories() -> None:
    """Checks that missing packages and plain modules have no directories."""
    assert namespace_directories("cyxla_no_such_package") == ()
    assert namespace_directories("cyxla._types") == ()
    plugin = find_plugin("cpu")
    assert namespace_directories("cyxla_pjrt_cpu") == (plugin.path.parent,)


def test_jax_gpu_plugins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Checks that JAX's CUDA and ROCm plugins are found with their CUDA libraries."""
    namespace = tmp_path / "jax_plugins"
    cuda = touch(namespace / "xla_cuda12" / "xla_cuda_plugin.so")
    rocm = touch(namespace / "xla_rocm7" / "xla_rocm_plugin.so")
    (namespace / "xla_cuda13").mkdir()
    touch(namespace / "other" / "xla_cuda_plugin.so")
    runtime = touch(tmp_path / "nvidia" / "cuda_runtime" / "lib" / "libcudart.so.12")
    directories = {"jax_plugins": (namespace,), "nvidia": (tmp_path / "nvidia",)}
    monkeypatch.setattr(
        _plugins, "namespace_directories", lambda name: directories.get(name, ())
    )
    assert jax_gpu_plugins() == [
        cx.Plugin(
            platform="rocm",
            path=rocm,
            source="jax_plugins.xla_rocm7",
            create_options={"platform_name": "ROCM", "preallocate": False},
        ),
        cx.Plugin(
            platform="cuda",
            path=cuda,
            source="jax_plugins.xla_cuda12",
            create_options={"platform_name": "cuda", "preallocate": False},
            preload=(runtime,),
        ),
    ]
    assert nvidia_libraries(13) == ()
    assert nvidia_libraries(11) == ()


def test_tpu_plugins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Checks that `libtpu` is found by its library."""
    package = tmp_path / "libtpu"
    package.mkdir()
    monkeypatch.setattr(_plugins, "namespace_directories", lambda name: (package,))
    assert tpu_plugins() == []
    library = touch(package / "libtpu.so")
    assert tpu_plugins() == [cx.Plugin(platform="tpu", path=library, source="libtpu")]


def test_find_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks the choice between installed plugins and the environment variables."""
    plugins: tuple[cx.Plugin, ...] = (TPU, CUDA, CPU)
    monkeypatch.setattr(_plugins, "available_plugins", lambda: plugins)
    assert find_plugin() == CPU
    assert find_plugin("cuda") == CUDA
    monkeypatch.setenv(cx.PLATFORM_ENVIRONMENT_VARIABLE, "tpu")
    assert find_plugin() == TPU
    with pytest.raises(
        RuntimeError, match="platform rocm is installed. Installed: cpu"
    ):
        find_plugin("rocm")
    monkeypatch.delenv(cx.PLATFORM_ENVIRONMENT_VARIABLE)
    plugins = (cx.Plugin(platform="metal", path=Path("m.so"), source="test"),)
    assert find_plugin().platform == "metal"
    plugins = ()
    with pytest.raises(RuntimeError, match="No PJRT plugin is installed"):
        find_plugin()
    with pytest.raises(RuntimeError, match="Installed: none"):
        find_plugin("cpu")
    monkeypatch.setenv(cx.PLUGIN_ENVIRONMENT_VARIABLE, "/plugins/libpjrt.so")
    assert find_plugin() == cx.Plugin(
        platform="cpu",
        path=Path("/plugins/libpjrt.so"),
        source=cx.PLUGIN_ENVIRONMENT_VARIABLE,
        create_options={"asynchronous": False, "cpu_device_count": 1},
    )
    assert find_plugin("tpu").create_options == {}


def test_initialize_by_path_and_record(sample_libraries: SampleLibraries) -> None:
    """Checks loading a plugin by path, by record with preloads, and from the environment."""
    path = str(find_plugin("cpu").path)
    result = run_python(
        f"""
        from pathlib import Path
        import cyxla as cx
        from cyxla import _plugins

        info = cx.initialize(Path({path!r}))
        assert info.platform == "cpu" and info.aliased_inputs
        assert _plugins.loaded_plugin.path == Path({path!r}).resolve()
        """
    )
    assert result.returncode == 0, result.stderr
    library = sample_libraries.preloaded
    result = run_python(
        f"""
        import ctypes
        from pathlib import Path
        import cyxla as cx

        plugin = cx.find_plugin("cpu")._replace(preload=(Path({str(library)!r}),))
        assert cx.initialize(plugin, options={{"cpu_device_count": 2}}).platform == "cpu"
        assert ctypes.CDLL({str(library)!r}).preloaded() == 1
        """
    )
    assert result.returncode == 0, result.stderr
    result = run_python(
        """
        import cyxla as cx

        assert cx.initialize().platform == "cpu"
        """,
        {cx.PLUGIN_ENVIRONMENT_VARIABLE: path},
    )
    assert result.returncode == 0, result.stderr
