"""Tests of the commands that `build_plugin.py` runs, checked with dry runs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import backend  # noqa: E402
import build_plugin  # noqa: E402


@pytest.mark.parametrize(
    ("platform_name", "configuration", "library", "startup"),
    [
        ("linux-x86_64", "--config=ci_linux_x86_64", "libpjrt_cpu.so", False),
        ("macos-arm64", "--macos_minimum_os=11.0", "libpjrt_cpu.dylib", False),
        ("windows-amd64", "--config=clang_local", "pjrt_cpu.dll", True),
    ],
)
def test_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    platform_name: str,
    configuration: str,
    library: str,
    startup: bool,
) -> None:
    """Checks that each platform clones JAX, writes jaxlib's options and builds its target."""
    monkeypatch.setattr(backend, "host_platform", lambda: platform_name)
    arguments = ["--output", str(tmp_path), "--source", str(tmp_path / "jax")]
    assert build_plugin.main([*arguments, "--disk-cache", "cache", "--dry-run"]) == 0
    output = capsys.readouterr().out
    settings = backend.read_toml("plugin.toml")
    assert f"git clone --depth 1 --branch {settings['jax_tag']}" in output
    assert "build/build.py build --wheels=jaxlib --configure_only" in output
    assert f"--bazel_options={configuration}" in output
    assert "--bazel_options=--config=public_cache" in output
    assert "--bazel_options=--disk_cache=" in output
    assert f"build //cyxla_pjrt_cpu:{library}" in output
    assert ("--output_user_root=C:/b" in output) == startup


def test_bazel_package_has_every_library() -> None:
    """Checks that every platform's library is a target of the Bazel package."""
    package = (Path(build_plugin.__file__).parent / "bazel" / "BUILD.bazel").read_text()
    for entry in backend.read_toml("plugin.toml")["platforms"].values():
        assert f'name = "{entry["library"]}"' in package
