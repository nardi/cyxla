"""Tests of the command line diagnostics."""

import pytest

from cyxla import _plugins
from cyxla.__main__ import main


def test_listing_and_loading(capsys: pytest.CaptureFixture[str]) -> None:
    """Checks that the installed plugin is listed and that loading prints the runtime."""
    assert main(["--load"]) == 0
    output = capsys.readouterr().out
    assert "cpu\tcyxla-pjrt-cpu\t" in output
    assert "Loaded cpu, PJRT C API 0." in output
    assert main(["--load", "cpu"]) == 0


def test_no_plugins(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks the message without installed plugins."""
    monkeypatch.setattr(_plugins, "available_plugins", lambda: ())
    assert main([]) == 0
    assert capsys.readouterr().out == "No PJRT plugins are installed.\n"
