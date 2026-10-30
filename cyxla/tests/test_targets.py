"""Tests of the fingerprint of a target and of the scan for custom calls."""

import sys

import cyxla as cx


def test_fingerprint_describes_the_plugin_and_machine() -> None:
    """Checks the fields of the fingerprint of the loaded CPU plugin."""
    fingerprint = cx.target_fingerprint()
    assert fingerprint.platform == "cpu"
    assert fingerprint.system == sys.platform
    assert fingerprint.plugin_size > 0
    assert fingerprint.plugin.startswith("cyxla-pjrt-cpu")
    assert fingerprint.api_version.count(".") == 1
    assert fingerprint.key.count("|") == 5
    assert fingerprint == cx.target_fingerprint()


def test_fingerprint_names_a_plugin_without_a_distribution(monkeypatch) -> None:  # noqa: ANN001
    """Checks that the file name stands in when the plugin has no version."""
    from cyxla import _plugins

    identity = _plugins.loaded_plugin
    assert identity is not None
    monkeypatch.setattr(_plugins, "loaded_plugin", identity._replace(version=""))
    assert cx.target_fingerprint().plugin == identity.path.name


def test_custom_calls_are_found_in_text() -> None:
    """Checks the scan for the targets of custom calls."""
    text = """
    %0 = stablehlo.custom_call @lapack_dgetrf(%arg0) : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %1 = stablehlo.custom_call @lapack_dgetrf(%0) : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %2 = "stablehlo.custom_call"(%1) {call_target_name = "x"} : (tensor<f64>) -> tensor<f64>
    %3 = stablehlo.custom_call @other.call(%1) : (tensor<f64>) -> tensor<f64>
    """
    assert cx.custom_call_targets(text) == ("lapack_dgetrf", "other.call")
    assert cx.custom_call_targets("no calls") == ()
