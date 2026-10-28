"""Runs the code of the GPU notebook on the CPU.

`notebooks/colab_gpu_tests.ipynb` can only run for real on a CUDA machine, so a mistake in
its code, like a stale name after a refactor, would otherwise only show up on the next
manual Colab run. These tests check that every code cell parses, and they run the checks
script of the notebook with the CPU as the expected platform. That covers everything the
checks assert about `cyxla` except what is particular to the CUDA stack.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import ENVIRONMENT_VARIABLES, without_coverage

import cyxla as cx

NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "colab_gpu_tests.ipynb"
"""The notebook that Colab opens."""

CHECKS_FILE = "gpu_checks.py"
"""The file that the notebook writes its checks to."""


def code_cells() -> list[str]:
    """Returns the source of each code cell of the notebook."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]


def written_script(filename: str) -> str:
    """Returns the body of the cell that starts with `%%writefile <filename>`."""
    magic = f"%%writefile {filename}\n"
    (script,) = [cell for cell in code_cells() if cell.startswith(magic)]
    return script.removeprefix(magic)


@pytest.mark.parametrize("cell", code_cells())
def test_notebook_cells_are_valid_python(cell: str) -> None:
    """Checks that every code cell parses, once the magics of Colab are set aside."""
    lines = [line for line in cell.splitlines() if not line.startswith("%")]
    ast.parse("\n".join(lines))


def test_checks_pass_on_the_cpu(tmp_path: Path) -> None:
    """Checks that the script of the notebook passes when the expected platform is the CPU."""
    script = tmp_path / CHECKS_FILE
    script.write_text(written_script(CHECKS_FILE), encoding="utf-8")
    environment = {
        key: value
        for key, value in without_coverage().items()
        if key not in (*ENVIRONMENT_VARIABLES, cx.CACHE_ENVIRONMENT_VARIABLE)
    }
    result = subprocess.run(
        [sys.executable, str(script), "cpu"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "8 of 8 checks passed on cpu." in result.stdout


def test_checks_fail_on_the_wrong_platform(tmp_path: Path) -> None:
    """Checks that the script reports failures with a non-zero exit status."""
    script = tmp_path / CHECKS_FILE
    script.write_text(written_script(CHECKS_FILE), encoding="utf-8")
    environment = {
        key: value
        for key, value in without_coverage().items()
        if key not in (*ENVIRONMENT_VARIABLES, cx.CACHE_ENVIRONMENT_VARIABLE)
    }
    result = subprocess.run(
        [sys.executable, str(script), "cuda"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode != 0
    assert "Failed: check_lazy_initialization" in result.stderr
