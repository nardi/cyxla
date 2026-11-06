"""Tests of the rules for the arguments and results of an executable."""

import numpy as np
import pytest

from cyxla import _arguments
from cyxla._types import ArrayType

FLOAT32_VECTOR = ArrayType(np.dtype(np.float32), (3,))
"""The type of an argument that is a vector of three single precision numbers."""

SUPPORTED = {np.dtype(np.float32), np.dtype(np.float64)}
"""The element types that these tests treat as supported by the runtime."""


def test_argument_converts_to_the_type_of_the_program() -> None:
    """Checks that float64 values pass for a float32 argument and come back flat."""
    array = _arguments.as_input(np.ones((3, 1)), FLOAT32_VECTOR, 0, SUPPORTED)
    assert array.dtype == np.float32
    assert array.shape == (3,)


def test_argument_without_a_known_type_keeps_its_dtype() -> None:
    """Checks that an unknown program type accepts the supported dtypes only."""
    assert _arguments.as_input([1.0, 2.0], None, 0, SUPPORTED).dtype == np.float64
    with pytest.raises(TypeError, match="Argument 1 has the dtype int8"):
        _arguments.as_input(np.zeros(2, np.int8), None, 1, SUPPORTED)


def test_lossy_conversion_and_wrong_size_are_refused() -> None:
    """Checks the errors for a dtype that loses information and for a wrong size."""
    integer = ArrayType(np.dtype(np.int32), (2,))
    with pytest.raises(TypeError, match="cannot convert"):
        _arguments.as_input(np.ones(2), integer, 0, SUPPORTED)
    with pytest.raises(ValueError, match="has 2 elements, but the program takes 3"):
        _arguments.as_input(np.ones(2), FLOAT32_VECTOR, 0, SUPPORTED)


def test_counts_are_checked_against_the_limits_and_the_types() -> None:
    """Checks the errors for too many arguments or outputs and for a wrong count."""
    _arguments.check_counts(1, 1, [FLOAT32_VECTOR], 8, 8)
    with pytest.raises(ValueError, match="at most 2 arguments and returns at most 1"):
        _arguments.check_counts(3, 1, None, 2, 1)
    with pytest.raises(ValueError, match="takes 1 arguments, not 2"):
        _arguments.check_counts(2, 1, [FLOAT32_VECTOR], 8, 8)


def test_results_and_zero_arguments_follow_the_types() -> None:
    """Checks the shapes of the results and of the zeros that warm up an executable."""
    (result,) = _arguments.empty_results([FLOAT32_VECTOR])
    assert result.shape == (3,) and result.dtype == np.float32
    (zeros,) = _arguments.zero_arguments([FLOAT32_VECTOR])
    assert not zeros.any()
    with pytest.raises(ValueError, match="not known"):
        _arguments.zero_arguments(None)


def test_prepared_arguments_apply_the_expected_types_in_order() -> None:
    """Checks that each argument meets the type at its own position."""
    prepared = _arguments.prepared_arguments(
        [np.ones(3), np.ones(3)], [FLOAT32_VECTOR, FLOAT32_VECTOR], SUPPORTED
    )
    assert [item.dtype for item in prepared] == [np.float32, np.float32]
