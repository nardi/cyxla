"""Tests of the JSON encoding of option records."""

import enum
import json
import math
from typing import Literal, NamedTuple

import pytest

import cyxla as cx


class Mode(enum.Enum):
    """A choice between two ways."""

    FAST = "fast"
    EXACT = "exact"


class Inner(NamedTuple):
    """A nested record."""

    flag: bool = True
    ratio: float = 0.5


class Options(NamedTuple):
    """A record with every supported kind of field."""

    mode: Mode = Mode.FAST
    count: int = 3
    name: str = "x"
    limit: float | None = None
    inner: Inner = Inner()
    sizes: tuple[int, ...] = (1, 2)


def test_round_trip() -> None:
    """Checks that a record with every kind of field survives JSON."""
    options = Options(Mode.EXACT, 7, "y", 2.5, Inner(False, 0.25), (4, 5, 6))
    data = json.loads(json.dumps(cx.encode_record(options)))
    assert cx.decode_record(Options, data) == options


def test_infinite_numbers_and_defaults() -> None:
    """Checks infinity, a float written as an integer and fields missing from old files."""
    data = json.loads(json.dumps(cx.encode_record(Options(limit=math.inf))))
    assert cx.decode_record(Options, data).limit == math.inf
    assert cx.decode_record(Options, {"inner": {"ratio": 1}}).inner == Inner(True, 1.0)
    assert cx.decode_record(Options, {}) == Options()
    assert cx.decode_record(Options, {"limit": None}).limit is None


def test_unsupported_values_are_rejected() -> None:
    """Checks the error for a value that has no JSON form."""
    with pytest.raises(TypeError, match="Cannot encode a value of type object"):
        cx.encode_record(object())


class Choices(NamedTuple):
    """A record with fields that are unions of several types."""

    names: tuple[str, ...] | Literal["detect"] = ()
    value: Mode | Inner | float | str = 1.5


def test_unions_of_several_types() -> None:
    """Checks that the member of a union is chosen by the form of the data."""
    for options in (
        Choices(("a", "b")),
        Choices("detect"),
        Choices(value=Mode.EXACT),
        Choices(value=Inner(False, 0.25)),
        Choices(value=2.0),
        Choices(value="free text"),
    ):
        data = json.loads(json.dumps(cx.encode_record(options)))
        assert cx.decode_record(Choices, data) == options
    assert cx.decode_record(Choices, {"value": 3}).value == 3.0


def test_union_without_a_matching_member() -> None:
    """Checks the error for data that none of the members of a union can hold."""
    with pytest.raises(ValueError, match="No member of"):
        cx.decode_record(Choices, {"value": [1]})
    with pytest.raises(ValueError, match="No member of"):
        cx.decode_record(Choices, {"names": True})
