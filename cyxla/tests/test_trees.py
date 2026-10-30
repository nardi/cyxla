"""Tests of the leaf fingerprints that check trees of arguments."""

from typing import NamedTuple

import numpy as np
import pytest

import cyxla as cx


class Pair(NamedTuple):
    """A tree node with named fields."""

    first: np.ndarray
    second: np.ndarray


def fingerprint(tree: object) -> tuple[cx.LeafInfo, ...]:
    """Returns the fingerprint of a tree of arrays."""
    return cx.tree_fingerprint(tree, lambda leaf: np.shape(leaf))


def test_fingerprint_lists_paths_and_shapes() -> None:
    """Checks the paths of dictionaries, tuples and named tuples."""
    tree = {"a": Pair(np.zeros(2), np.zeros((3, 1))), "b": (np.zeros(()),)}
    assert fingerprint(tree) == (
        cx.LeafInfo("['a'].first", (2,)),
        cx.LeafInfo("['a'].second", (3, 1)),
        cx.LeafInfo("['b'][0]", ()),
    )


def test_is_leaf_stops_flattening() -> None:
    """Checks that a node can be treated as one leaf."""
    tree = {"a": Pair(np.zeros(2), np.zeros(2))}
    result = cx.tree_fingerprint(
        tree, lambda leaf: (2,), is_leaf=lambda node: isinstance(node, Pair)
    )
    assert result == (cx.LeafInfo("['a']", (2,)),)


def test_equal_fingerprints_pass() -> None:
    """Checks that two trees with the same leaves are accepted."""
    first = fingerprint({"a": np.zeros(2)})
    cx.check_fingerprint(first, fingerprint({"a": np.ones(2)}), "values")


def test_differences_name_the_first_leaf() -> None:
    """Checks the messages for another path, shape and number of leaves."""
    expected = fingerprint({"a": np.zeros(2), "b": np.zeros(3)})
    with pytest.raises(ValueError, match=r"leaf \['c'\] at position 0"):
        cx.check_fingerprint(expected, fingerprint({"c": np.zeros(2)}), "values")
    with pytest.raises(ValueError, match=r"shape \(4,\) at \['b'\]"):
        cx.check_fingerprint(
            expected, fingerprint({"a": np.zeros(2), "b": np.zeros(4)}), "values"
        )
    with pytest.raises(ValueError, match="have 2 leaves, but .* for 1 leaves"):
        cx.check_fingerprint(expected[:1], expected, "values")
    with pytest.raises(ValueError, match="have 1 leaves, but .* for 2 leaves"):
        cx.check_fingerprint(expected, expected[:1], "values")


def test_fingerprints_survive_json() -> None:
    """Checks that encoding and decoding a fingerprint changes nothing."""
    original = fingerprint({"a": np.zeros((2, 3))})
    assert cx.decode_fingerprint(cx.encode_fingerprint(original)) == original
