"""Checks that a tree of arguments has the leaves that a program was built for.

A serialized program works on flat vectors. The caller gives the trees that the vectors are
cut from, so the file only keeps a fingerprint of each tree, the path and shape of every
leaf in order. Checking it catches a tree that has other leaves or the same leaves in another
order before it can scramble values.
"""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple


class LeafInfo(NamedTuple):
    """The path and shape of one leaf of a tree."""

    path: str
    """The path from the root, as `jax.tree_util.keystr` writes it."""
    shape: tuple[int, ...]


def tree_fingerprint(
    tree: Any,
    shape_of: Callable[[Any], tuple[int, ...]],
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[LeafInfo, ...]:
    """Returns the path and shape of every leaf of a tree, in the order JAX flattens it.

    `shape_of` gives the shape of a leaf and `is_leaf` stops the flattening early, as the
    argument of `jax.tree.flatten` does. The tree is typed `Any` because it may be any
    pytree.
    """
    import jax

    flat, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_leaf)
    return tuple(
        LeafInfo(jax.tree_util.keystr(path), tuple(shape_of(leaf)))
        for path, leaf in flat
    )


def check_fingerprint(
    expected: Sequence[LeafInfo], actual: Sequence[LeafInfo], description: str
) -> None:
    """Raises a `ValueError` that names the first difference between two fingerprints."""
    for position, (want, got) in enumerate(zip(expected, actual, strict=False)):
        if want.path != got.path:
            raise ValueError(
                f"The {description} have the leaf {got.path} at position {position}, but "
                f"the problem was built for {want.path}."
            )
        if want.shape != got.shape:
            raise ValueError(
                f"The {description} have the shape {got.shape} at {got.path}, but the "
                f"problem was built for {want.shape}."
            )
    if len(expected) != len(actual):
        raise ValueError(
            f"The {description} have {len(actual)} leaves, but the problem was built for "
            f"{len(expected)} leaves."
        )


def encode_fingerprint(fingerprint: Sequence[LeafInfo]) -> list[list[Any]]:
    """Returns a fingerprint as JSON-compatible lists."""
    return [[leaf.path, list(leaf.shape)] for leaf in fingerprint]


def decode_fingerprint(data: Sequence[Sequence[Any]]) -> tuple[LeafInfo, ...]:
    """Returns a fingerprint from the lists that `encode_fingerprint` made."""
    return tuple(LeafInfo(path, tuple(shape)) for path, shape in data)
