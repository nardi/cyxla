"""JSON encoding of the option records of a library.

Option records are nested `NamedTuple`s of numbers, strings, booleans, enums and tuples. The
encoder turns one into plain JSON values, and the decoder rebuilds it from the annotations
of its class, so a record that gains a field with a default still reads files of older
versions. A field may be a union of several types, such as a tuple of names or the literal
`"detect"`.
"""

import enum
import types
import typing
from typing import Any, get_args, get_origin, get_type_hints


def encode_record(value: Any) -> Any:
    """Returns a record as JSON-compatible values.

    The value is typed `Any` because it may be any of the supported records.
    """
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {name: encode_record(item) for name, item in value._asdict().items()}
    if isinstance(value, tuple):
        return [encode_record(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"Cannot encode a value of type {type(value).__name__}.")


def accepts(annotation: Any, data: Any) -> bool:
    """Returns whether JSON values have the form that `encode_record` gives an annotation.

    A union of several types needs this to pick the member to decode with. The annotation
    and the data are typed `Any` because they range over every supported type.
    """
    origin = get_origin(annotation)
    if origin is typing.Literal:
        return data in get_args(annotation)
    if origin is tuple:
        return isinstance(data, list)
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return data in [member.value for member in annotation]
    if isinstance(annotation, type) and issubclass(annotation, tuple):
        return isinstance(data, dict)
    if annotation is float:
        return isinstance(data, int | float) and not isinstance(data, bool)
    return isinstance(annotation, type) and isinstance(data, annotation)


def decode_record(annotation: Any, data: Any) -> Any:
    """Returns the value of an annotated type from the JSON values of `encode_record`.

    Fields that are missing in `data` get their defaults. The annotation and the result are
    typed `Any` because they range over every supported type.
    """
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        if data is None:
            return None
        members = [item for item in arguments if item is not type(None)]
        if len(members) == 1:
            return decode_record(members[0], data)
        for member in members:
            if accepts(member, data):
                return decode_record(member, data)
        raise ValueError(f"No member of {annotation} can hold the value {data!r}.")
    if origin is tuple:
        item = arguments[0]
        return tuple(decode_record(item, entry) for entry in data)
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return annotation(data)
    if isinstance(annotation, type) and issubclass(annotation, tuple):
        hints = get_type_hints(annotation)
        fields = {
            name: decode_record(hints[name], data[name])
            for name in annotation._fields
            if name in data
        }
        return annotation(**fields)
    if annotation is float and isinstance(data, int):
        return float(data)
    return data
