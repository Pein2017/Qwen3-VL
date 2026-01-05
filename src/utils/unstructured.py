"""Escape-hatch helpers for intentionally unstructured payloads.

These utilities support the Schema Constitution rule that unstructured mappings
are allowed only when documented and validated at entry.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, cast

UnstructuredMapping = Mapping[str, Any]
UnstructuredMutableMapping = MutableMapping[str, Any]
UnstructuredSequence = Sequence[UnstructuredMapping]


def require_mapping(value: object, *, context: str) -> UnstructuredMapping:
    """Validate an intentionally unstructured mapping at entry."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return cast(UnstructuredMapping, value)


def require_mutable_mapping(value: object, *, context: str) -> UnstructuredMutableMapping:
    """Validate an intentionally unstructured mutable mapping at entry."""
    if not isinstance(value, MutableMapping):
        raise TypeError(f"{context} must be a mutable mapping")
    return cast(UnstructuredMutableMapping, value)


def require_mapping_sequence(
    value: object, *, context: str
) -> UnstructuredSequence:
    """Validate a sequence of intentionally unstructured mappings at entry."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a sequence of mappings")
    for idx, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{context}[{idx}] must be a mapping")
    return cast(UnstructuredSequence, value)
