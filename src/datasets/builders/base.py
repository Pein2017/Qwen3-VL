"""Base builder interface"""

from typing import Any, Iterable

from ..contracts import ConversationRecord, validate_conversation_record
from src.utils.unstructured import UnstructuredMutableMapping


class BaseBuilder:
    """Base class for message builders.

    Builders construct messages from a single record for training.
    """

    def __init__(self, **kwargs: Any):
        """Initialize builder with configuration."""
        self.config = kwargs

    def build(self, record: ConversationRecord) -> UnstructuredMutableMapping:
        """Build messages from a single record (unstructured payload)."""
        raise NotImplementedError(
            "Subclasses must implement build() for single records"
        )

    def build_many(
        self, records: Iterable[ConversationRecord]
    ) -> UnstructuredMutableMapping:
        """Build messages from one record.

        Dynamic pairing/grouping has been removed; providing more than one record
        is treated as an error to fail fast on legacy call sites.
        """

        items = [validate_conversation_record(r) for r in records]
        if not items:
            raise ValueError("build_many requires at least one record")
        if len(items) == 1:
            return self.build(items[0])
        raise ValueError(
            "Dynamic pairing is no longer supported; provide exactly one record to build_many()."
        )

    def __call__(self, record: ConversationRecord) -> UnstructuredMutableMapping:
        """Allow builder to be called as a function with a single record."""

        rec = validate_conversation_record(record)
        return self.build(rec)


__all__ = ["BaseBuilder"]
