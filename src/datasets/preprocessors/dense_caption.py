"""Dense caption preprocessing"""

from typing import Any, Dict, Optional

from .base import BasePreprocessor


class DenseCaptionPreprocessor(BasePreprocessor):
    """Preprocessor for dense caption dataset records.

    Validates and normalizes record structure for dense captioning tasks.
    """

    def __init__(
        self, *, require_objects: bool = False, require_summary: bool = False, **kwargs
    ):
        """Initialize preprocessor.

        Args:
            require_objects: If True, skip records without objects
            require_summary: If True, skip records without summary
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.require_objects = require_objects
        self.require_summary = require_summary

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a dense caption record.

        Args:
            row: Input record with images, objects, width, height, etc.

        Returns:
            Processed record or None if validation fails
        """
        # Validate required fields
        if not row.get("images"):
            return None

        if self.require_objects and not row.get("objects"):
            return None

        if self.require_summary and not row.get("summary"):
            return None

        # Ensure objects is a list
        if "objects" in row and not isinstance(row["objects"], list):
            row["objects"] = []

        # Ensure width/height exist
        if "width" not in row or "height" not in row:
            return None

        return row


__all__ = ["DenseCaptionPreprocessor"]
