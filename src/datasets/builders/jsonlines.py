"""Grouped JSON conversation builder"""

import json
import os
import base64
from typing import Any, Dict, List, Literal

from .base import BaseBuilder
from ..geometry import normalize_points
from ..utils import extract_object_points
from ..contracts import ConversationRecord


class JSONLinesBuilder(BaseBuilder):
    """Builder for grouped JSON conversations.

    Produces a single-round chat where the user embeds all images and the assistant
    responds with one JSON object grouped by 图片_N keys.

    Supports two output modes:
    - dense: grouped JSON with geometry + desc (default)
    - summary: grouped JSON with per-image summary strings (loaded from dataset)

    Mode selection is determined per pairing group (at dataset level via system_prompt_summary).
    """

    def __init__(
        self,
        *,
        user_prompt: str,
        emit_norm: Literal["none", "norm100", "norm1000"],
        mode: Literal["dense", "summary"] = "dense",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.user_prompt = user_prompt
        self.emit_norm = emit_norm
        self.mode = mode
        # Group key prefix is fixed by the chat template (图片_N)

    def _get_summary_text(self, record: Dict[str, Any], record_index: int) -> str:
        """Extract and validate summary from record.

        Args:
            record: The data record
            record_index: Index of the record (for error reporting)

        Returns:
            The summary string

        Raises:
            ValueError: if summary is missing or invalid
        """
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(
                f"Missing or invalid 'summary' for record index {record_index}; "
                f"expected non-empty string. Please ensure all records in JSONL have a 'summary' field "
                f"when using summary mode."
            )
        return summary

    def build(
        self, record_a: ConversationRecord, record_b: ConversationRecord
    ) -> Dict[str, Any]:
        """Build grouped JSON messages from two records."""
        records = [record_a, record_b]

        grouped: Dict[str, Dict[str, Any]] = {}
        user_contents: List[Dict[str, Any]] = []
        objects_out: Dict[str, List[Any]] = {"ref": [], "bbox": [], "image_id": []}

        image_slot = 0
        record_index = 0
        for record in records:
            images = record.get("images", []) or []
            objects = record.get("objects", []) or []

            if not images and not objects:
                continue

            image_slot += 1

            for image in images:
                user_contents.append({"type": "image", "image": self._to_url(image)})

            label = f"图片_{image_slot}"

            # Branch on mode: all records in this group use the same mode
            if self.mode == "summary":
                # Summary mode: load and validate summary from record
                grouped[label] = self._get_summary_text(record, record_index)
            else:
                # Dense mode: build full grouped entry with geometry
                grouped[label] = self._build_group_entry(objects, record)
                self._update_objects_metadata(objects_out, objects, image_slot - 1)

            record_index += 1

        user_contents.append({"type": "text", "text": self.user_prompt})

        assistant_text = json.dumps(grouped, ensure_ascii=False)
        messages = [
            {"role": "user", "content": user_contents},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

        merged = {"messages": messages}
        if objects_out["bbox"]:
            merged["objects"] = objects_out

        return merged

    def build_many(self, records: List[ConversationRecord]) -> Dict[str, Any]:
        """Build grouped JSON messages from N records.

        Note: Visible labels (e.g., 图片_N) are injected by the chat template.
        This builder only appends image blocks followed by the user prompt.
        """
        grouped: Dict[str, Dict[str, Any]] = {}
        user_contents: List[Dict[str, Any]] = []
        objects_out: Dict[str, List[Any]] = {"ref": [], "bbox": [], "image_id": []}

        image_slot = 0
        record_index = 0
        for record in records:
            images = record.get("images", []) or []
            objects = record.get("objects", []) or []
            if not images and not objects:
                continue
            image_slot += 1
            for image in images:
                user_contents.append({"type": "image", "image": self._to_url(image)})

            label = f"图片_{image_slot}"

            # Branch on mode: all records in this group use the same mode
            if self.mode == "summary":
                # Summary mode: load and validate summary from record
                grouped[label] = self._get_summary_text(record, record_index)
            else:
                # Dense mode: build full grouped entry with geometry
                grouped[label] = self._build_group_entry(objects, record)
                self._update_objects_metadata(objects_out, objects, image_slot - 1)

            record_index += 1

        user_contents.append({"type": "text", "text": self.user_prompt})

        assistant_text = json.dumps(grouped, ensure_ascii=False)
        messages = [
            {"role": "user", "content": user_contents},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

        merged = {"messages": messages}
        if objects_out["bbox"]:
            merged["objects"] = objects_out
        return merged

    def _build_group_entry(
        self, objects: List[Dict[str, Any]], record: Dict[str, Any]
    ) -> Dict[str, Any]:
        width = float(record.get("width") or 1)
        height = float(record.get("height") or 1)

        grouped_objects: Dict[str, Any] = {}
        for idx, obj in enumerate(objects, start=1):
            geom_type, points = extract_object_points(obj)
            payload: Dict[str, Any] = {"desc": obj.get("desc", "")}
            if geom_type and points:
                # For line objects, emit line_points before line for better causality
                if geom_type == "line":
                    payload["line_points"] = len(points) // 2
                    payload[geom_type] = self._format_points(points, width, height)
                else:
                    payload[geom_type] = self._format_points(points, width, height)
            grouped_objects[f"object_{idx}"] = payload
        return grouped_objects

    def _update_objects_metadata(
        self,
        objects_out: Dict[str, List[Any]],
        objects: List[Dict[str, Any]],
        image_id: int,
    ) -> None:
        for obj in objects:
            geom_type, points = extract_object_points(obj)
            if not geom_type or not points:
                continue
            objects_out["bbox"].append(points)
            objects_out["image_id"].append(image_id)
            desc = obj.get("desc")
            if desc:
                objects_out["ref"].append(desc.split("/")[0])

    def _format_points(
        self, points: List[float], width: float, height: float
    ) -> List[int | float]:
        if self.emit_norm == "none":
            return [float(v) for v in points]
        normalized = normalize_points(points, width, height, self.emit_norm)
        return [int(v) for v in normalized]

    def _to_url(self, image: Any) -> str:
        """Canonicalize an image entry to a URL string for the template.

        - If dict with bytes: produce a data URL (PNG)
        - If relative path: prefix with ROOT_IMAGE_DIR when available
        - If absolute path: pass through
        """
        if isinstance(image, dict) and "bytes" in image:
            b = image["bytes"]
            if not isinstance(b, (bytes, bytearray)):
                raise TypeError("image bytes must be bytes-like")
            b64 = base64.b64encode(b).decode("ascii")
            return f"data:image/png;base64,{b64}"
        if isinstance(image, str):
            if not os.path.isabs(image):
                root = os.environ.get("ROOT_IMAGE_DIR")
                if root:
                    return os.path.join(root, image)
            return image
        # Fallback: stringify
        return str(image)


__all__ = ["JSONLinesBuilder"]
