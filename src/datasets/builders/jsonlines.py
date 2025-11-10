"""JSON conversation builder for dense captioning"""

import base64
import json
import os
import re
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple

from ..contracts import ConversationRecord, validate_conversation_record
from ..geometry import normalize_points
from ..utils import extract_object_points
from .base import BaseBuilder


_POINT_PAIR_BLOCK_RE = re.compile(
    r"(\n\s*)\[\s*\n\s*([-0-9.eE]+)\s*,\s*\n\s*([-0-9.eE]+)\s*\n\s*\]"
)
_XY_OBJECT_BLOCK_RE = re.compile(
    r"(\n\s*)\{\s*\n\s*\"x\":\s*([-0-9.eE]+)\s*,\s*\n\s*\"y\":\s*([-0-9.eE]+)\s*\n\s*\}"
)


class JSONLinesBuilder(BaseBuilder):
    """Builder for dense caption conversations.

    Produces a single-round chat where the user embeds the image, followed by the
    assistant emitting the minimal object hierarchy (no 图片_N wrapper).

    Modes:
    - ``dense``: assistant returns a JSON object mapping ``object_{n}`` keys to
      geometry/description payloads.
    - ``summary``: assistant returns the summary string stored in the record.
    """

    def __init__(
        self,
        *,
        user_prompt: str,
        emit_norm: Literal["none", "norm100", "norm1000"],
        mode: Literal["dense", "summary"] = "dense",
        json_format: Literal["type_a", "type_b", "type_c", "type_d"] = "type_b",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.user_prompt = user_prompt
        self.emit_norm = emit_norm
        self.mode = mode
        self.json_format = json_format

    def _get_summary_text(self, record: ConversationRecord, record_index: int) -> str:
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

    def build(self, record: ConversationRecord) -> Dict[str, Any]:
        """Build a single-record conversation payload."""
        return self.build_many([record])

    def build_many(self, records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        """Build conversation messages from one record.

        Dynamic pairing is no longer supported; this method fails if more than one
        record is provided to highlight legacy call paths.
        """

        records_list = list(records)
        if len(records_list) != 1:
            raise ValueError(
                "Dynamic pairing is no longer supported; JSONLinesBuilder expects exactly one record."
            )

        record = validate_conversation_record(records_list[0])

        user_contents: List[Dict[str, Any]] = []
        objects_out: Dict[str, List[Any]] = {"ref": [], "bbox": [], "image_id": []}
        objects_payload: Dict[str, Any] = {}

        images = record.get("images", []) or []
        objects = record.get("objects", []) or []

        for image in images:
            user_contents.append({"type": "image", "image": self._to_url(image)})

        if self.mode == "summary":
            assistant_payload: Any = self._get_summary_text(record, 0)
        else:
            assistant_payload = self._build_group_entry(objects, record)
            self._update_objects_metadata(objects_out, objects, 0)
            objects_payload = assistant_payload

        user_contents.append({"type": "text", "text": self.user_prompt})

        if self.mode == "summary":
            assistant_text = (
                assistant_payload
                if isinstance(assistant_payload, str)
                else self._render_json_text(assistant_payload)
            )
        else:
            assistant_text = self._render_json_text(assistant_payload)

        messages = [
            {"role": "user", "content": user_contents},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

        merged: Dict[str, Any] = {"messages": messages}
        if objects_payload:
            merged["assistant_payload"] = objects_payload
        if objects_out["bbox"]:
            merged["objects"] = objects_out
        return merged

    def _build_group_entry(
        self, objects: List[Dict[str, Any]], record: ConversationRecord
    ) -> Dict[str, Any]:
        width = float(record.get("width") or 1)
        height = float(record.get("height") or 1)

        grouped_objects: Dict[str, Any] = {}
        for idx, obj in enumerate(objects, start=1):
            geom_type, points = extract_object_points(obj)
            payload: Dict[str, Any] = {
                "desc": self._sanitize_desc(obj.get("desc"), idx)
            }
            if geom_type and points:
                # For line objects, emit line_points before line for better causality
                if geom_type == "line":
                    payload["line_points"] = len(points) // 2
                    payload[geom_type] = self._format_points(points, width, height)
                else:
                    payload[geom_type] = self._format_points(points, width, height)
            grouped_objects[f"object_{idx}"] = payload
        return grouped_objects

    def _sanitize_desc(self, value: Any, object_index: int) -> str:
        if not isinstance(value, str):
            raise ValueError(
                f"Object object_{object_index} must provide a string 'desc'; got {type(value)!r}"
            )

        desc = value.strip()
        if not desc:
            raise ValueError(
                f"Object object_{object_index} has empty 'desc' after stripping whitespace"
            )

        if any(char in desc for char in "\n\r\t"):
            raise ValueError(
                f"Object object_{object_index} desc contains forbidden control whitespace"
            )

        disallowed_patterns = (" ,", ", ", " /", "/ ", " :", ": ", "  ")
        for pattern in disallowed_patterns:
            if pattern in desc:
                raise ValueError(
                    "Object object_{} desc contains disallowed whitespace pattern '{}'".format(
                        object_index, pattern
                    )
                )

        return desc

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

    def _render_json_text(self, payload: Mapping[str, Any]) -> str:
        text_payload = self._prepare_text_payload(payload)
        indent, separators = self._json_style()
        assistant_text = json.dumps(
            text_payload,
            ensure_ascii=False,
            indent=indent,
            separators=separators,
        )
        if self.json_format == "type_c":
            assistant_text = self._format_type_c_pairs(assistant_text)
        elif self.json_format == "type_d":
            assistant_text = self._format_type_d_points(assistant_text)
        return assistant_text

    def _prepare_text_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        formatted: Dict[str, Any] = {}
        for key, entry in payload.items():
            if isinstance(entry, Mapping):
                formatted[key] = self._format_object_entry(entry)
            else:
                formatted[key] = entry
        return formatted

    def _format_object_entry(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        formatted_entry: Dict[str, Any] = {}
        for field, value in entry.items():
            if field in {"quad", "line"} and isinstance(value, list):
                formatted_entry[field] = self._format_geometry_sequence(value)
            elif field == "bbox_2d" and isinstance(value, list):
                formatted_entry[field] = list(value)
            else:
                formatted_entry[field] = value
        return formatted_entry

    def _format_geometry_sequence(self, values: List[int | float]) -> List[Any]:
        if not values:
            return []
        if len(values) % 2 != 0:
            return list(values)
        emit_objects = self.json_format == "type_d"
        grouped: List[Any] = []
        for idx in range(0, len(values), 2):
            x = values[idx]
            y = values[idx + 1]
            if emit_objects:
                grouped.append({"x": x, "y": y})
            else:
                grouped.append([x, y])
        return grouped

    def _json_style(self) -> Tuple[Optional[int], Tuple[str, str]]:
        if self.json_format == "type_a":
            return None, (",", ":")
        if self.json_format == "type_b":
            return None, (", ", ": ")
        # Type C and D share the same indentation defaults
        return 2, (", ", ": ")

    def _format_type_c_pairs(self, text: str) -> str:
        def _repl(match: re.Match[str]) -> str:
            indent, x, y = match.groups()
            return f"{indent}[{x}, {y}]"

        return _POINT_PAIR_BLOCK_RE.sub(_repl, text)

    def _format_type_d_points(self, text: str) -> str:
        def _repl(match: re.Match[str]) -> str:
            indent, x, y = match.groups()
            return f'{indent}{{ "x": {x}, "y": {y} }}'

        return _XY_OBJECT_BLOCK_RE.sub(_repl, text)

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
