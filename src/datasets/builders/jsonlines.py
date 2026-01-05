"""JSON conversation builder for dense captioning"""

import base64
import copy
import json
import os
from collections.abc import Iterable, Mapping
from typing import Literal, Sequence, cast

from ..contracts import ConversationRecord, DatasetObject, validate_conversation_record
from src.utils.unstructured import UnstructuredMutableMapping
from ..geometry import normalize_points
from ..utils import extract_object_points
from .base import BaseBuilder
from data_conversion.utils.sorting import sort_objects_tlbr


class JSONLinesBuilder(BaseBuilder):
    """Builder for dense caption conversations.

    Produces a single-round chat where the user embeds the image, followed by the
    assistant emitting the minimal object hierarchy (no 图片_N wrapper).

    Modes:
    - ``dense``: assistant returns a JSON object mapping ``object_{n}`` keys to
      geometry/description payloads.
    - ``summary``: assistant returns the summary string stored in the record.
    - Optional ``assistant_prefix`` prepends a single line before the assistant payload.
    """

    def __init__(
        self,
        *,
        user_prompt: str,
        emit_norm: Literal["none", "norm100", "norm1000"],
        mode: Literal["dense", "summary"] = "dense",
        json_format: Literal["standard"] = "standard",
        assistant_prefix: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.user_prompt = user_prompt
        self.emit_norm = emit_norm
        self.mode = mode
        self.json_format = json_format
        self.assistant_prefix = assistant_prefix.strip() if assistant_prefix else None

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

    def build(self, record: ConversationRecord) -> UnstructuredMutableMapping:
        """Build a single-record conversation payload."""
        return self.build_many([record])

    def build_many(
        self, records: Iterable[ConversationRecord]
    ) -> UnstructuredMutableMapping:
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

        # Pass-through for pre-authored chat records (text-only fusion sources).
        messages = record.get("messages")
        if messages:
            payload: UnstructuredMutableMapping = {"messages": copy.deepcopy(messages)}
            metadata = record.get("metadata")
            if metadata is not None:
                payload["metadata"] = copy.deepcopy(metadata)
            return payload

        user_contents: list[dict[str, object]] = []
        objects_out: dict[str, list[object]] = {"ref": [], "bbox": [], "image_id": []}
        objects_payload: UnstructuredMutableMapping = {}

        images = record.get("images", []) or []
        objects_seq: Sequence[DatasetObject] = record.get("objects") or []
        objects = list(objects_seq)
        sorted_objects_raw = sort_objects_tlbr(cast(list[dict[str, object]], objects))
        sorted_objects = cast(list[DatasetObject], sorted_objects_raw)

        for image in images:
            user_contents.append({"type": "image", "image": self._to_url(image)})

        if self.mode == "summary":
            assistant_text = self._get_summary_text(record, 0)
        else:
            assistant_payload = self._build_group_entry(sorted_objects, record)
            assistant_text = self._render_json_text(assistant_payload)
            self._update_objects_metadata(objects_out, sorted_objects, 0)
            objects_payload = assistant_payload

        user_contents.append({"type": "text", "text": self.user_prompt})

        if self.assistant_prefix:
            assistant_text = f"{self.assistant_prefix}\n{assistant_text}"

        messages = [
            {"role": "user", "content": user_contents},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

        merged: UnstructuredMutableMapping = {"messages": messages}
        if objects_payload:
            merged["assistant_payload"] = objects_payload
        if objects_out["bbox"]:
            merged["objects"] = objects_out
        # Preserve metadata from record for fusion dataset grouping
        if "metadata" in record:
            merged["metadata"] = copy.deepcopy(record["metadata"])
        return merged

    def _build_group_entry(
        self, objects: list[DatasetObject], record: ConversationRecord
    ) -> UnstructuredMutableMapping:
        width = float(record.get("width") or 1)
        height = float(record.get("height") or 1)

        grouped_objects: UnstructuredMutableMapping = {}
        for idx, obj in enumerate(objects, start=1):
            geom_type, points = extract_object_points(obj)
            payload: UnstructuredMutableMapping = {
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

    def _sanitize_desc(self, value: object, object_index: int) -> str:
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

        disallowed_patterns = (" ,", ", ", " /", "/ ", "  ")
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
        objects_out: dict[str, list[object]],
        objects: list[DatasetObject],
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
                head = ""
                for token in str(desc).split(","):
                    token = token.strip()
                    if token.startswith("类别="):
                        head = token.split("=", 1)[1]
                        break
                if not head:
                    head = str(desc).split("/", 1)[0]
                objects_out["ref"].append(head)

    def _format_points(
        self, points: list[float], width: float, height: float
    ) -> list[int | float]:
        if self.emit_norm == "none":
            return [float(v) for v in points]
        normalized = normalize_points(points, width, height, self.emit_norm)
        return [int(v) for v in normalized]

    def _render_json_text(self, payload: Mapping[str, object]) -> str:
        text_payload = self._prepare_text_payload(payload)
        indent, separators = self._json_style()
        assistant_text = json.dumps(
            text_payload,
            ensure_ascii=False,
            indent=indent,
            separators=separators,
        )
        return assistant_text

    def _prepare_text_payload(
        self, payload: Mapping[str, object]
    ) -> UnstructuredMutableMapping:
        formatted: UnstructuredMutableMapping = {}
        for key, entry in payload.items():
            if isinstance(entry, Mapping):
                formatted[key] = self._format_object_entry(entry)
            else:
                formatted[key] = entry
        return formatted

    def _format_object_entry(
        self, entry: Mapping[str, object]
    ) -> UnstructuredMutableMapping:
        formatted_entry: UnstructuredMutableMapping = {}
        for field, value in entry.items():
            if field in {"poly", "line"} and isinstance(value, list):
                formatted_entry[field] = self._format_geometry_sequence(value)
            elif field == "bbox_2d" and isinstance(value, list):
                formatted_entry[field] = list(value)
            else:
                formatted_entry[field] = value
        return formatted_entry

    def _format_geometry_sequence(self, values: list[int | float]) -> list[object]:
        if not values:
            return []
        if len(values) % 2 != 0:
            return list(values)
        grouped: list[object] = []
        for idx in range(0, len(values), 2):
            x = values[idx]
            y = values[idx + 1]
            grouped.append([x, y])
        return grouped

    def _json_style(self) -> tuple[int | None, tuple[str, str]]:
        return None, (", ", ": ")

    def _to_url(self, image: object) -> str:
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
