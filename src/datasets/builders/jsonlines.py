"""JSON conversation builder for dense captioning"""

import base64
import json
import os
from typing import Any, Dict, Iterable, List, Literal, Mapping

from .base import BaseBuilder
from .toon import GEOMETRY_TO_ID, ToonRow, encode_toon_block
from ..geometry import normalize_points
from ..utils import extract_object_points
from ..contracts import ConversationRecord, validate_conversation_record


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
        toon_mode: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.user_prompt = user_prompt
        self.emit_norm = emit_norm
        self.mode = mode
        self.toon_mode = bool(toon_mode)

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
                else json.dumps(assistant_payload, ensure_ascii=False)
            )
        else:
            if self.toon_mode:
                toon_rows = self._build_toon_rows(assistant_payload)
                assistant_text = encode_toon_block(toon_rows)
            else:
                assistant_text = json.dumps(assistant_payload, ensure_ascii=False)

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

    def _build_toon_rows(self, grouped_objects: Dict[str, Any]) -> List[ToonRow]:
        rows: List[ToonRow] = []
        for object_key, payload in grouped_objects.items():
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"TOON mode requires mapping payloads; got {type(payload)!r} for {object_key}"
                )

            desc = str(payload.get("desc", ""))
            geometry_keys = [
                key for key in payload.keys() if key not in {"desc", "line_points"}
            ]

            if not geometry_keys:
                raise ValueError(
                    f"TOON mode cannot encode object without geometry: {object_key}"
                )
            if len(geometry_keys) > 1:
                raise ValueError(
                    f"TOON mode requires a single geometry type per object; "
                    f"found {geometry_keys!r} in {object_key}"
                )

            geometry_key = geometry_keys[0]
            type_id = GEOMETRY_TO_ID.get(geometry_key)
            if type_id is None:
                raise ValueError(
                    f"Unsupported geometry '{geometry_key}' in TOON mode"
                )

            coords_any = payload.get(geometry_key)
            if not isinstance(coords_any, Iterable):
                raise ValueError(
                    f"Geometry coordinates must be iterable for {object_key}"
                )

            coords: List[float | int] = []
            for value in coords_any:  # type: ignore[assignment]
                if isinstance(value, bool):
                    raise ValueError(
                        f"Coordinate values cannot be boolean in TOON mode: {object_key}"
                    )
                if isinstance(value, (int, float)):
                    coords.append(value)
                else:
                    raise ValueError(
                        f"Coordinate values must be numeric in TOON mode; "
                        f"received {type(value)!r} in {object_key}"
                    )

            if type_id == GEOMETRY_TO_ID["bbox_2d"] and len(coords) != 4:
                raise ValueError(
                    "bbox objects must provide exactly 4 coordinates in TOON mode"
                )
            if type_id == GEOMETRY_TO_ID["quad"] and len(coords) != 8:
                raise ValueError(
                    "quad objects must provide exactly 8 coordinates in TOON mode"
                )
            if type_id == GEOMETRY_TO_ID["line"] and len(coords) % 2 != 0:
                raise ValueError(
                    "line objects must provide an even number of coordinates in TOON mode"
                )

            rows.append(ToonRow(type_id=type_id, desc=desc, coords=tuple(coords)))

        return rows

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
