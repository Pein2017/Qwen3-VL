"""TOON encoding/decoding helpers for dense captioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

FIELD_NAMES: Tuple[str, str, str] = ("type", "desc", "xs")
HEADER_KEY = "objs"
DEFAULT_DELIMITER = ","
INDENT = "  "

GEOMETRY_TO_ID = {"bbox_2d": 0, "quad": 1, "line": 2}
GEOMETRY_FROM_ID = {value: key for key, value in GEOMETRY_TO_ID.items()}


class ToonFormatError(ValueError):
    """Raised when TOON payloads cannot be encoded or decoded."""


@dataclass(frozen=True)
class ToonRow:
    """Structured row carrying geometry information."""

    type_id: int
    desc: str
    coords: Tuple[float | int, ...]


def encode_toon_block(
    rows: Sequence[ToonRow], *, delimiter: str = DEFAULT_DELIMITER
) -> str:
    """Encode rows into a TOON block."""

    _validate_delimiter(delimiter)
    header = _format_header(len(rows), delimiter)
    if not rows:
        return header

    encoded_rows = [header]
    for row in rows:
        encoded_rows.append(_encode_row(row, delimiter))
    return "\n".join(encoded_rows)


def decode_toon_block(block: str) -> List[ToonRow]:
    """Parse a TOON block into structured rows."""

    if block is None:
        raise ToonFormatError("TOON block must be a string, got None")

    lines = [line for line in (block.splitlines()) if line.strip()]
    if not lines:
        raise ToonFormatError("TOON block is empty")

    header_line = lines[0].strip()
    header_info = _parse_header(header_line)
    delimiter = header_info["delimiter"]
    expected_length = header_info["length"]

    data_lines = lines[1:]
    if expected_length != len(data_lines):
        raise ToonFormatError(
            "TOON block row count mismatch: expected "
            f"{expected_length}, found {len(data_lines)}"
        )

    rows: List[ToonRow] = []
    for line in data_lines:
        rows.append(_decode_row(line, delimiter))
    return rows


def decode_toon_payload(block: str) -> dict[str, dict[str, object]]:
    """Decode a TOON block into the canonical dense-caption payload."""

    rows = decode_toon_block(block)
    return toon_rows_to_payload(rows)


def toon_rows_to_payload(rows: Iterable[ToonRow]) -> dict[str, dict[str, object]]:
    """Convert rows into the canonical `object_{n}` mapping used in JSON mode."""

    payload: dict[str, dict[str, object]] = {}
    for idx, row in enumerate(rows, start=1):
        geometry = GEOMETRY_FROM_ID.get(row.type_id)
        if geometry is None:
            raise ToonFormatError(f"Unsupported geometry type id: {row.type_id}")

        coords = list(row.coords)
        if geometry == "bbox_2d" and len(coords) != 4:
            raise ToonFormatError(
                "bbox rows must contain exactly 4 coordinates; "
                f"received {len(coords)}"
            )
        if geometry == "quad" and len(coords) != 8:
            raise ToonFormatError(
                "quad rows must contain exactly 8 coordinates; "
                f"received {len(coords)}"
            )
        if geometry == "line":
            if len(coords) % 2 != 0:
                raise ToonFormatError(
                    "line rows must have an even number of coordinates"
                )

        entry: dict[str, object] = {"desc": row.desc, geometry: coords}
        if geometry == "line":
            entry["line_points"] = len(coords) // 2

        payload[f"object_{idx}"] = entry

    return payload


def _encode_row(row: ToonRow, delimiter: str) -> str:
    geometry = GEOMETRY_FROM_ID.get(row.type_id)
    if geometry is None:
        raise ToonFormatError(f"Unsupported geometry type id: {row.type_id}")

    coords = _coerce_coords(row.coords)
    if geometry == "bbox_2d" and len(coords) != 4:
        raise ToonFormatError(
            "bbox objects must emit exactly 4 coordinates; "
            f"received {len(coords)}"
        )
    if geometry == "quad" and len(coords) != 8:
        raise ToonFormatError(
            "quad objects must emit exactly 8 coordinates; "
            f"received {len(coords)}"
        )
    if geometry == "line":
        if len(coords) % 2 != 0:
            raise ToonFormatError(
                "line objects must emit an even number of coordinates"
            )

    desc_token = _encode_desc(row.desc, delimiter)
    coord_tokens = [_format_numeric(value) for value in coords]
    joined = delimiter.join([str(int(row.type_id)), desc_token, *coord_tokens])
    return f"{INDENT}{joined}"


def _decode_row(line: str, delimiter: str) -> ToonRow:
    stripped = line.strip()
    tokens = _split_values(stripped, delimiter)
    if len(tokens) < 2:
        raise ToonFormatError("TOON row must contain at least type and desc columns")

    type_token = tokens[0]
    try:
        type_id = int(type_token)
    except ValueError as exc:
        raise ToonFormatError(f"Invalid geometry type token: {type_token!r}") from exc

    if type_id not in GEOMETRY_FROM_ID:
        raise ToonFormatError(f"Unsupported geometry type id: {type_id}")

    desc_token = tokens[1]
    desc = _parse_string_token(desc_token)

    coord_tokens = tokens[2:]
    coords: List[float | int] = []
    for token in coord_tokens:
        coords.append(_parse_numeric_token(token))

    geometry = GEOMETRY_FROM_ID[type_id]
    if geometry == "bbox_2d" and len(coords) != 4:
        raise ToonFormatError(
            "bbox rows must contain exactly 4 coordinates; "
            f"received {len(coords)}"
        )
    if geometry == "quad" and len(coords) != 8:
        raise ToonFormatError(
            "quad rows must contain exactly 8 coordinates; "
            f"received {len(coords)}"
        )
    if geometry == "line" and len(coords) % 2 != 0:
        raise ToonFormatError(
            "line rows must contain an even number of coordinates"
        )

    return ToonRow(type_id=type_id, desc=desc, coords=tuple(coords))


def _format_header(length: int, delimiter: str) -> str:
    suffix = ""
    if delimiter == "\t":
        suffix = "\t"
    elif delimiter == "|":
        suffix = "|"
    elif delimiter != DEFAULT_DELIMITER:
        raise ToonFormatError(f"Unsupported delimiter: {delimiter!r}")

    field_segment = delimiter.join(FIELD_NAMES)
    return f"{HEADER_KEY}[{length}{suffix}]{{{field_segment}}}:"


def _parse_header(line: str) -> dict[str, object]:
    if not line.endswith(":"):
        raise ToonFormatError("TOON header must end with ':'")

    body = line[:-1]
    bracket_start = body.find("[")
    if bracket_start == -1:
        raise ToonFormatError("TOON header missing '[' segment")

    key = body[:bracket_start]
    if key != HEADER_KEY:
        raise ToonFormatError(f"Unexpected TOON header key: {key!r}")

    bracket_end = body.find("]", bracket_start)
    if bracket_end == -1:
        raise ToonFormatError("TOON header missing closing ']' segment")

    bracket_segment = body[bracket_start + 1 : bracket_end]

    delimiter = DEFAULT_DELIMITER
    if bracket_segment.startswith("#"):
        bracket_segment = bracket_segment[1:]
    if bracket_segment.endswith("\t"):
        delimiter = "\t"
        bracket_segment = bracket_segment[:-1]
    elif bracket_segment.endswith("|"):
        delimiter = "|"
        bracket_segment = bracket_segment[:-1]

    try:
        length = int(bracket_segment)
    except ValueError as exc:
        raise ToonFormatError(
            f"Invalid TOON header length segment: {bracket_segment!r}"
        ) from exc

    brace_start = body.find("{", bracket_end)
    brace_end = body.find("}", brace_start)
    if brace_start == -1 or brace_end == -1:
        raise ToonFormatError("TOON header missing field declaration")

    fields_segment = body[brace_start + 1 : brace_end]
    fields = _split_values(fields_segment, delimiter)
    normalized_fields = [_parse_field_name(field) for field in fields if field]

    if tuple(normalized_fields) != FIELD_NAMES:
        raise ToonFormatError(
            "TOON header fields mismatch: expected "
            f"{FIELD_NAMES}, received {tuple(normalized_fields)}"
        )

    return {"length": length, "delimiter": delimiter}


def _parse_field_name(token: str) -> str:
    token = token.strip()
    if not token:
        raise ToonFormatError("Empty field name in TOON header")
    if token.startswith('"'):
        return _parse_string_token(token)
    return token


def _split_values(segment: str, delimiter: str) -> List[str]:
    values: List[str] = []
    current: List[str] = []
    in_quotes = False
    i = 0
    while i < len(segment):
        char = segment[i]
        if char == "\\" and in_quotes:
            if i + 1 >= len(segment):
                raise ToonFormatError("Invalid escape at end of segment")
            current.append(char)
            current.append(segment[i + 1])
            i += 2
            continue
        if char == '"':
            in_quotes = not in_quotes
            current.append(char)
            i += 1
            continue
        if char == delimiter and not in_quotes:
            values.append("".join(current).strip())
            current.clear()
            i += 1
            continue
        current.append(char)
        i += 1
    if current or segment.endswith(delimiter):
        values.append("".join(current).strip())
    return values


def _encode_desc(value: str, delimiter: str) -> str:
    if _is_safe_unquoted(value, delimiter):
        return value
    return f'"{_escape_string(value)}"'


def _parse_string_token(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if token.startswith('"'):
        if not token.endswith('"') or len(token) == 1:
            raise ToonFormatError("Unterminated quoted string in TOON row")
        content = token[1:-1]
        return _unescape_string(content)
    return token


def _parse_numeric_token(token: str) -> float | int:
    token = token.strip()
    if not token:
        raise ToonFormatError("Empty numeric token in TOON row")
    try:
        if _looks_integer(token):
            return int(token, 10)
        return float(token)
    except ValueError as exc:
        raise ToonFormatError(f"Invalid numeric token: {token!r}") from exc


def _looks_integer(token: str) -> bool:
    if token.startswith("-"):
        token = token[1:]
    return token.isdigit()


def _coerce_coords(values: Sequence[float | int]) -> Tuple[float | int, ...]:
    coords: List[float | int] = []
    for value in values:
        if isinstance(value, bool):
            raise ToonFormatError("Coordinate values cannot be boolean")
        if isinstance(value, (int, float)):
            coords.append(value)
            continue
        raise ToonFormatError(
            f"Coordinate values must be numeric; received {type(value)!r}"
        )
    return tuple(coords)


def _format_numeric(value: float | int) -> str:
    if isinstance(value, bool):
        raise ToonFormatError("Coordinate values cannot be boolean")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, ".15g")
    raise ToonFormatError(
        f"Coordinate values must be numeric; received {type(value)!r}"
    )


def _is_safe_unquoted(value: str, delimiter: str) -> bool:
    if not value:
        return False
    if value != value.strip():
        return False
    if _is_literal_like(value):
        return False
    if ":" in value:
        return False
    if any(ch in value for ch in "{}[]"):
        return False
    if "\n" in value or "\r" in value or "\t" in value:
        return False
    if delimiter in value:
        return False
    if value.startswith("-"):
        return False
    if '"' in value or "\\" in value:
        return False
    return True


def _is_literal_like(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "false", "null"}:
        return True
    return _is_numeric_like(value)


def _is_numeric_like(value: str) -> bool:
    if not value:
        return False
    num = value
    if num.startswith("-"):
        num = num[1:]
    if not num:
        return False
    # Integer
    if num.isdigit():
        return True
    # Decimal or scientific notation
    try:
        float(value)
        return True
    except ValueError:
        return False


def _escape_string(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _unescape_string(value: str) -> str:
    result: List[str] = []
    i = 0
    while i < len(value):
        char = value[i]
        if char != "\\":
            result.append(char)
            i += 1
            continue
        if i + 1 >= len(value):
            raise ToonFormatError("Trailing backslash in string literal")
        nxt = value[i + 1]
        if nxt == "n":
            result.append("\n")
        elif nxt == "t":
            result.append("\t")
        elif nxt == "r":
            result.append("\r")
        elif nxt == "\\":
            result.append("\\")
        elif nxt == '"':
            result.append('"')
        else:
            raise ToonFormatError(f"Unsupported escape sequence: \\{nxt}")
        i += 2
    return "".join(result)


def _validate_delimiter(delimiter: str) -> None:
    if delimiter not in {DEFAULT_DELIMITER, "\t", "|"}:
        raise ToonFormatError(f"Unsupported delimiter: {delimiter!r}")


__all__ = [
    "DEFAULT_DELIMITER",
    "FIELD_NAMES",
    "GEOMETRY_FROM_ID",
    "GEOMETRY_TO_ID",
    "ToonFormatError",
    "ToonRow",
    "decode_toon_block",
    "decode_toon_payload",
    "encode_toon_block",
    "toon_rows_to_payload",
]

