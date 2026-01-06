"""Parsing and schema validation helpers for dense GRPO rewards."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from ..summary.parsing import JsonDuplicateKeyError, loads_json_rejecting_duplicate_keys
from src.utils.unstructured import UnstructuredMapping

_OBJECT_KEY_RE = re.compile(r"^object_(\d+)$")
_TASK_TOKEN = "<TASK=DETECTION>"
_WHITESPACE_RE = re.compile(r"\s+")


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _as_float(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError(f"{path} must be a finite number")
    return float(value)


def is_dense_mode(meta: Any) -> bool:
    if not isinstance(meta, Mapping):
        return False
    meta_map = cast(UnstructuredMapping, meta)
    return meta_map.get("_fusion_mode") == "dense"


def resolve_dense_domain_token(meta: Any) -> str | None:
    """Resolve expected <DOMAIN=...> token for dense samples.

    Resolution order (best-effort):
    - metadata._fusion_template (e.g. target_dense_bbu / target_dense_rru / target_dense)
    - metadata._fusion_source (e.g. bbu_dense / rru_dense)
    """

    if not isinstance(meta, Mapping):
        return None

    meta_map = cast(UnstructuredMapping, meta)

    template = meta_map.get("_fusion_template")
    if isinstance(template, str):
        tmpl = template.strip().lower()
        if tmpl in {"target_dense_bbu", "summary_bbu", "target_dense"} or "bbu" in tmpl:
            return "BBU"
        if tmpl in {"target_dense_rru", "summary_rru"} or "rru" in tmpl:
            return "RRU"

    source = meta_map.get("_fusion_source")
    if isinstance(source, str):
        src = source.strip().lower()
        if src.startswith("bbu"):
            return "BBU"
        if src.startswith("rru"):
            return "RRU"

    return None


def expected_dense_header(meta: Any) -> str | None:
    """Return the expected dense header string for a sample, or None when unknown.

    Header contract (line 1):
      `<DOMAIN={BBU|RRU}>, <TASK=DETECTION>`

    Domain token is resolved from fusion metadata (best-effort) and is intentionally
    not derived from the model output or from GT payload.
    """

    if not is_dense_mode(meta):
        return None
    expected_domain = resolve_dense_domain_token(meta)
    if expected_domain is None:
        return None
    return f"<DOMAIN={expected_domain}>, {_TASK_TOKEN}"


def check_dense_completion_format(*, lines: Sequence[str], meta: Any) -> bool:
    """Cheap dense completion format check (no JSON parsing).

    This is designed for `dense.format` to avoid redundant strict parsing.
    """

    if len(lines) != 2:
        return False
    expected = expected_dense_header(meta)
    if expected is None:
        return False
    if lines[0].strip() != expected:
        return False
    json_line = lines[1].strip()
    return json_line.startswith("{") and json_line.endswith("}")


def _normalize_kv_text(value: str) -> str:
    return _WHITESPACE_RE.sub("", value or "")


@dataclass(frozen=True)
class DenseAttr:
    key: str
    value: str


@dataclass(frozen=True)
class DenseDesc:
    raw: str
    attrs: tuple[DenseAttr, ...]

    def get(self, key: str) -> str:
        needle = str(key)
        for item in self.attrs:
            if item.key == needle:
                return item.value
        return ""


@dataclass(frozen=True)
class DenseGeometry:
    type: Literal["bbox_2d", "poly", "line"]
    points: tuple[float, ...]


@dataclass(frozen=True)
class DenseObject:
    key: str
    desc: DenseDesc
    geometry: DenseGeometry

    @property
    def category(self) -> str:
        return self.desc.get("类别")


@dataclass(frozen=True)
class DenseParsedPayload:
    objects: tuple[DenseObject, ...]


@dataclass(frozen=True)
class DenseParsedStrict:
    """Strict parsing result for a model completion in dense mode."""

    domain_token: str
    objects: tuple[DenseObject, ...]


def parse_desc(desc: str) -> DenseDesc:
    raw = str(desc)
    attrs: list[DenseAttr] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k_norm = _normalize_kv_text(k)
        v_norm = _normalize_kv_text(v)
        if not k_norm:
            continue
        attrs.append(DenseAttr(key=k_norm, value=v_norm))
    return DenseDesc(raw=raw, attrs=tuple(attrs))


def _flatten_points(
    value: object,
    *,
    path: str,
    min_points: int,
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{path} must be a sequence")

    seq: list[object] = list(cast(Sequence[object], value))
    if not seq:
        raise ValueError(f"{path} must be non-empty")

    # Pair-form: [[x, y], ...]
    pair_form = True
    for item in seq:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            pair_form = False
            break

    if pair_form:
        pts: list[float] = []
        for idx, pair_obj in enumerate(seq):
            pair_seq = list(cast(Sequence[object], pair_obj))
            if len(pair_seq) != 2:
                raise ValueError(f"{path}[{idx}] must have length 2")
            x, y = pair_seq
            pts.append(_as_float(x, path=f"{path}[{idx}][0]"))
            pts.append(_as_float(y, path=f"{path}[{idx}][1]"))
        if len(pts) // 2 < min_points:
            raise ValueError(f"{path} must contain at least {min_points} points")
        return tuple(pts)

    # Flat-form: [x1, y1, ...]
    pts_f: list[float] = []
    for idx, p in enumerate(seq):
        pts_f.append(_as_float(p, path=f"{path}[{idx}]"))
    if len(pts_f) % 2 != 0:
        raise ValueError(f"{path} must have even length")
    if len(pts_f) // 2 < min_points:
        raise ValueError(f"{path} must contain at least {min_points} points")
    return tuple(pts_f)


def _parse_object_key(raw: object, *, path: str) -> tuple[str, int]:
    key = str(raw)
    match = _OBJECT_KEY_RE.match(key)
    if not match:
        raise ValueError(f"{path}: invalid object key {key!r} (expected object_N)")
    return key, int(match.group(1))


def parse_dense_object(
    key: str,
    raw: object,
    *,
    path: str,
) -> DenseObject:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path} must be a mapping")

    raw_map = cast(UnstructuredMapping, raw)

    desc_raw = raw_map.get("desc")
    if not isinstance(desc_raw, str) or not desc_raw.strip():
        raise ValueError(f"{path}.desc must be a non-empty string")
    desc = parse_desc(desc_raw)

    geom_keys = [k for k in ("bbox_2d", "poly", "line") if k in raw_map]
    if len(geom_keys) != 1:
        raise ValueError(
            f"{path} must contain exactly one geometry key in {{bbox_2d, poly, line}}"
        )
    gkey = geom_keys[0]
    gval = raw_map.get(gkey)

    if gkey == "bbox_2d":
        if not isinstance(gval, Sequence) or isinstance(gval, (str, bytes)):
            raise TypeError(f"{path}.bbox_2d must be a sequence")
        bbox = list(cast(Sequence[object], gval))
        if len(bbox) != 4:
            raise ValueError(f"{path}.bbox_2d must have length 4")
        bbox_points: list[float] = []
        for idx, v in enumerate(bbox):
            bbox_points.append(_as_float(v, path=f"{path}.bbox_2d[{idx}]"))
        if "line_points" in raw_map:
            raise ValueError(f"{path}.line_points is only allowed for line geometry")
        geom = DenseGeometry(type="bbox_2d", points=tuple(bbox_points))
        return DenseObject(key=key, desc=desc, geometry=geom)

    if gkey == "poly":
        poly_points = _flatten_points(gval, path=f"{path}.poly", min_points=3)
        if "line_points" in raw_map:
            raise ValueError(f"{path}.line_points is only allowed for line geometry")
        geom = DenseGeometry(type="poly", points=poly_points)
        return DenseObject(key=key, desc=desc, geometry=geom)

    # line
    line_points = _flatten_points(gval, path=f"{path}.line", min_points=2)
    if "line_points" in raw_map:
        lp = raw_map.get("line_points")
        if not isinstance(lp, (int, float)) or isinstance(lp, bool):
            raise TypeError(f"{path}.line_points must be a number")
        expected = len(line_points) // 2
        if int(lp) != expected:
            raise ValueError(
                f"{path}.line_points must equal number of point pairs ({expected})"
            )
    geom = DenseGeometry(type="line", points=line_points)
    return DenseObject(key=key, desc=desc, geometry=geom)


def parse_dense_payload_mapping(*, raw: object, path: str) -> DenseParsedPayload:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{path} must be a mapping of object_N -> object payload")

    raw_map = cast(UnstructuredMapping, raw)
    sortable: list[tuple[int, str, DenseObject]] = []
    for k, v in raw_map.items():
        key, idx = _parse_object_key(k, path=f"{path}.keys")
        obj = parse_dense_object(key, v, path=f"{path}.{key}")
        sortable.append((idx, key, obj))

    sortable.sort(key=lambda x: (x[0], x[1]))
    return DenseParsedPayload(objects=tuple(item[2] for item in sortable))


def parse_dense_completion_strict(*, text: str, meta: Any) -> DenseParsedStrict:
    if not is_dense_mode(meta):
        raise ValueError("not a dense-mode sample")

    lines = text.strip().splitlines()
    if len(lines) != 2:
        raise ValueError("dense completion must contain exactly 2 lines")

    expected_domain = resolve_dense_domain_token(meta)
    if expected_domain is None:
        raise ValueError("unable to resolve expected dense DOMAIN token from metadata")

    expected_header = f"<DOMAIN={expected_domain}>, {_TASK_TOKEN}"
    if lines[0].strip() != expected_header:
        raise ValueError("dense header mismatch")

    json_line = lines[1].strip()
    try:
        parsed = loads_json_rejecting_duplicate_keys(json_line)
    except JsonDuplicateKeyError:
        raise
    except Exception as exc:
        raise ValueError("dense JSON parse failed") from exc

    payload = parse_dense_payload_mapping(raw=parsed, path="completion.json")
    return DenseParsedStrict(domain_token=expected_domain, objects=payload.objects)


__all__ = [
    "DenseAttr",
    "DenseDesc",
    "DenseGeometry",
    "DenseObject",
    "DenseParsedPayload",
    "DenseParsedStrict",
    "JsonDuplicateKeyError",
    "check_dense_completion_format",
    "expected_dense_header",
    "is_dense_mode",
    "parse_dense_completion_strict",
    "parse_dense_payload_mapping",
    "parse_desc",
    "resolve_dense_domain_token",
]
