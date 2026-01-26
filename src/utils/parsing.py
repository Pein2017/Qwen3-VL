"""Shared coercion helpers for unstructured inputs (YAML / JSON / CLI-like values).

These helpers are used at boundary layers where we ingest loosely-typed mappings.
They intentionally:
- accept the target type unchanged
- accept common string representations (case/whitespace-insensitive)
- reject bool for int/float (bool is a subclass of int)
- raise TypeError for type mismatches and ValueError for invalid values

See OpenSpec change: 2026-01-21-refactor-src-architecture.
"""

from __future__ import annotations

import math


def _field_label(field: str | None) -> str:
    return str(field) if field else "value"


def coerce_bool(value: object, *, field: str | None = None) -> bool:
    """Coerce a value into a bool.

    Accepted string tokens (case-insensitive, whitespace-insensitive):
    - truthy:  true, 1, yes, y, on
    - falsy:   false, 0, no, n, off
    """
    label = _field_label(field)

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if value in (0, 1, 0.0, 1.0):
            return bool(value)
        raise ValueError(f"{label} must be boolean (0 or 1), got {value!r}")

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(
            f"{label} string value {value!r} is not a recognized boolean representation"
        )

    raise TypeError(f"{label} must be a boolean, got {type(value)!r}")


def coerce_int(value: object, *, field: str | None = None) -> int:
    """Coerce a value into an int.

    - Rejects bool explicitly.
    - Accepts ints, integer-valued floats, and numeric strings.
    """
    label = _field_label(field)

    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer (bool is not allowed)")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"{label} must be an integer, got {value!r}")

    if isinstance(value, str):
        stripped = value.strip()
        try:
            return int(stripped)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer, got {value!r}") from exc

    raise TypeError(f"{label} must be an integer, got {type(value)!r}")


def coerce_float(value: object, *, field: str | None = None) -> float:
    """Coerce a value into a float.

    - Rejects bool explicitly.
    - Accepts ints, floats, and numeric strings.
    """
    label = _field_label(field)

    if isinstance(value, bool):
        raise TypeError(f"{label} must be a number (bool is not allowed)")

    if isinstance(value, (int, float)):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"{label} must be a finite number, got {value!r}")
        return parsed

    if isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = float(stripped)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a number, got {value!r}") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{label} must be a finite number, got {value!r}")
        return parsed

    raise TypeError(f"{label} must be a number, got {type(value)!r}")


__all__ = ["coerce_bool", "coerce_int", "coerce_float"]
