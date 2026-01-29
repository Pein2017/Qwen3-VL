from __future__ import annotations

import pytest

from src.utils.parsing import coerce_bool, coerce_float, coerce_int


def test_coerce_bool_accepts_canonical_tokens() -> None:
    assert coerce_bool(True, field="x") is True
    assert coerce_bool(False, field="x") is False
    assert coerce_bool(" YES ", field="x") is True
    assert coerce_bool("off", field="x") is False
    assert coerce_bool(1, field="x") is True
    assert coerce_bool(0, field="x") is False


def test_coerce_bool_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        coerce_bool(2, field="x")
    with pytest.raises(ValueError):
        coerce_bool("maybe", field="x")
    with pytest.raises(TypeError):
        coerce_bool([], field="x")


def test_coerce_int_rejects_bool() -> None:
    with pytest.raises(TypeError):
        coerce_int(True, field="x")


def test_coerce_int_accepts_int_like_values() -> None:
    assert coerce_int(3, field="x") == 3
    assert coerce_int(3.0, field="x") == 3
    assert coerce_int(" 4 ", field="x") == 4
    with pytest.raises(ValueError):
        coerce_int(3.2, field="x")


def test_coerce_float_rejects_bool_and_non_finite() -> None:
    with pytest.raises(TypeError):
        coerce_float(True, field="x")
    with pytest.raises(ValueError):
        coerce_float("nan", field="x")


def test_coerce_float_accepts_numeric_values() -> None:
    assert coerce_float(1, field="x") == 1.0
    assert coerce_float("0.5", field="x") == 0.5
