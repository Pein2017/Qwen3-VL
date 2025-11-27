"""Tests for vis_qwen3.parse_prediction with strict JSON validation."""

import json
from typing import Any, Dict, List

import pytest

from vis_tools.vis_qwen3 import parse_prediction


def test_parse_prediction_valid_bbox_2d():
    """Test parsing valid bbox_2d object with flat coordinates."""
    json_text = json.dumps({
        "object_1": {
            "desc": "BBU设备/华为/显示完整",
            "bbox_2d": [100, 200, 300, 400]
        }
    })
    result = parse_prediction(json_text)
    assert len(result) == 1
    assert result[0]["desc"] == "BBU设备/华为/显示完整"
    assert result[0]["type"] == "bbox_2d"
    assert result[0]["points"] == [100, 200, 300, 400]


def test_parse_prediction_valid_poly_nested():
    """Test parsing valid polygon object with nested coordinate pairs."""
    json_text = json.dumps({
        "object_1": {
            "desc": "天线/杆塔/完好",
            "poly": [[0, 0], [50, 0], [50, 30], [0, 30]]
        }
    })
    result = parse_prediction(json_text)
    assert len(result) == 1
    assert result[0]["desc"] == "天线/杆塔/完好"
    assert result[0]["type"] == "poly"
    assert result[0]["points"] == [0, 0, 50, 0, 50, 30, 0, 30]


def test_parse_prediction_valid_line_with_line_points():
    """Test parsing valid line object with line_points field."""
    json_text = json.dumps({
        "object_1": {
            "desc": "光纤/有保护",
            "line": [[10, 10], [20, 20], [30, 40]],
            "line_points": 3
        }
    })
    result = parse_prediction(json_text)
    assert len(result) == 1
    assert result[0]["desc"] == "光纤/有保护"
    assert result[0]["type"] == "line"
    assert result[0]["points"] == [10, 10, 20, 20, 30, 40]


def test_parse_prediction_rejects_line_without_line_points():
    """Test that line objects without line_points are rejected."""
    json_text = json.dumps({
        "object_1": {
            "desc": "光纤/有保护",
            "line": [[10, 10], [20, 20], [30, 40]]
        }
    })
    result = parse_prediction(json_text)
    assert len(result) == 0, "Line without line_points should be rejected"


def test_parse_prediction_rejects_line_with_mismatched_line_points():
    """Test that line objects with mismatched line_points count are rejected."""
    json_text = json.dumps({
        "object_1": {
            "desc": "光纤/有保护",
            "line": [[10, 10], [20, 20], [30, 40]],
            "line_points": 2  # Should be 3, not 2
        }
    })
    result = parse_prediction(json_text)
    assert len(result) == 0, "Line with mismatched line_points should be rejected"


def test_parse_prediction_rejects_line_with_invalid_line_points():
    """Test that line objects with invalid line_points are rejected."""
    json_text = json.dumps({
        "object_1": {
            "desc": "光纤/有保护",
            "line": [[10, 10], [20, 20]],
            "line_points": 1  # Must be >= 2
        }
    })
    result = parse_prediction(json_text)
    assert len(result) == 0, "Line with line_points < 2 should be rejected"


def test_parse_prediction_rejects_multiple_geometry_keys():
    """Test that objects with multiple geometry keys are rejected."""
    json_text = json.dumps({
        "object_1": {
            "desc": "混合几何",
            "bbox_2d": [100, 200, 300, 400],
            "poly": [[0, 0], [50, 0], [50, 30], [0, 30]]
        }
    })
    result = parse_prediction(json_text)
    assert len(result) == 0, "Objects with multiple geometry keys should be rejected"


def test_parse_prediction_rejects_no_geometry_key():
    """Test that objects without any geometry key are rejected."""
    json_text = json.dumps({
        "object_1": {
            "desc": "无几何"
        }
    })
    result = parse_prediction(json_text)
    assert len(result) == 0, "Objects without geometry key should be rejected"


def test_parse_prediction_clamps_coordinates():
    """Test that coordinates are clamped to [0, 1000] range."""
    json_text = json.dumps({
        "object_1": {
            "desc": "超出范围",
            "bbox_2d": [-100, 500, 1500, 2000]
        }
    })
    result = parse_prediction(json_text)
    assert len(result) == 1
    assert result[0]["points"] == [0, 500, 1000, 1000]


def test_parse_prediction_multiple_objects():
    """Test parsing multiple objects in one JSON."""
    json_text = json.dumps({
        "object_1": {
            "desc": "BBU设备/华为",
            "bbox_2d": [100, 200, 300, 400]
        },
        "object_2": {
            "desc": "天线/杆塔",
            "poly": [[0, 0], [50, 0], [50, 30], [0, 30]]
        },
        "object_3": {
            "desc": "光纤/有保护",
            "line": [[10, 10], [20, 20], [30, 40]],
            "line_points": 3
        }
    })
    result = parse_prediction(json_text)
    assert len(result) == 3
    assert result[0]["desc"] == "BBU设备/华为"
    assert result[1]["desc"] == "天线/杆塔"
    assert result[2]["desc"] == "光纤/有保护"


def test_parse_prediction_empty_input():
    """Test parsing empty input."""
    result = parse_prediction("")
    assert len(result) == 0


def test_parse_prediction_invalid_json():
    """Test parsing invalid JSON."""
    result = parse_prediction("not valid json {")
    assert len(result) == 0


def test_parse_prediction_flat_poly_coordinates():
    """Test parsing polygon with flat coordinates (should be flattened correctly)."""
    json_text = json.dumps({
        "object_1": {
            "desc": "天线/杆塔",
            "poly": [0, 0, 50, 0, 50, 30, 0, 30]
        }
    })
    result = parse_prediction(json_text)
    assert len(result) == 1
    assert result[0]["points"] == [0, 0, 50, 0, 50, 30, 0, 30]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
