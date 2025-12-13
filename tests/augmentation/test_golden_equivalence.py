from __future__ import annotations

import io
import json
import importlib.util
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from src.datasets.augment import apply_augmentations

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
_GENERATE_PATH = FIXTURE_ROOT / "generate_golden.py"

_spec = importlib.util.spec_from_file_location("golden_fixture_module", _GENERATE_PATH)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)  # type: ignore[arg-type]
_build_pipeline = _module._build_pipeline  # type: ignore[attr-defined]
_extract_geoms = _module._extract_geoms  # type: ignore[attr-defined]


def _load_record() -> Dict[str, Any]:
    with open(FIXTURE_ROOT / "input_record.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_images(img_paths: List[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in img_paths]


def _load_expected(seed: int):
    seed_dir = FIXTURE_ROOT / "golden" / f"seed_{seed}"
    imgs = []
    idx = 0
    while (seed_dir / f"image{idx}.png").exists():
        imgs.append(Image.open(seed_dir / f"image{idx}.png").convert("RGB"))
        idx += 1
    with open(seed_dir / "geometries.json", "r", encoding="utf-8") as f:
        geoms = json.load(f)
    with open(seed_dir / "telemetry.json", "r", encoding="utf-8") as f:
        telemetry = json.load(f)
    return imgs, geoms, telemetry


def _telemetry_to_dict(telemetry: Any) -> Dict[str, Any]:
    if telemetry is None:
        return {}
    return {
        "kept_indices": list(getattr(telemetry, "kept_indices", []) or []),
        "coverages": list(getattr(telemetry, "coverages", []) or []),
        "allows_geometry_drops": bool(getattr(telemetry, "allows_geometry_drops", False)),
        "width": getattr(telemetry, "width", None),
        "height": getattr(telemetry, "height", None),
        "padding_ratio": getattr(telemetry, "padding_ratio", None),
        "skip_reason": getattr(telemetry, "skip_reason", None),
        "skip_counts": dict(getattr(telemetry, "skip_counts", {}) or {}),
    }


def _flatten_geom(geom: Dict[str, Any]) -> List[float]:
    if "bbox_2d" in geom:
        return list(geom["bbox_2d"])
    if "poly" in geom:
        return list(geom["poly"])
    if "line" in geom:
        return list(geom["line"])
    return []


def _assert_images_close(current: List[Image.Image], expected: List[Image.Image]):
    assert len(current) == len(expected)
    for cur, exp in zip(current, expected):
        assert cur.size == exp.size
        cur_arr = np.asarray(cur, dtype=np.float32) / 255.0
        exp_arr = np.asarray(exp, dtype=np.float32) / 255.0
        diff = np.abs(cur_arr - exp_arr)
        assert diff.max() <= 1e-4


def _assert_geoms_close(current: List[Dict[str, Any]], expected: List[Dict[str, Any]]):
    assert len(current) == len(expected)
    for cur, exp in zip(current, expected):
        # Ignore internal augmentation provenance keys (e.g., copy/paste source tracking).
        def _geom_kind(g: Dict[str, Any]) -> str:
            for key in ("bbox_2d", "poly", "line"):
                if key in g:
                    return key
            return ""

        assert _geom_kind(cur) == _geom_kind(exp)
        cur_vals = _flatten_geom(cur)
        exp_vals = _flatten_geom(exp)
        assert len(cur_vals) == len(exp_vals)
        assert np.allclose(cur_vals, exp_vals, atol=1e-4)


def _assert_telemetry_close(cur: Dict[str, Any], exp: Dict[str, Any]):
    assert cur.keys() == exp.keys()
    for key, val in exp.items():
        if isinstance(val, list):
            assert np.allclose(val, cur.get(key, []), atol=1e-6)
        else:
            assert cur.get(key) == val


def _decode_images(img_entries: List[Dict[str, Any]]) -> List[Image.Image]:
    out: List[Image.Image] = []
    for entry in img_entries:
        b = entry.get("bytes")
        img = Image.open(io.BytesIO(b)).convert("RGB")
        out.append(img)
    return out


def test_golden_equivalence_matches_pre_refactor_outputs():
    record = _load_record()
    images = _load_images(record["images"])
    geoms = _extract_geoms(record)

    for seed in (7, 42, 1337, 2024):
        pipeline = _build_pipeline()
        rng = random.Random(seed)
        out_imgs_bytes, out_geoms = apply_augmentations(images, geoms, pipeline, rng=rng)
        cur_images = _decode_images(out_imgs_bytes)

        expected_imgs, expected_geoms, expected_telemetry = _load_expected(seed)
        _assert_images_close(cur_images, expected_imgs)
        _assert_geoms_close(out_geoms, expected_geoms)
        current_telemetry = _telemetry_to_dict(getattr(pipeline, "last_summary", None))
        _assert_telemetry_close(current_telemetry, expected_telemetry)
