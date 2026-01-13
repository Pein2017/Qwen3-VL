from __future__ import annotations

from random import Random
from typing import cast

from PIL import Image

from src.datasets.augmentation.base import Compose, PatchOp
from src.datasets.contracts import ConversationRecord, DatasetObject
from src.datasets.preprocessors.augmentation import AugmentationPreprocessor


class _RewritePolyPoints(PatchOp):
    """PatchOp stub that rewrites poly vertex count without crop telemetry."""

    def __init__(self) -> None:
        self.allows_geometry_drops = False

    def apply(self, images, geoms, *, width: int, height: int, rng):  # type: ignore[override]
        out: list[DatasetObject] = []
        for g in cast(list[DatasetObject], geoms):
            if "poly" not in g:
                out.append(g)
                continue
            poly = list(cast(list[float], g["poly"]))
            # Insert an extra vertex on the first edge to change point count:
            # (x0,y0)->(x1,y1) becomes (x0,y0)->(mx,my)->(x1,y1)
            x0, y0, x1, y1 = poly[0], poly[1], poly[2], poly[3]
            mx = (x0 + x1) * 0.5
            my = (y0 + y1) * 0.5
            rewritten = [x0, y0, mx, my, x1, y1, *poly[4:]]
            out.append(cast(DatasetObject, {"poly": rewritten}))
        return images, out


def test_preprocessor_updates_poly_points_when_present() -> None:
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    record = cast(
        ConversationRecord,
        {
            "images": [img],
            "objects": [
                {
                    "poly": [0, 0, 10, 0, 10, 10, 0, 10],
                    "poly_points": 4,
                    "desc": "类别=RRU设备,可见性=完整",
                }
            ],
            "width": 64,
            "height": 64,
        },
    )

    pipeline = Compose([_RewritePolyPoints()])
    pre = AugmentationPreprocessor(augmenter=pipeline, rng=Random(0), bypass_prob=0.0)
    out = pre(record)
    assert out is not None

    objs = cast(list[DatasetObject], out.get("objects") or [])
    assert len(objs) == 1
    assert "poly" in objs[0]
    assert "poly_points" in objs[0]
    assert len(cast(list[float], objs[0]["poly"])) == 10  # 5 points
    assert int(objs[0]["poly_points"]) == 5
