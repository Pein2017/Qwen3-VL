from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Sequence

import matplotlib.patches as patches
from matplotlib import font_manager
import matplotlib.pyplot as plt
from PIL import Image

from data_conversion.pipeline.coordinate_manager import CoordinateManager

logger = logging.getLogger(__name__)


def canonicalize_poly(points: List[int | float]) -> List[int]:
    """
    Canonicalize polygon ordering to match prompts.py (top-left start, clockwise traversal).

    Uses CoordinateManager._canonical_poly_ordering so visualization matches the
    exact ordering enforced during data conversion/inference.
    """
    if not isinstance(points, (list, tuple)) or len(points) < 8 or len(points) % 2 != 0:
        return [int(round(v)) for v in (points or [])]

    point_pairs = [
        (float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)
    ]
    try:
        ordered = CoordinateManager._canonical_poly_ordering(point_pairs)
    except ValueError as exc:
        logger.warning("Failed to canonicalize polygon %s: %s", points, exc)
        ordered = point_pairs

    return [int(round(v)) for xy in ordered for v in xy]


def draw_objects(ax, img: Image.Image, objects: List[Dict[str, Any]], color_map: Dict[str, str], scaled: bool, show_labels: bool = True) -> None:
    ax.imshow(img)
    ax.axis("off")
    w, h = img.size
    for obj in objects:
        gtype = obj["type"]
        pts = obj["points"]
        desc = obj.get("desc", "")
        pts_px = pts if scaled else _inverse_scale(pts, w, h)
        # NOTE: Do NOT canonicalize polygon points - they are already in correct clockwise order
        # from geometry transformations. Reordering breaks rotated polygons!
        color = color_map.get(desc) or "#000000"
        
        # Determine label position (top-left corner of geometry)
        label_x, label_y = None, None
        
        if gtype == "bbox_2d" and len(pts_px) == 4:
            x1, y1, x2, y2 = pts_px
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
            label_x, label_y = x1, y1
        elif gtype == "poly" and len(pts_px) >= 8 and len(pts_px) % 2 == 0:
            # Ensure visualization uses canonical ordering (same as pipeline)
            pts_px = canonicalize_poly(pts_px)
            poly_coords = [(pts_px[i], pts_px[i + 1]) for i in range(0, len(pts_px), 2)]
            poly = patches.Polygon(poly_coords, closed=True, fill=False, edgecolor=color, linewidth=2, linestyle="--", alpha=0.9)
            ax.add_patch(poly)
            # Use top-most point for label
            label_x = min(pts_px[::2])
            label_y = min(pts_px[1::2])
        elif gtype == "line" and len(pts_px) >= 4 and len(pts_px) % 2 == 0:
            xs = pts_px[::2]
            ys = pts_px[1::2]
            ax.plot(xs, ys, color=color, linewidth=3, linestyle="-", marker="o", markersize=3, alpha=0.9)
            # Use first point for label
            label_x, label_y = xs[0], ys[0]
        
        # Draw text label if requested and desc is non-empty
        if show_labels and desc and label_x is not None and label_y is not None:
            ax.text(
                label_x,
                label_y - 5,  # Offset above the shape
                desc,
                color=color,
                fontsize=8,
                weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=color, linewidth=1, pad=2),
                verticalalignment='bottom',
                horizontalalignment='left',
            )


def inverse_scale(points: List[int | float], w: int, h: int) -> List[int]:
    return _inverse_scale(points, w, h)


def _inverse_scale(points: List[int | float], w: int, h: int) -> List[int]:
    out: List[int] = []
    for i, v in enumerate(points):
        try:
            fv = float(v)
        except Exception:
            fv = 0.0
        fv = max(0.0, min(1000.0, fv))
        out.append(int(round(fv / 1000.0 * (w if i % 2 == 0 else h))))
    return out


def generate_colors(labels: List[str]) -> Dict[str, str]:
    base_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F8C471", "#82E0AA", "#F1948A", "#85929E", "#F4D03F",
        "#AED6F1", "#A9DFBF", "#F9E79F", "#D7BDE2", "#A2D9CE",
        "#FADBD8", "#D5DBDB",
    ]
    colors: Dict[str, str] = {}
    for i, label in enumerate(sorted(set(labels))):
        colors[label] = base_colors[i % len(base_colors)]
    return colors


def create_legend(fig, color_map: Dict[str, str], counts: Dict[str, List[int]]) -> None:
    legend_elements = []
    active = [label for label, c in counts.items() if c[0] > 0 or c[1] > 0]
    active.sort(key=lambda label: sum(counts[label]), reverse=True)
    for label in active:
        import matplotlib.patches as patches
        gt_c, pr_c = counts[label]
        legend_label = f"{label} ({gt_c}/{pr_c})"
        legend_elements.append(patches.Patch(facecolor="none", edgecolor=color_map[label], label=legend_label))
    if not legend_elements:
        return
    legend = fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        framealpha=0.95,
        fontsize=8,
        title="Object Categories (GT/Pred)",
        title_fontsize=9,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("lightgray")


def create_caption_completeness_text(fig, caption_series: Sequence[tuple[str, Dict[str, int]]]) -> None:
    if not caption_series:
        return
    headers = [label for label, _ in caption_series]
    lines = ["Caption completeness (GT vs Aug)"]
    header_line = "状态".ljust(6) + " ".join(label.center(8) for label in headers)
    lines.append(header_line)
    for status in ("显示完整", "只显示部分", "未标记"):
        row = status.ljust(6)
        for _, counts in caption_series:
            row += str(counts.get(status, 0)).center(8)
        lines.append(row)
    fig.text(
        0.02,
        0.02,
        "\n".join(lines),
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="lightgray"),
    )


__all__ = [
    "canonicalize_poly",
    "draw_objects",
    "inverse_scale",
    "generate_colors",
    "create_legend",
    "create_caption_completeness_text",
]


_font_candidates = [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK KR",
    ]),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK KR",
    ]),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", ["WenQuanYi Zen Hei"]),
    ("/usr/share/fonts/truetype/arphic/ukai.ttc", ["AR PL UKai CN"]),
]
_font_installed = False
for path, family_list in _font_candidates:
    if not os.path.exists(path):
        continue
    try:
        font_manager.fontManager.addfont(path)
        # Ensure sans-serif fallback list exists
        current_sans = plt.rcParams.get("font.sans-serif", [])
        if isinstance(current_sans, str):
            current_sans = [current_sans]
        updated = list(current_sans)
        for fam in family_list:
            if fam not in updated:
                updated.append(fam)
        plt.rcParams["font.sans-serif"] = updated
        # Point default family to sans-serif so the above list is used
        plt.rcParams["font.family"] = ["sans-serif"]
        _font_installed = True
        break
    except Exception:
        continue
if not _font_installed:
    plt.rcParams.setdefault("font.family", ["sans-serif"])
