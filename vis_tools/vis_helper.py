from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib.patches as patches
from PIL import Image


def canonicalize_quad(points8: List[int | float]) -> List[int]:
    if not isinstance(points8, (list, tuple)) or len(points8) != 8:
        return [int(round(v)) for v in (points8 or [])]
    pts = [(float(points8[i]), float(points8[i + 1])) for i in range(0, 8, 2)]
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0

    def classify_corner(p: Tuple[float, float]) -> Tuple[int, float]:
        x, y = p
        if x <= cx and y <= cy:
            return (0, -(x + y))
        elif x >= cx and y <= cy:
            return (1, x - y)
        elif x >= cx and y >= cy:
            return (2, x + y)
        else:
            return (3, -x + y)

    sorted_pts = sorted(pts, key=classify_corner)
    if len({classify_corner(p)[0] for p in sorted_pts}) != 4:
        sorted_by_y = sorted(pts, key=lambda p: p[1])
        top = sorted(sorted_by_y[:2], key=lambda p: p[0])
        bottom = sorted(sorted_by_y[2:], key=lambda p: p[0])
        sorted_pts = [top[0], top[1], bottom[1], bottom[0]]
    return [int(round(v)) for xy in sorted_pts for v in xy]


def draw_objects(ax, img: Image.Image, objects: List[Dict[str, Any]], color_map: Dict[str, str], scaled: bool) -> None:
    ax.imshow(img)
    ax.axis("off")
    w, h = img.size
    for obj in objects:
        gtype = obj["type"]
        pts = obj["points"]
        desc = obj.get("desc", "")
        pts_px = pts if scaled else _inverse_scale(pts, w, h)
        if gtype == "quad" and len(pts_px) == 8:
            pts_px = canonicalize_quad(pts_px)
        color = color_map.get(desc) or "#000000"
        if gtype == "bbox_2d" and len(pts_px) == 4:
            x1, y1, x2, y2 = pts_px
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        elif gtype == "quad" and len(pts_px) == 8:
            quad_coords = [(pts_px[i], pts_px[i + 1]) for i in range(0, 8, 2)]
            poly = patches.Polygon(quad_coords, closed=True, fill=False, edgecolor=color, linewidth=2, linestyle="--", alpha=0.9)
            ax.add_patch(poly)
        elif gtype == "line" and len(pts_px) >= 4 and len(pts_px) % 2 == 0:
            xs = pts_px[::2]
            ys = pts_px[1::2]
            ax.plot(xs, ys, color=color, linewidth=3, linestyle="-", marker="o", markersize=3, alpha=0.9)


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
    active = [l for l, c in counts.items() if c[0] > 0 or c[1] > 0]
    active.sort(key=lambda l: sum(counts[l]), reverse=True)
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


__all__ = [
    "canonicalize_quad",
    "draw_objects",
    "inverse_scale",
    "generate_colors",
    "create_legend",
]


