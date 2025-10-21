"""
Qwen3-VL visualization script (no CLI):
  - Define configs at top
  - Load model/processor
  - Read JSONL
  - Inference with training user prompt
  - Parse norm1000 predictions; inverse-scale to pixels
  - Plot GT (left) vs Pred (right) in a 1×2 layout and save
"""

from __future__ import annotations

import os
import sys
import json
import ast
import re
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ==============================
# Configs (edit these directly)
# ==============================

# Required paths
CKPT_PATH = "output/stage_1_full_aligner_only/v0-20251021-002857/eff_batch_32-lr_5e-4-epoch_10/checkpoint-540"  # HF dir or merged checkpoint
JSONL_PATH = "data/ds_v2_full/val.jsonl"

# Runtime settings
LIMIT = 10
DEVICE = "cuda:1"
SAVE_DIR = "vis_out/stage_1_longer_epoch"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.05
TOP_P = 0.8


# Optional: override training user prompt (None uses training default)
USER_PROMPT_OVERRIDE: str | None = None

# Dump/Plot settings
SAVE_JSONL = True
DUMP_JSONL_PATH = os.path.join(SAVE_DIR, "gt_vs_pred.jsonl")
PLOT_FROM_JSONL = False
PLOT_JSONL_PATH = DUMP_JSONL_PATH


# ======================================================
# Setup imports to reuse the training user prompt (src/)
# ======================================================

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
try:
    from config.prompts import SYSTEM_PROMPT_B, USER_PROMPT as TRAIN_USER_PROMPT  # type: ignore
except Exception:
    raise Exception("Failed to import prompts")

SYSTEM_PROMPT_TEXT = SYSTEM_PROMPT_B
USER_PROMPT_TEXT = USER_PROMPT_OVERRIDE or TRAIN_USER_PROMPT


# ======================
# Load model/processor
# ======================

print(f"[INFO] Loading model from: {CKPT_PATH}")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    CKPT_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.to(DEVICE)
model.eval()

# Enable CUDA perf/kvcache optimizations when available
try:
    if torch.cuda.is_available() and (isinstance(DEVICE, str) and DEVICE.startswith("cuda")):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            # Prefer FlashAttention kernels where possible
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        except Exception:
            pass
except Exception:
    pass

# Ensure KV cache is used by default
try:
    model.config.use_cache = True
except Exception:
    pass
try:
    model.generation_config.use_cache = True
except Exception:
    pass

processor = AutoProcessor.from_pretrained(CKPT_PATH, trust_remote_code=True)


# ======================
# Inference helpers
# ======================

def run_infer_one(pil_img: Image.Image, prompt: str) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEXT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    # Use the tokenizer's chat template owned by the processor
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    inputs = processor(images=[pil_img], text=[text], return_tensors="pt")
    # Move tensors to model device
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(TEMPERATURE > 0),
            temperature=TEMPERATURE,
            top_p=TOP_P,
            use_cache=True,
        )
    # Strip prompt tokens from the front
    start = inputs["input_ids"].shape[-1]
    gen_only = gen[:, start:]

    # Decode both raw (with specials) and cleaned
    try:
        raw_text = processor.batch_decode(
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
    except Exception:
        raw_text = processor.tokenizer.batch_decode(
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
    try:
        clean_text = processor.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    except Exception:
        clean_text = processor.tokenizer.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    return raw_text, clean_text


# ======================
# Parsing + inverse scale
# ======================

GEOM_KEYS = ("bbox_2d", "quad", "line")


def _extract_outer_json(text: str) -> str | None:
    """Return the largest balanced JSON block by curly braces.

    This drops any trailing incomplete object by truncating to the last position
    where the brace depth returns to zero. If no complete block exists, returns None.
    """
    start: int | None = None
    depth = 0
    last_good_end: int | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0 and start is None:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    last_good_end = i
    if start is not None and last_good_end is not None and last_good_end >= start:
        return text[start : last_good_end + 1]
    # Fallback regex (might capture smaller balanced section)
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else None


def _json_loads_best_effort(s: str):
    try:
        return json.loads(s)
    except Exception:
        try:
            s2 = re.sub(r",\s*\}", "}", re.sub(r",\s*\]", "]", s))
            s2 = s2.replace("'", '"')
            return json.loads(s2)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return None


def parse_prediction(text: str) -> List[Dict[str, Any]]:
    def _build_objects_from_dict(obj_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        parsed_local: List[Dict[str, Any]] = []
        for _, val in sorted(obj_dict.items(), key=lambda kv: kv[0] if isinstance(kv[0], str) else str(kv[0])):
            if not isinstance(val, dict):
                continue
            desc = val.get("desc", "")
            gtype = next((g for g in GEOM_KEYS if g in val), None)
            if gtype is None:
                continue
            pts = val.get(gtype)
            if not isinstance(pts, (list, tuple)) or len(pts) % 2 != 0:
                # Skip objects with incomplete coordinates (e.g., truncated tail)
                continue
            try:
                pts = [int(round(float(x))) for x in pts]
            except Exception:
                continue
            parsed_local.append({"desc": desc, "type": gtype, "points": pts})
        return parsed_local

    # First try: parse as full JSON (balanced root)
    raw = _extract_outer_json(text)
    if raw:
        obj = _json_loads_best_effort(raw)
        if isinstance(obj, dict):
            # Support grouped {"图片_1": {...}} or flat {"object_1": {...}}
            if any(isinstance(k, str) and k.startswith("图片_") for k in obj.keys()):
                try:
                    first_key = sorted(
                        [k for k in obj.keys() if isinstance(k, str) and k.startswith("图片_")],
                        key=lambda x: int(str(x).split("_")[-1]),
                    )[0]
                    obj = obj.get(first_key) or {}
                except Exception:
                    obj = {}
            if isinstance(obj, dict) and obj:
                return _build_objects_from_dict(obj)

    # Fallback: extract per-object dicts from a truncated group body
    # Find the group body start: the '{' after the first occurrence of '图片_'
    try:
        grp_idx = text.find("图片_")
        if grp_idx == -1:
            grp_idx = text.find("\u56fe\u7247_")  # unicode-escaped fallback
        if grp_idx != -1:
            # Find the first '{' after the colon following the group key
            colon_idx = text.find(":", grp_idx)
            brace_idx = text.find("{", colon_idx)
            if brace_idx != -1:
                i = brace_idx + 1
                results: List[Dict[str, Any]] = []
                while True:
                    key_idx = text.find("\"object_", i)
                    if key_idx == -1:
                        break
                    # Find the '{' that starts the value
                    val_colon = text.find(":", key_idx)
                    val_lbrace = text.find("{", val_colon)
                    if val_lbrace == -1:
                        break
                    # Scan braces to find the matching '}' for this object value
                    depth = 0
                    j = val_lbrace
                    end_found = None
                    while j < len(text):
                        ch = text[j]
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                end_found = j
                                break
                        j += 1
                    if end_found is None:
                        # object truncated; stop here (drop incomplete)
                        break
                    # Build a tiny JSON to parse this object
                    obj_str = text[key_idx: end_found + 1]
                    # Reconstruct as {"object_N": { ... }} to json.loads
                    left_quote = obj_str.find('"')
                    sep_colon = obj_str.find(":", left_quote)
                    obj_key = obj_str[left_quote:sep_colon].strip()
                    obj_val = obj_str[sep_colon + 1:].strip()
                    tiny = "{" + obj_key + ":" + obj_val + "}"
                    tiny_obj = _json_loads_best_effort(tiny)
                    if isinstance(tiny_obj, dict):
                        results.extend(_build_objects_from_dict(tiny_obj))
                    i = end_found + 1
                return results
    except Exception:
        pass

    return []


def inverse_scale(points: List[int | float], w: int, h: int) -> List[int]:
    out: List[int] = []
    for i, v in enumerate(points):
        try:
            fv = float(v)
        except Exception:
            fv = 0.0
        fv = max(0.0, min(1000.0, fv))
        out.append(int(round(fv / 1000.0 * (w if i % 2 == 0 else h))))
    return out


# ======================
# Drawing utilities
# ======================

COL_GT = (0.0, 0.8, 0.0)
COL_PRED = (1.0, 0.2, 0.2)


def _generate_colors(labels: List[str]) -> Dict[str, str]:
    base_colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#96CEB4",
        "#FFEAA7",
        "#DDA0DD",
        "#98D8C8",
        "#F7DC6F",
        "#BB8FCE",
        "#85C1E9",
        "#F8C471",
        "#82E0AA",
        "#F1948A",
        "#85929E",
        "#F4D03F",
        "#AED6F1",
        "#A9DFBF",
        "#F9E79F",
        "#D7BDE2",
        "#A2D9CE",
        "#FADBD8",
        "#D5DBDB",
    ]
    colors: Dict[str, str] = {}
    for i, label in enumerate(sorted(set(labels))):
        colors[label] = base_colors[i % len(base_colors)]
    return colors


def _create_legend(fig, color_map: Dict[str, str], counts: Dict[str, List[int]]) -> None:
    legend_elements = []
    active = [l for l, c in counts.items() if c[0] > 0 or c[1] > 0]
    active.sort(key=lambda l: sum(counts[l]), reverse=True)
    for label in active:
        gt_c, pr_c = counts[label]
        legend_label = f"{label} ({gt_c}/{pr_c})"
        legend_elements.append(
            patches.Patch(facecolor="none", edgecolor=color_map[label], label=legend_label)
        )
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


def _canonicalize_quad(points8: List[int | float]) -> List[int]:
    """Reorder 4 points (x1,y1,...,x4,y4) to a canonical clockwise order starting at top-left.

    Mirrors the logic in vis_generation.py (_canonical_quad_ordering):
    - Classify corners by quadrant relative to centroid with tie-breakers
    - If corners are not distinct, fallback by sorting rows then columns
    """
    if not isinstance(points8, (list, tuple)) or len(points8) != 8:
        return [int(round(v)) for v in (points8 or [])]
    pts = [(float(points8[i]), float(points8[i + 1])) for i in range(0, 8, 2)]
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0

    def classify_corner(p: tuple[float, float]) -> tuple[int, float]:
        x, y = p
        if x <= cx and y <= cy:
            return (0, -(x + y))  # top-left: minimize x+y
        elif x >= cx and y <= cy:
            return (1, x - y)  # top-right: maximize x-y
        elif x >= cx and y >= cy:
            return (2, x + y)  # bottom-right: maximize x+y
        else:  # x <= cx and y >= cy
            return (3, -x + y)  # bottom-left: maximize -x+y

    sorted_pts = sorted(pts, key=classify_corner)
    # Ensure we actually have 4 distinct corners; otherwise fallback row-wise
    if len({classify_corner(p)[0] for p in sorted_pts}) != 4:
        sorted_by_y = sorted(pts, key=lambda p: p[1])
        top = sorted(sorted_by_y[:2], key=lambda p: p[0])
        bottom = sorted(sorted_by_y[2:], key=lambda p: p[0])
        sorted_pts = [top[0], top[1], bottom[1], bottom[0]]
    return [int(round(v)) for xy in sorted_pts for v in xy]


def _draw_objects(ax, img: Image.Image, objects: List[Dict[str, Any]], color_map: Dict[str, str], scaled: bool) -> None:
    ax.imshow(img)
    ax.axis("off")
    w, h = img.size
    for obj in objects:
        gtype = obj["type"]
        pts = obj["points"]
        desc = obj.get("desc", "")
        pts_px = pts if scaled else inverse_scale(pts, w, h)
        # Canonicalize quad ordering to avoid self-crossing (use vis_generation logic)
        if gtype == "quad" and len(pts_px) == 8:
            pts_px = _canonicalize_quad(pts_px)

        color = color_map.get(desc) or "#000000"
        if gtype == "bbox_2d" and len(pts_px) == 4:
            x1, y1, x2, y2 = pts_px
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        elif gtype == "quad" and len(pts_px) == 8:
            # Draw polygon with dashed outline similar to vis_generation
            quad_coords = [(pts_px[i], pts_px[i + 1]) for i in range(0, 8, 2)]
            poly = patches.Polygon(quad_coords, closed=True, fill=False, edgecolor=color, linewidth=2, linestyle="--", alpha=0.9)
            ax.add_patch(poly)
        elif gtype == "line" and len(pts_px) >= 4 and len(pts_px) % 2 == 0:
            xs = pts_px[::2]
            ys = pts_px[1::2]
            ax.plot(xs, ys, color=color, linewidth=3, linestyle="-", marker="o", markersize=3, alpha=0.9)


# ======================
# JSONL reading + GT extraction
# ======================


def load_records(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(records: List[Dict[str, Any]], jsonl_path: str) -> None:
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _guess_geom_type_from_len(n_points: int) -> str | None:
    if n_points == 4:
        return "bbox_2d"
    if n_points == 8:
        return "quad"
    if n_points % 2 == 0 and n_points >= 4:
        return "line"
    return None


def extract_gt_objects(rec: Dict[str, Any], image_index: int) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    objects = rec.get("objects")

    # Format A: top-level arrays {ref: [...], bbox: [[...], ...], image_id: [...]}
    if isinstance(objects, dict) and isinstance(objects.get("bbox"), list):
        bboxes = objects.get("bbox") or []
        refs = objects.get("ref") or [""] * len(bboxes)
        image_ids = objects.get("image_id") or [0] * len(bboxes)
        for i, pts in enumerate(bboxes):
            if not isinstance(pts, (list, tuple)) or len(pts) % 2 != 0:
                continue
            if i < len(image_ids) and image_ids[i] != image_index:
                continue
            gtype = _guess_geom_type_from_len(len(pts))
            if gtype is None:
                continue
            desc = refs[i] if i < len(refs) else ""
            objs.append({"desc": desc, "type": gtype, "points": [float(x) for x in pts]})
        return objs

    # Format B: list of per-object dicts [{desc, bbox_2d|quad|line, image_id?}, ...]
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            gtype = next((k for k in ("bbox_2d", "quad", "line") if k in obj), None)
            if gtype is None:
                continue
            pts = obj.get(gtype)
            if not isinstance(pts, (list, tuple)) or len(pts) % 2 != 0:
                continue
            if obj.get("image_id") is not None and int(obj["image_id"]) != image_index:
                continue
            objs.append({"desc": obj.get("desc", ""), "type": gtype, "points": [float(x) for x in pts]})
        return objs

    return objs


# ======================
# Main loop
# ======================


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    root = Path(JSONL_PATH).resolve().parent
    # Provide a consistent default for image path resolving in downstream libs
    os.environ.setdefault("ROOT_IMAGE_DIR", str(root))

    # Optionally plot from a dumped JSONL (skip inference)
    if PLOT_FROM_JSONL:
        print(f"[INFO] Plotting from JSONL (no inference): {PLOT_JSONL_PATH}")
        dumped = list(load_records(PLOT_JSONL_PATH))
        count = 0
        for rec in dumped:
            img_path = rec.get("image_path")
            if not img_path:
                continue
            if not Path(img_path).is_absolute():
                img_path = str((root / img_path).resolve())
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            gt_objs = rec.get("gt", [])
            pred_objs = rec.get("pred", [])

            fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
            _draw_objects(
                ax_l,
                img,
                [
                    {
                        "type": o["type"],
                        "points": [int(round(p)) for p in o["points"]],
                        "desc": o.get("desc", ""),
                    }
                    for o in gt_objs
                ],
                COL_GT,
                scaled=True,
            )
            ax_l.set_title("GT")
            _draw_objects(ax_r, img, pred_objs, COL_PRED, scaled=False)
            ax_r.set_title("Prediction" + ("" if pred_objs else " (parse failed)"))
            out_path = Path(SAVE_DIR) / f"vis_{count:05d}.jpg"
            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            count += 1
            if count >= LIMIT:
                break
        print(f"[DONE] Saved {count} figures to {SAVE_DIR}")
        return

    count = 0
    dumped_records: List[Dict[str, Any]] = []
    for rec in load_records(JSONL_PATH):
        images = (rec.get("images") or [])[:1]  # one image per figure
        if not images:
            continue

        img_path = images[0]
        if not Path(img_path).is_absolute():
            img_path = str((root / img_path).resolve())
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        gt = extract_gt_objects(rec, image_index=0)

        try:
            raw_text, clean_text = run_infer_one(img, USER_PROMPT_TEXT)
            # Print raw first for debugging
            try:
                print("\n=== RAW RESPONSE (skip_special_tokens=False) ===")
                print(raw_text)
            except Exception:
                pass
            pred_objs = parse_prediction(clean_text)
        except Exception:
            pred_objs = []

        # Accumulate JSONL entries for later plotting
        if SAVE_JSONL:
            dumped_records.append(
                {
                    "image_path": images[0],  # keep original relative path
                    "gt": [
                        {
                            "desc": o.get("desc", ""),
                            "type": o["type"],
                            "points": [int(round(p)) for p in o["points"]],
                        }
                        for o in gt
                    ],
                    "pred": pred_objs,
                    "raw_text": raw_text if 'raw_text' in locals() else "",
                    "clean_text": clean_text if 'clean_text' in locals() else "",
                }
            )

        # Build legend color map using labels appearing in this figure
        labels = [o.get("desc", "") for o in gt] + [o.get("desc", "") for o in pred_objs]
        color_map = _generate_colors(labels)

        # Track counts for legend
        counts: Dict[str, List[int]] = {}
        for o in gt:
            key = o.get("desc", "")
            counts.setdefault(key, [0, 0])[0] += 1
        for o in pred_objs:
            key = o.get("desc", "")
            counts.setdefault(key, [0, 0])[1] += 1

        # Plot 1×2
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 7))
        # GT is already in pixels
        _draw_objects(
            ax_l,
            img,
            [
                {
                    "type": o["type"],
                    "points": [int(round(p)) for p in o["points"]],
                    "desc": o.get("desc", ""),
                }
                for o in gt
            ],
            color_map,
            scaled=True,
        )
        ax_l.set_title("GT")

        _draw_objects(ax_r, img, pred_objs, color_map, scaled=False)
        ax_r.set_title("Prediction" + ("" if pred_objs else " (parse failed)"))

        out_path = Path(SAVE_DIR) / f"vis_{count:05d}.jpg"
        fig.tight_layout()
        _create_legend(fig, color_map, counts)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        count += 1
        if count >= LIMIT:
            break

    # Write out dump JSONL if requested
    if SAVE_JSONL and dumped_records:
        write_jsonl(dumped_records, DUMP_JSONL_PATH)
        print(f"[INFO] Dumped GT vs Pred JSONL: {DUMP_JSONL_PATH}")

    print(f"[DONE] Saved {count} figures to {SAVE_DIR}")


if __name__ == "__main__":
    main()


