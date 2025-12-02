"""
Qwen3-VL visualization script:
  - Define configs at top
  - Load model/processor
  - Read JSONL
  - Inference with training user prompt
  - Parse norm1000 predictions; inverse-scale to pixels
  - Plot GT (left) vs Pred (middle) vs Legend (right) in a 1×3 layout and save

Usage:
  python vis_qwen3.py [device_id]
  python vis_qwen3.py 1  # Use cuda:1
  python vis_qwen3.py    # Use cuda:0 (default)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import torch
import gc
from pathlib import Path
from typing import Any, Dict, List

_SKIP_VIS_DEPS = os.environ.get("QWEN3_VL_NO_VIS_DEPS") is not None

if not _SKIP_VIS_DEPS:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
else:
    patches = None  # type: ignore
    plt = None  # type: ignore
    torch = None  # type: ignore
    Image = None  # type: ignore
from transformers import (  # noqa: E402
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.generation.logits_process import (  # noqa: E402
    LogitsProcessor,
    LogitsProcessorList,
)

from vis_tools import evaluate as geom_eval  # noqa: E402
from vis_tools.vis_helper import (  # noqa: E402
    canonicalize_poly,
    draw_objects,
)

# ==============================
# Parse CLI arguments (deferred to main)
# ==============================


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Qwen3-VL visualization script")
    parser.add_argument(
        "device_id",
        type=int,
        nargs="?",
        default=None,
        help="CUDA device ID (optional, if not provided uses DEVICE from runtime settings)",
    )
    return parser.parse_args()


# ==============================
# Configs (edit these directly)
# ==============================

# Required paths
CKPT_PATH = "output/12-1/fusion_dlora_merged/lm_head/checkpoint-700"  # HF dir or merged checkpoint  # HF dir or merged checkpoint
JSONL_PATH = "data/bbu_full_1024_poly-need_review/val.jsonl"

# Runtime settings
LIMIT = 10
DEVICE = "cuda:3"  # Default device; can be overridden by CLI arg in main()
SAVE_DIR = "vis_out/12-1/lm_head"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.01  # Moderate temperature for diversity without excessive randomness
TOP_P = 0.95  # Nucleus sampling - cuts off low-probability tail for better diversity
REPETITION_PENALTY = (
    1.05  # Minimal global penalty to preserve recall (only prevents token-level loops)
)
NO_REPEAT_NGRAM_SIZE = 0  # Prevent repeating 5-grams (catches entire object structures without shifting distribution much)

# Optional stopping/duplicate controls
# - Set True to stop once the root JSON '}' closes (does not limit object count)
STOP_AT_BALANCED_JSON = False
# - Optional safety cap on number of objects the model may emit (None disables)
MAX_OBJECTS_CAP: int | None = None
# Dedup aggressiveness: require at least this many matched tokens and presence of geometry marker
DEDUP_MIN_PREFIX_TOKENS = 24
GEOMETRY_MARKERS = ('"line"', '"bbox_2d"', '"poly"')
DEDUP_MAX_MATCH_WINDOW_TOKENS = 256

# JSON format: must match training format ("standard")
# If None, will try to infer from checkpoint path (looks for "json_format_standard")
JSON_FORMAT: str | None = None  # Set to "standard" to override auto-detection

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
    from src.config.prompts import (  # type: ignore
        USER_PROMPT_JSON,
        build_dense_system_prompt,
    )
except Exception:
    raise Exception("Failed to import prompts")
try:
    from src.datasets.geometry import normalize_points  # type: ignore
except Exception:
    normalize_points = None  # type: ignore

# Infer json_format from checkpoint path if not explicitly set
_inferred_format: str | None = None
if JSON_FORMAT is None:
    # Try to infer from checkpoint path (looks for "json_format_standard")
    if "json_format_standard" in CKPT_PATH or "json_format-standard" in CKPT_PATH:
        _inferred_format = "standard"

# Use explicit format or inferred, fallback to "standard"
json_format = JSON_FORMAT or _inferred_format or "standard"
if json_format not in ("standard",):
    print(f"[WARNING] Unknown json_format '{json_format}', defaulting to 'standard'")
    json_format = "standard"

# Build system prompt with the correct format hint (matches training)
SYSTEM_PROMPT_TEXT = build_dense_system_prompt(json_format)
default_user_prompt = USER_PROMPT_JSON

USER_PROMPT_TEXT = USER_PROMPT_OVERRIDE or default_user_prompt

print(f"[INFO] Using JSON format: {json_format}")
print("[INFO] System prompt format hint: standard")


if not _SKIP_VIS_DEPS:
    # ======================
    # Load model/processor
    # ======================

    print(f"[INFO] Loading model from: {CKPT_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CKPT_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(torch.device(DEVICE))  # type: ignore[arg-type]
    model.eval()

    # Enable CUDA perf/kvcache optimizations when available
    try:
        if torch.cuda.is_available() and (
            isinstance(DEVICE, str) and DEVICE.startswith("cuda")
        ):
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
                torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_mem_efficient=False, enable_math=False
                )
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
        gc = getattr(model, "generation_config", None)
        if gc is not None:
            gc.use_cache = True
    except Exception:
        pass

processor = AutoProcessor.from_pretrained(CKPT_PATH, trust_remote_code=True)
processor.image_processor.do_resize = False

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
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    inputs = processor(images=[pil_img], text=[text], return_tensors="pt")
    # Move tensors to model device
    inputs = {
        k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()
    }

    # Build optional stopping criteria (balanced JSON and/or object cap)
    prompt_len = int(inputs.get("input_ids").shape[-1])  # type: ignore[union-attr]

    stopping: StoppingCriteriaList | None = None

    if STOP_AT_BALANCED_JSON or (MAX_OBJECTS_CAP is not None and MAX_OBJECTS_CAP > 0):

        class _BalancedJsonStopper(StoppingCriteria):
            def __init__(self, prompt_len: int, max_objects: int | None) -> None:
                self.prompt_len = prompt_len
                self.max_objects = max_objects

            def _decode(self, ids: torch.Tensor) -> str:
                try:
                    return processor.tokenizer.decode(
                        ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )  # type: ignore[attr-defined]
                except Exception:
                    return ""

            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
            ) -> torch.BoolTensor:
                batch_size = input_ids.shape[0] if input_ids is not None else 1
                device = (
                    input_ids.device if input_ids is not None else torch.device("cpu")
                )
                if input_ids is None or input_ids.size(0) == 0:
                    return torch.full(
                        (batch_size,), False, device=device, dtype=torch.bool
                    )  # type: ignore[return-value]
                seq = input_ids[0]
                gen_ids = seq[self.prompt_len :]
                if gen_ids.numel() == 0:
                    return torch.full(
                        (batch_size,), False, device=device, dtype=torch.bool
                    )  # type: ignore[return-value]
                text = self._decode(gen_ids)
                if not text:
                    return torch.full(
                        (batch_size,), False, device=device, dtype=torch.bool
                    )  # type: ignore[return-value]
                if self.max_objects is not None and self.max_objects > 0:
                    try:
                        if text.count('"object_') >= self.max_objects:
                            return torch.full(
                                (batch_size,), True, device=device, dtype=torch.bool
                            )  # type: ignore[return-value]
                    except Exception:
                        pass
                if not STOP_AT_BALANCED_JSON:
                    return torch.full(
                        (batch_size,), False, device=device, dtype=torch.bool
                    )  # type: ignore[return-value]
                depth = 0
                started = False
                in_str = False
                esc = False
                for ch in text:
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    else:
                        if ch == '"':
                            in_str = True
                            continue
                        if ch == "{":
                            depth += 1
                            started = True
                        elif ch == "}":
                            if depth > 0:
                                depth -= 1
                                if started and depth == 0:
                                    return torch.full(
                                        (batch_size,),
                                        True,
                                        device=device,
                                        dtype=torch.bool,
                                    )  # type: ignore[return-value]
                return torch.full((batch_size,), False, device=device, dtype=torch.bool)  # type: ignore[return-value]

        stopping = StoppingCriteriaList(
            [_BalancedJsonStopper(prompt_len, MAX_OBJECTS_CAP)]
        )

    # Build logits processor to block exact duplicate object values (e.g., identical line arrays)
    class _DedupObjectValueProcessor(LogitsProcessor):
        def __init__(self, prompt_len: int) -> None:
            self.prompt_len = prompt_len
            self.seen_values: set[str] = set()
            self.bad_token_sequences: list[list[int]] = []

        def _decode(self, ids: torch.Tensor) -> str:
            try:
                return processor.tokenizer.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )  # type: ignore[attr-defined]
            except Exception:
                return ""

        def _encode(self, s: str) -> list[int]:
            try:
                return processor.tokenizer.encode(s, add_special_tokens=False)  # type: ignore[attr-defined]
            except Exception:
                return []

        def _extract_closed_object_values(self, text: str) -> list[str]:
            out: list[str] = []
            i = 0
            while True:
                key_idx = text.find('"object_', i)
                if key_idx == -1:
                    break
                val_colon = text.find(":", key_idx)
                lbrace = text.find("{", val_colon)
                if lbrace == -1:
                    break
                depth = 0
                j = lbrace
                end_found = None
                in_str = False
                esc = False
                while j < len(text):
                    ch = text[j]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    else:
                        if ch == '"':
                            in_str = True
                        elif ch == "{":
                            depth += 1
                        elif ch == "}":
                            if depth > 0:
                                depth -= 1
                                if depth == 0:
                                    end_found = j
                                    break
                    j += 1
                if end_found is None:
                    break
                val_str = text[lbrace : end_found + 1]
                out.append(val_str)
                i = end_found + 1
            return out

        def _update_seen(self, gen_ids: torch.Tensor) -> None:
            text = self._decode(gen_ids)
            if not text:
                return
            for val in self._extract_closed_object_values(text):
                if val in self.seen_values:
                    continue
                self.seen_values.add(val)
                ids = self._encode(val)
                if ids:
                    self.bad_token_sequences.append(ids)

        def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
        ) -> torch.FloatTensor:  # type: ignore[override]
            if input_ids is None or input_ids.size(0) == 0:
                return scores
            seq = input_ids[0]
            gen_ids = seq[self.prompt_len :]
            if gen_ids.numel() == 0:
                return scores
            # Learn newly completed object values
            self._update_seen(gen_ids)
            # Block continuing any exact duplicate of a completed object value
            for bad in self.bad_token_sequences:
                m = len(bad)
                if m == 0:
                    continue
                k = min(m - 1, gen_ids.numel())
                if k <= 0:
                    continue
                # Decode a small tail window and require geometry presence to reduce early interference
                win = min(k, DEDUP_MAX_MATCH_WINDOW_TOKENS)
                try:
                    tail_str = self._decode(gen_ids[-win:])
                except Exception:
                    tail_str = ""
                if not tail_str or not any(
                    marker in tail_str for marker in GEOMETRY_MARKERS
                ):
                    continue
                if k < DEDUP_MIN_PREFIX_TOKENS:
                    continue
                if torch.equal(
                    gen_ids[-k:],
                    torch.tensor(bad[:k], device=gen_ids.device, dtype=gen_ids.dtype),
                ):
                    next_id = bad[k]
                    if 0 <= next_id < scores.size(-1):
                        scores[..., int(next_id)] = -float("inf")
            return scores

    logits_processor = LogitsProcessorList([_DedupObjectValueProcessor(prompt_len)])

    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(TEMPERATURE > 0),
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,  # Prevents repeating object structures without global distribution shift
            logits_processor=logits_processor,
            stopping_criteria=stopping,
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

GEOM_KEYS = ("bbox_2d", "poly", "line")


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
    """Parse JSON with multiple fallback strategies."""
    try:
        return json.loads(s)
    except Exception:
        try:
            # Remove trailing commas before } and ]
            s2 = re.sub(r",\s*\}", "}", re.sub(r",\s*\]", "]", s))
            s2 = s2.replace("'", '"')
            return json.loads(s2)
        except Exception:
            try:
                # Try json_repair for malformed JSON (handles trailing commas, etc.)
                import json_repair

                repaired = json_repair.repair_json(s)
                return json.loads(repaired)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    return None


def parse_prediction(text: str) -> List[Dict[str, Any]]:
    """Parse prediction text into list of objects. Handles standard JSON format."""
    if not text or not text.strip():
        return []

    def _flatten_coords(pts: Any) -> List[float] | None:
        """Flatten nested coordinate arrays like [[x1, y1], [x2, y2]] -> [x1, y1, x2, y2]."""
        if not isinstance(pts, (list, tuple)):
            return None
        if not pts:
            return []
        # Check if it's nested (first element is a list/tuple)
        if isinstance(pts[0], (list, tuple)):
            flattened = []
            for item in pts:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    flattened.extend([float(item[0]), float(item[1])])
                else:
                    return None  # Invalid nested format
            return flattened
        # Already flat, return as-is
        return [float(x) for x in pts]

    def _build_objects_from_dict(obj_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        parsed_local: List[Dict[str, Any]] = []
        for _, val in sorted(
            obj_dict.items(),
            key=lambda kv: kv[0] if isinstance(kv[0], str) else str(kv[0]),
        ):
            if not isinstance(val, dict):
                continue
            desc = val.get("desc", "")

            # Enforce exactly one geometry key (strict validation)
            geom_keys_present = [g for g in GEOM_KEYS if g in val]
            if len(geom_keys_present) != 1:
                # Skip objects with zero or multiple geometry keys
                continue
            gtype = geom_keys_present[0]

            pts = val.get(gtype)
            # Flatten nested coordinate arrays (handles standard format)
            pts = _flatten_coords(pts)
            if pts is None or len(pts) % 2 != 0:
                # Skip objects with incomplete coordinates (e.g., truncated tail)
                continue

            # For line objects, line_points is required and must match coordinate count
            if gtype == "line":
                lp = val.get("line_points")
                if not isinstance(lp, int) or lp < 2:
                    # line_points is required and must be >= 2
                    continue
                expected = 2 * lp
                if len(pts) != expected:
                    # Coordinate count must exactly match 2 * line_points
                    continue

            # Clamp coordinates to norm1000 range [0, 1000]
            try:
                pts = [max(0, min(1000, int(round(float(x))))) for x in pts]
            except Exception:
                continue
            parsed_local.append({"desc": desc, "type": gtype, "points": pts})
        return parsed_local

    # First try: parse as full JSON (balanced root)
    raw = _extract_outer_json(text)
    if raw:
        obj = _json_loads_best_effort(raw)
        if isinstance(obj, dict):
            # Support legacy grouped {"图片_1": {...}} or flat {"object_1": {...}}
            if any(isinstance(k, str) and k.startswith("图片_") for k in obj.keys()):
                try:
                    first_key = sorted(
                        [
                            k
                            for k in obj.keys()
                            if isinstance(k, str) and k.startswith("图片_")
                        ],
                        key=lambda x: int(str(x).split("_")[-1]),
                    )[0]
                    obj = obj.get(first_key) or {}
                except Exception:
                    obj = {}
            if isinstance(obj, dict) and obj:
                parsed = _build_objects_from_dict(obj)
                if parsed:
                    return parsed

    # Fallback: extract per-object dicts from a truncated group body
    # Find the group body start: the '{' after the first occurrence of an object key
    try:
        markers = ["object_", "图片_", "\u56fe\u7247_"]
        grp_idx = -1
        for marker in markers:
            grp_idx = text.find(marker)
            if grp_idx != -1:
                break
        if grp_idx != -1:
            # Find the first '{' after the colon following the group key
            colon_idx = text.find(":", grp_idx)
            brace_idx = text.find("{", colon_idx)
            if brace_idx != -1:
                i = brace_idx + 1
                results: List[Dict[str, Any]] = []
                while True:
                    key_idx = text.find('"object_', i)
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
                    obj_str = text[key_idx : end_found + 1]
                    # Reconstruct as {"object_N": { ... }} to json.loads
                    left_quote = obj_str.find('"')
                    sep_colon = obj_str.find(":", left_quote)
                    obj_key = obj_str[left_quote:sep_colon].strip()
                    obj_val = obj_str[sep_colon + 1 :].strip()
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


def deduplicate_predictions(
    pred_objs: List[Dict[str, Any]], verbose: bool = True
) -> tuple[List[Dict[str, Any]], int]:
    """Remove identical predictions based on desc, type, and points.

    Args:
        pred_objs: List of prediction objects
        verbose: If True, print deduplication stats

    Returns:
        Tuple of (deduplicated list, number of duplicates removed)
    """
    if not pred_objs:
        return pred_objs, 0

    seen: set[tuple[str, str, tuple[int, ...]]] = set()
    deduplicated: List[Dict[str, Any]] = []
    duplicates_removed = 0

    for obj in pred_objs:
        desc = obj.get("desc", "")
        gtype = obj.get("type", "")
        pts = tuple(obj.get("points", []))

        # Create a unique key for this prediction
        key = (desc, gtype, pts)

        if key in seen:
            duplicates_removed += 1
            continue

        seen.add(key)
        deduplicated.append(obj)

    if verbose and duplicates_removed > 0:
        print(
            f"[INFO] Deduplication: removed {duplicates_removed} duplicate(s), "
            f"keeping {len(deduplicated)} unique prediction(s)"
        )

    return deduplicated, duplicates_removed


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


def _create_legend(
    ax_legend, color_map: Dict[str, str], counts: Dict[str, List[int]]
) -> None:
    """Place legend in a dedicated subplot axis instead of overlaying the figure."""
    ax_legend.axis("off")

    legend_elements = []
    active = [label for label, c in counts.items() if c[0] > 0 or c[1] > 0]
    active.sort(key=lambda label: sum(counts[label]), reverse=True)
    for label in active:
        gt_c, pr_c = counts[label]
        legend_label = f"{label} ({gt_c}/{pr_c})"
        legend_elements.append(
            patches.Patch(
                facecolor="none", edgecolor=color_map[label], label=legend_label
            )
        )
    if not legend_elements:
        return

    legend = ax_legend.legend(
        handles=legend_elements,
        loc="center",
        framealpha=0.95,
        fontsize=10,
        title="Object Categories (GT/Pred)",
        title_fontsize=11,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("lightgray")


def _canonicalize_poly(points: List[int | float]) -> List[int]:
    return canonicalize_poly(points)


def _draw_objects(
    ax,
    img: Image.Image,
    objects: List[Dict[str, Any]],
    color_map: Dict[str, str],
    scaled: bool,
) -> None:
    ax.imshow(img)
    ax.axis("off")
    w, h = img.size
    for obj in objects:
        gtype = obj["type"]
        pts = obj["points"]
        desc = obj.get("desc", "")
        pts_px = pts if scaled else inverse_scale(pts, w, h)
        # Canonicalize polygon ordering to avoid self-crossing (use vis_generation logic)
        if gtype == "poly" and len(pts_px) >= 8 and len(pts_px) % 2 == 0:
            pts_list: List[int | float] = [int(p) for p in pts_px]
            pts_px = _canonicalize_poly(pts_list)

        color = color_map.get(desc) or "#000000"
        if gtype == "bbox_2d" and len(pts_px) == 4:
            x1, y1, x2, y2 = pts_px
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2
                )
            )
        elif gtype == "poly" and len(pts_px) >= 8 and len(pts_px) % 2 == 0:
            # Draw polygon with dashed outline similar to vis_generation
            poly_coords = [(pts_px[i], pts_px[i + 1]) for i in range(0, len(pts_px), 2)]
            poly = patches.Polygon(
                poly_coords,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=2,
                linestyle="--",
                alpha=0.9,
            )
            ax.add_patch(poly)
        elif gtype == "line" and len(pts_px) >= 4 and len(pts_px) % 2 == 0:
            xs = pts_px[::2]
            ys = pts_px[1::2]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=3,
                linestyle="-",
                marker="o",
                markersize=3,
                alpha=0.9,
            )


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
    if n_points >= 8 and n_points % 2 == 0:
        return "poly"
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
            objs.append(
                {"desc": desc, "type": gtype, "points": [float(x) for x in pts]}
            )
        return objs

    # Format B: list of per-object dicts [{desc, bbox_2d|poly|line, image_id?}, ...]
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            gtype = next((k for k in ("bbox_2d", "poly", "line") if k in obj), None)
            if gtype is None:
                continue
            pts = obj.get(gtype)
            if not isinstance(pts, (list, tuple)) or len(pts) % 2 != 0:
                continue
            if obj.get("image_id") is not None and int(obj["image_id"]) != image_index:
                continue
            objs.append(
                {
                    "desc": obj.get("desc", ""),
                    "type": gtype,
                    "points": [float(x) for x in pts],
                }
            )
        return objs

    return objs


# ======================
# Main loop
# ======================


def main() -> None:
    # Parse command line arguments
    args = _parse_args()
    global DEVICE
    # Use CLI arg if provided, otherwise use runtime setting
    if args.device_id is not None:
        DEVICE = f"cuda:{args.device_id}"
    # DEVICE already set to "cuda:7" from runtime settings if CLI arg not provided

    os.makedirs(SAVE_DIR, exist_ok=True)
    root = Path(JSONL_PATH).resolve().parent
    # Provide a consistent default for image path resolving in downstream libs
    os.environ.setdefault("ROOT_IMAGE_DIR", str(root))

    # Optionally plot from a dumped JSONL (skip inference)
    if PLOT_FROM_JSONL:
        print(f"[INFO] Plotting from JSONL (no inference): {PLOT_JSONL_PATH}")
        dumped = list(load_records(PLOT_JSONL_PATH))
        count = 0
        total_gt = 0
        total_pred = 0
        total_matched = 0
        total_missing = 0
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

            w, h = img.size
            # Prefer normalized GT if present; otherwise normalize from pixels
            gt_norm = rec.get("gt_norm1000") or rec.get("gt_norm")
            if gt_norm:
                gt_objs = [
                    {
                        "desc": o.get("desc", ""),
                        "type": o.get("type", ""),
                        "points": o.get("points", []),
                    }
                    for o in gt_norm
                    if isinstance(o, dict) and isinstance(o.get("points", []), list)
                ]
                gt_px = [
                    {
                        "desc": o.get("desc", ""),
                        "type": o.get("type", ""),
                        "points": inverse_scale(o.get("points", []), w, h),
                    }
                    for o in gt_objs
                ]
            else:
                gt_px = rec.get("gt", [])
                if normalize_points is not None:
                    gt_objs = [
                        {
                            "desc": o.get("desc", ""),
                            "type": o.get("type", ""),
                            "points": normalize_points(
                                o.get("points", []), w, h, "norm1000"
                            ),
                        }
                        for o in gt_px
                    ]
                else:
                    gt_objs = gt_px

            pred_objs_norm = rec.get("pred_norm1000") or rec.get("pred", [])
            # Deduplicate predictions to remove identical objects (norm space)
            pred_objs_norm, _ = deduplicate_predictions(pred_objs_norm, verbose=True)

            # Convert predictions from norm1000 to pixels for evaluation and drawing
            w, h = img.size
            pred_px = [
                {
                    "desc": o.get("desc", ""),
                    "type": o.get("type", ""),
                    "points": inverse_scale(o.get("points", []), w, h),
                }
                for o in pred_objs_norm
            ]

            # Geometry-based evaluation for this sample
            eval_res = geom_eval.match_geometries(
                gt_objs, pred_objs_norm, iou_threshold=0.5
            )
            print(
                "[EVAL] image="
                f"{count:05d}: GT={eval_res.num_gt}, Pred={eval_res.num_pred}, "
                f"Matched={eval_res.num_matched}, Missing={eval_res.num_missing}"
            )
            total_gt += eval_res.num_gt
            total_pred += eval_res.num_pred
            total_matched += eval_res.num_matched
            total_missing += eval_res.num_missing

            # Build legend color map
            labels = [o.get("desc", "") for o in gt_objs] + [
                o.get("desc", "") for o in pred_objs_norm
            ]
            color_map = _generate_colors(labels)
            legend_counts: Dict[str, List[int]] = {}
            for o in gt_objs:
                key = o.get("desc", "")
                legend_counts.setdefault(key, [0, 0])[0] += 1
            for o in pred_objs_norm:
                key = o.get("desc", "")
                legend_counts.setdefault(key, [0, 0])[1] += 1

            fig, (ax_l, ax_r, ax_legend) = plt.subplots(1, 3, figsize=(18, 6))
            draw_objects(
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
                color_map,
                scaled=True,
            )
            ax_l.set_title("GT")
            draw_objects(ax_r, img, pred_px, color_map, scaled=True)
            ax_r.set_title("Prediction" + ("" if pred_px else " (parse failed)"))
            _create_legend(ax_legend, color_map, legend_counts)

            out_path = Path(SAVE_DIR) / f"vis_{count:05d}.jpg"
            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            count += 1
            if count >= LIMIT:
                break

        if count > 0:
            print(
                "[EVAL] Aggregate over "
                f"{count} samples: GT={total_gt}, Pred={total_pred}, "
                f"Matched={total_matched}, Missing={total_missing}"
            )
        print(f"[DONE] Saved {count} figures to {SAVE_DIR}")
        return

    count = 0
    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_missing = 0
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
        # Use metadata width/height when available; fallback to actual image size
        try:
            meta_w = int(rec.get("width")) if rec.get("width") is not None else None
            meta_h = int(rec.get("height")) if rec.get("height") is not None else None
        except Exception:
            meta_w = meta_h = None
        w_img, h_img = img.size
        w = meta_w or w_img
        h = meta_h or h_img

        raw_text = ""
        clean_text = ""
        try:
            raw_text, clean_text = run_infer_one(img, USER_PROMPT_TEXT)
            # Print raw first for debugging
            try:
                print("\n=== RAW RESPONSE (skip_special_tokens=False) ===")
                print(raw_text)
            except Exception:
                pass
            # Try parsing with clean_text first, fallback to raw_text if needed
            pred_objs = parse_prediction(clean_text)
            if not pred_objs:
                # If clean_text parsing failed, try raw_text (might preserve formatting better)
                print("[WARNING] Parsing with clean_text failed, trying raw_text...")
                pred_objs = parse_prediction(raw_text)
            # Deduplicate predictions to remove identical objects
            pred_objs, _ = deduplicate_predictions(pred_objs, verbose=True)
        except Exception as e:
            print(f"[ERROR] Failed to parse prediction: {e}")
            pred_objs = []

        # Convert predictions from norm1000 to pixels for evaluation and drawing
        pred_px = [
            {
                "desc": o.get("desc", ""),
                "type": o.get("type", ""),
                "points": inverse_scale(o.get("points", []), w, h),
            }
            for o in pred_objs
        ]

        # Geometry-based evaluation for this sample
        gt_px = [
            {
                "type": o["type"],
                "points": [int(round(p)) for p in o["points"]],
                "desc": o.get("desc", ""),
            }
            for o in gt
        ]
        gt_norm = []
        if normalize_points is not None and w and h:
            for o in gt_px:
                pts_norm = normalize_points(o.get("points", []), w, h, "norm1000")
                gt_norm.append(
                    {"type": o["type"], "points": pts_norm, "desc": o.get("desc", "")}
                )
        else:
            gt_norm = gt_px

        eval_res = geom_eval.match_geometries(gt_norm, pred_objs, iou_threshold=0.5)
        print(
            "[EVAL] image="
            f"{count:05d}: GT={eval_res.num_gt}, Pred={eval_res.num_pred}, "
            f"Matched={eval_res.num_matched}, Missing={eval_res.num_missing}"
        )
        total_gt += eval_res.num_gt
        total_pred += eval_res.num_pred
        total_matched += eval_res.num_matched
        total_missing += eval_res.num_missing

        # Accumulate JSONL entries for later plotting
        if SAVE_JSONL:
            dumped_records.append(
                {
                    "image_path": images[0],  # keep original relative path
                    "gt": gt_px,
                    "gt_norm1000": gt_norm,
                    "pred": pred_objs,
                    "width": w,
                    "height": h,
                    "raw_text": raw_text,
                    "clean_text": clean_text,
                }
            )

        # Build legend color map using labels appearing in this figure
        labels = [o.get("desc", "") for o in gt_px] + [
            o.get("desc", "") for o in pred_px
        ]
        color_map = _generate_colors(labels)

        # Track counts for legend
        counts: Dict[str, List[int]] = {}
        for o in gt_px:
            key = o.get("desc", "")
            counts.setdefault(key, [0, 0])[0] += 1
        for o in pred_px:
            key = o.get("desc", "")
            counts.setdefault(key, [0, 0])[1] += 1

        # Plot 1×3 (GT | Pred | Legend)
        fig, (ax_l, ax_r, ax_legend) = plt.subplots(1, 3, figsize=(18, 6))
        # GT is already in pixels
        draw_objects(
            ax_l,
            img,
            gt_px,
            color_map,
            scaled=True,
        )
        ax_l.set_title("GT")

        draw_objects(ax_r, img, pred_px, color_map, scaled=True)
        ax_r.set_title("Prediction" + ("" if pred_px else " (parse failed)"))

        _create_legend(ax_legend, color_map, counts)

        out_path = Path(SAVE_DIR) / f"vis_{count:05d}.jpg"
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        count += 1
        if count >= LIMIT:
            break

    # Write out dump JSONL if requested
    if SAVE_JSONL and dumped_records:
        write_jsonl(dumped_records, DUMP_JSONL_PATH)
        print(f"[INFO] Dumped GT vs Pred JSONL: {DUMP_JSONL_PATH}")

    if count > 0:
        print(
            "[EVAL] Aggregate over "
            f"{count} samples: GT={total_gt}, Pred={total_pred}, "
            f"Matched={total_matched}, Missing={total_missing}"
        )

    print(f"[DONE] Saved {count} figures to {SAVE_DIR}")


if __name__ == "__main__":
    main()
