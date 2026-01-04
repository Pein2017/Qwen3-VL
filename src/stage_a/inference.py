#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core inference engine for Stage-A per-image summarization.

This module discovers groups from mission-based directories, runs batched
inference on images, and outputs grouped JSONL records with strict validation.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, cast

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BatchEncoding,
    GenerationConfig,
    PreTrainedTokenizerBase,
    Qwen3VLForConditionalGeneration,
)

from data_conversion.utils.exif_utils import apply_exif_orientation

from ..utils import get_logger
from .prompts import SUMMARY_SYSTEM_PROMPT, build_system_prompt, build_user_prompt
from src.prompts.summary_profiles import DEFAULT_SUMMARY_PROFILE_RUNTIME


def _build_generation_config(gen_config: dict[str, object]) -> GenerationConfig:
    return GenerationConfig(**gen_config)

# Import distributed helpers from stage_b (reusable lightweight module)
try:
    from ..stage_b.distributed import (
        barrier,
        broadcast_object,
        get_local_rank,
        get_rank,
        get_world_size,
        init_distributed,
        is_main_process,
    )
except ImportError:
    # Fallback if stage_b is not available (shouldn't happen in normal usage)
    def get_world_size() -> int:
        import os

        value = os.environ.get("WORLD_SIZE")
        return int(value) if value is not None else 1

    def get_rank() -> int:
        import os

        value = os.environ.get("RANK")
        return int(value) if value is not None else 0

    def get_local_rank() -> int:
        import os

        value = os.environ.get("LOCAL_RANK")
        return int(value) if value is not None else 0

    def is_main_process() -> bool:
        return get_rank() == 0

    def init_distributed(*, timeout_seconds: int = 1800) -> None:
        import torch.distributed as dist

        if not dist.is_available() or dist.is_initialized():
            return
        world_size = get_world_size()
        if world_size <= 1:
            return
        from datetime import timedelta

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=int(timeout_seconds)),
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(get_local_rank())

    def barrier() -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and get_world_size() > 1:
            dist.barrier()

    def broadcast_object(obj: object | None, *, src: int = 0) -> object:
        return obj


logger = get_logger(__name__)


class ProcessorProtocol(Protocol):
    tokenizer: PreTrainedTokenizerBase

    def apply_chat_template(self, *args: object, **kwargs: object) -> str: ...

    def __call__(self, *args: object, **kwargs: object) -> BatchEncoding: ...

    def batch_decode(self, *args: object, **kwargs: object) -> list[str]: ...


def _safe_pbar_update(pbar: object, delta: int) -> None:
    update = getattr(pbar, "update", None)
    if callable(update):
        update(delta)

# Supported image extensions
SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}

# Group ID extraction regex
GROUP_REGEX = re.compile(r"^(QC-[A-Za-z]+-[0-9]{8}-[0-9]+)")

# Label directory to label mapping
LABEL_DIR_MAP = {
    "审核通过": "pass",
    "审核不通过": "fail",
}


@dataclass
class GroupInfo:
    """Information about a discovered group."""

    paths: list[Path]
    label: str
    mission: str
    group_id: str


def _natural_key(s: str) -> list[object]:
    """Natural sort key for filenames with numbers."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _maybe_parse_json_object(text: str) -> dict[str, object] | None:
    """Attempt to parse a JSON object from a string.
    Returns dict if text is a JSON object; otherwise None.
    """
    t = text.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return None
    try:
        obj = json.loads(t)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _is_summary_json(obj: dict[str, object]) -> bool:
    """Heuristic: identify summary JSON schema."""
    if not isinstance(obj, dict):
        return False
    required = {"dataset", "统计"}
    return required.issubset(obj.keys())


def _format_summary_json(obj: dict[str, object]) -> str:
    """Normalize summary JSON objects to a single-line string.

    - Keeps only the canonical top-level keys used by the summary contract.
    - Uses the canonical separators (", ", ": ") to match prompt/contract.
    """
    ordered: dict[str, object] = {}
    for key in ("dataset", "统计", "备注", "分组统计", "异常"):
        if key in obj:
            ordered[key] = obj[key]
    return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))


def _extract_summary_json_line(text: str) -> str | None:
    """Extract summary-statistics JSON from model output text.

    Stage-A summary-mode models may emit:
    - Single-line JSON: {"dataset": "...", "统计": [...]}
    - Two-line output: "<DOMAIN=...>, <TASK=...>" + JSON on the next line

    This helper finds the JSON object and returns a normalized single-line JSON
    string, or None when the input does not contain a valid summary JSON.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None

    obj = _maybe_parse_json_object(stripped)
    if obj is not None and _is_summary_json(obj):
        return _format_summary_json(obj)

    # Multi-line outputs: search per-line (reverse to favor the JSON tail).
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    for line in reversed(lines):
        candidate = _maybe_parse_json_object(line)
        if candidate is not None and _is_summary_json(candidate):
            return _format_summary_json(candidate)

    # Fallback: extract the first {...} block from the full text.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate_text = stripped[start : end + 1].strip()
        candidate = _maybe_parse_json_object(candidate_text)
        if candidate is not None and _is_summary_json(candidate):
            return _format_summary_json(candidate)

    return None


def sanitize_single_image_summary(text: str) -> str:
    """Extract and sanitize the summary for a single image.

    - If `text` is a JSON object containing keys like image_1/图片_1, keep only the first image key
    - Otherwise use the raw `text`
    - Do not alter content beyond trimming outer whitespace
    """
    summary_text = text.strip()

    if summary_text.startswith("无关图片"):
        return "无关图片"

    extracted = _extract_summary_json_line(summary_text)
    if extracted is not None:
        return extracted

    obj = _maybe_parse_json_object(summary_text)
    if obj is not None and _is_summary_json(obj):
        return _format_summary_json(obj)
    if obj is not None:
        if "统计" in obj:
            return summary_text
        extracted: str | None = None
        for key in ("image_1", "图片_1"):
            val = obj.get(key)
            if isinstance(val, str):
                extracted = val.strip()
                break
        if extracted is None:
            # Fallback: join all string values if expected key missing
            collected: list[str] = []
            for value in obj.values():
                if isinstance(value, str):
                    collected.append(value.strip())
            if collected:
                extracted = "，".join(collected)
        if extracted is not None:
            summary_text = extracted

    # Keep raw content (no count-marker removal or dedupe). Only strip surrounding whitespace.
    summary_text = summary_text.strip()
    return summary_text


_GROUP_PREFIX_RE = re.compile(r"^(组\d+[:：])+")
_GROUP_PREFIX_OF_RE = re.compile(r"^组(\d+)的")

_BBU_CATEGORIES = {
    "BBU设备",
    "挡风板",
    "光纤",
    "电线",
    "标签",
    "BBU安装螺丝",
    "机柜处接地螺丝",
    "地排处接地螺丝",
    "ODF端光纤插头",
    "BBU端光纤插头",
}

_RRU_CATEGORIES = {
    "RRU设备",
    "紧固件",
    "RRU接地端",
    "地排接地端螺丝",
    "尾纤",
    "接地线",
    "标签",
    "站点距离",
}


def sanitize_summary_by_dataset(text: str, dataset: str) -> str:
    """Normalize summary text with dataset-specific constraints.

    For RRU, drop non-domain items (e.g., BBU-only objects). If nothing remains,
    return "无关图片". Preserve label/station OCR content.
    """
    summary_text = text.strip()
    if not summary_text:
        return summary_text

    extracted = _extract_summary_json_line(summary_text)
    if extracted is not None:
        summary_text = extracted

    obj = _maybe_parse_json_object(summary_text)
    if obj is not None and _is_summary_json(obj):
        return _format_summary_json(obj)

    obj = _maybe_parse_json_object(summary_text)
    if obj is not None and "统计" in obj:
        allowed = _RRU_CATEGORIES if dataset.lower() == "rru" else _BBU_CATEGORIES
        entries = obj.get("统计")
        if isinstance(entries, list):
            filtered = [
                entry
                for entry in entries
                if isinstance(entry, dict) and entry.get("类别") in allowed
            ]
        else:
            filtered = []

        if not filtered:
            return "无关图片"

        obj["统计"] = filtered
        if dataset.lower() == "rru":
            obj.pop("备注", None)
        else:
            obj.pop("分组统计", None)
        obj["dataset"] = dataset.upper()
        return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))

    parts = [p.strip() for p in summary_text.split("，") if p.strip()]
    if not parts:
        return summary_text

    def _normalize_irrelevant_only(items: list[str]) -> list[str]:
        cleaned = []
        for item in items:
            if item.startswith("无关图片"):
                continue
            cleaned.append(item)
        return cleaned

    if dataset.lower() != "rru":
        cleaned = _normalize_irrelevant_only(parts)
        return "无关图片" if not cleaned else "，".join(cleaned)

    allowed_prefixes = (
        "标签/",
        "站点距离/",
        "RRU设备",
        "RRU接地端",
        "地排接地端螺丝",
        "紧固件",
        "尾纤",
        "接地线",
    )

    cleaned: list[str] = []
    for item in parts:
        if item.startswith("无关图片"):
            continue
        normalized = _GROUP_PREFIX_OF_RE.sub(r"组\1:", item.replace("：", ":")).strip()
        if not normalized:
            continue
        group_prefix = ""
        match = _GROUP_PREFIX_RE.match(normalized)
        core = normalized
        if match:
            group_prefix = match.group(0)
            core = normalized[len(group_prefix) :]

        if not core.startswith("标签/"):
            core = re.sub(r"备注[:：].*$", "", core).strip()
            core = re.sub(r"[，,、;；]+$", "", core).strip()
            if not core:
                continue

        if core.startswith("RRU设备/"):
            normalized_core = "RRU设备"
            cleaned.append(
                f"{group_prefix}{normalized_core}" if group_prefix else normalized_core
            )
            continue

        if core.startswith(allowed_prefixes):
            cleaned.append(f"{group_prefix}{core}" if group_prefix else core)
            continue

    if not cleaned:
        return "无关图片"
    return "，".join(cleaned)




def _trim_trailing_eos_pad(
    gen_ids: torch.Tensor, eos_id: int | None, pad_id: int | None
) -> torch.Tensor:
    """Trim trailing EOS/PAD token ids from a [1, T] tensor and return sliced view."""
    if gen_ids.ndim != 2 or gen_ids.size(0) != 1:
        return gen_ids
    t = gen_ids.size(1)
    end = t
    while end > 0:
        last = int(gen_ids[0, end - 1].item())
        if (eos_id is not None and last == eos_id) or (
            pad_id is not None and last == pad_id
        ):
            end -= 1
            continue
        break
    if end == t:
        return gen_ids
    return gen_ids[:, :end]


def _extract_group_id(path: Path) -> str:
    """Extract group ID from filename or parent directory.

    Args:
        path: Image file path

    Returns:
        Group ID string
    """
    # Try regex match on filename stem
    m = GROUP_REGEX.match(path.stem)
    if m:
        return m.group(1)
    # Fallback: use immediate parent directory name
    return path.parent.name


def discover_groups(root: Path, mission: str | None = None) -> list[GroupInfo]:
    """Discover and group images from mission-based directory structure.

    Expected structure:
        <root>/<mission>/{审核通过|审核不通过}/<group_id>/*.{jpg,jpeg,png}

    Args:
        root: Root directory path
        mission: Mission name to filter (processes only this mission)

    Returns:
        List of GroupInfo objects (duplicates allowed when the same group_id
        appears under different label directories)

    Raises:
        FileNotFoundError: If root or mission directory doesn't exist
    """
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    # Discover mission directories
    if mission:
        mission_dirs = [root / mission]
        if not mission_dirs[0].exists():
            raise FileNotFoundError(f"Mission directory not found: {mission_dirs[0]}")
    else:
        # Auto-discover all mission subdirectories
        mission_dirs = [d for d in root.iterdir() if d.is_dir()]

    groups: list[GroupInfo] = []

    for mission_dir in sorted(mission_dirs, key=lambda p: p.name):
        mission_name = mission_dir.name

        # Scan label directories: process "审核不通过" first, then "审核通过"
        # Order: fail labels first, then pass labels
        label_order = ["审核不通过", "审核通过"]
        for label_dir_name in label_order:
            if label_dir_name not in LABEL_DIR_MAP:
                continue
            label_str = LABEL_DIR_MAP[label_dir_name]
            label_dir = mission_dir / label_dir_name
            if not label_dir.exists() or not label_dir.is_dir():
                continue

            # Scan group directories
            group_dirs: Sequence[Path] = sorted(
                (d for d in label_dir.iterdir() if d.is_dir()), key=lambda p: p.name
            )
            for group_dir in group_dirs:
                # Discover images in group
                image_paths: list[Path] = []
                for img_path in group_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXT:
                        image_paths.append(img_path)

                if not image_paths:
                    logger.warning(f"No images found in group: {group_dir}")
                    continue

                # Natural sort images
                image_paths.sort(key=lambda p: _natural_key(p.name))

                # Extract group ID
                group_id = _extract_group_id(image_paths[0])

                # Store group info (allow duplicates across label dirs)
                groups.append(
                    GroupInfo(
                        paths=image_paths,
                        label=label_str,
                        mission=mission_name,
                        group_id=group_id,
                    )
                )
        # Warn on unexpected subdirectories to avoid silent skips
        for child in mission_dir.iterdir():
            if child.is_dir() and child.name not in LABEL_DIR_MAP:
                logger.warning(f"Skipping unknown label directory: {child.name}")

    return groups


T = TypeVar("T")


def _sample_subset(
    items: Sequence[T], target: int | None, rng: random.Random
) -> list[T]:
    """Return up to `target` random items from the list respecting deterministic RNG."""
    if target is None or target >= len(items):
        return list(items)
    return rng.sample(items, target)


def _sample_groups(
    groups: list[GroupInfo],
    pass_target: int | None,
    fail_target: int | None,
    seed: int,
) -> tuple[list[GroupInfo], dict[str, int]]:
    """Seeded sampling of pass/fail groups, preserving original order.

    Note: When both targets are None, sampling is disabled and all groups are returned.
    """
    sampling_enabled = pass_target is not None or fail_target is not None

    pass_indices = [i for i, g in enumerate(groups) if g.label == "pass"]
    fail_indices = [i for i, g in enumerate(groups) if g.label == "fail"]
    rng_pass = random.Random(seed)
    rng_fail = random.Random(seed + 1)

    sampled_pass_indices = _sample_subset(pass_indices, pass_target, rng_pass)
    sampled_fail_indices = _sample_subset(fail_indices, fail_target, rng_fail)

    selected_indices: set[int] = set(sampled_pass_indices + sampled_fail_indices)
    if not sampling_enabled:
        selected_indices = set(range(len(groups)))

    sampled = [g for i, g in enumerate(groups) if i in selected_indices]

    stats = {
        "pass_total": len(pass_indices),
        "pass_selected": len(sampled_pass_indices),
        "fail_total": len(fail_indices),
        "fail_selected": len(sampled_fail_indices),
    }
    return sampled, stats


def load_model_processor(
    checkpoint: str, device: str, max_pixels: int = 786432
) -> tuple[Qwen3VLForConditionalGeneration, ProcessorProtocol]:
    """Load Qwen3-VL model and processor.

    Args:
        checkpoint: HuggingFace checkpoint path
        device: Device string (cuda:N or cpu)
        max_pixels: Maximum pixels for image resizing (default: 786432 for efficiency)

    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading model from {checkpoint}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2"
        if torch.cuda.is_available()
        else "eager",
        trust_remote_code=True,
    )
    cast(torch.nn.Module, model).to(device)
    model.eval()

    processor = cast(
        ProcessorProtocol,
        AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True),
    )
    # IMPORTANT: For decoder-only generation with variable-length prompts (vision tokens vary by image),
    # batched inference MUST use left-padding. Right-padding puts PAD tokens at the end of shorter prompts
    # and can destabilize generation when mixed-size images are batched together (common in per_image mode).
    try:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "left"
            tok.truncation_side = "left"
            logger.info(
                "Tokenizer padding_side=%s truncation_side=%s",
                getattr(tok, "padding_side", None),
                getattr(tok, "truncation_side", None),
            )
    except Exception:
        logger.warning(
            "Failed to set tokenizer padding/truncation side", exc_info=False
        )

    # Configure max_pixels for compute efficiency
    # Lower values = faster inference but lower image quality
    # 786432 = 1024×768 equivalent (good balance)
    # Default Qwen3-VL is 12845056 (very high resolution)
    # Enforce pixel budget on the image processor (Qwen2VLImageProcessorFast)
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        try:
            # Qwen fast processor expects pixel budgets via the size dict
            #   size["shortest_edge"] = min_pixels, size["longest_edge"] = max_pixels
            if not hasattr(ip, "size") or not isinstance(ip.size, dict):
                ip.size = {}

            # Preserve any explicit min_pixels if provided; otherwise fall back to existing size or default 56*56
            min_pix = getattr(ip, "min_pixels", None)
            if min_pix is None:
                min_pix = ip.size.get("shortest_edge", 56 * 56)

            ip.size["shortest_edge"] = int(min_pix)
            ip.size["longest_edge"] = int(max_pixels)

            # Mirror onto convenience attributes when present
            try:
                ip.min_pixels = int(min_pix)
            except Exception:
                pass
            try:
                ip.max_pixels = int(max_pixels)
            except Exception:
                pass

            logger.info(
                "Set pixel budget: min_pixels=%s max_pixels=%s (size=%s)",
                ip.size.get("shortest_edge"),
                ip.size.get("longest_edge"),
                ip.size,
            )
        except Exception:
            logger.warning(
                "Failed to set pixel budget on image_processor; continuing with defaults",
                exc_info=False,
            )
    else:
        logger.warning("Could not set max_pixels (processor may not support it)")
    # Log critical image processor settings for alignment diagnostics
    try:
        ip = getattr(processor, "image_processor", None)
        logger.info(
            "ImageProcessor settings: do_resize=%s, patch_size=%s, merge_size=%s, min_pixels=%s, max_pixels=%s, size=%s",
            getattr(ip, "do_resize", None),
            getattr(ip, "patch_size", None),
            getattr(ip, "merge_size", None),
            getattr(ip, "min_pixels", None),
            getattr(ip, "max_pixels", None),
            getattr(ip, "size", None),
        )
    except Exception:
        pass

    logger.info(f"Model loaded on {device}")
    return model, processor


def infer_one_image(
    model: Qwen3VLForConditionalGeneration,
    processor: ProcessorProtocol,
    image: Image.Image,
    user_text: str,
    gen_config: dict[str, object],
    verify: bool = False,
    system_prompt: str | None = None,
) -> tuple[str, str]:
    """Run inference on a single image.

    Args:
        model: Qwen3-VL model
        processor: ProcessorProtocol
        image: PIL Image
        user_text: User prompt text
        gen_config: Generation config dict
        verify: Whether to verify inputs
        system_prompt: Optional system prompt (defaults to SUMMARY_SYSTEM_PROMPT)

    Returns:
        Tuple of (raw_text, clean_text)

    Raises:
        ValueError: If clean_text is empty after stripping
    """
    # Use provided system prompt or fallback to default
    sys_prompt = system_prompt if system_prompt is not None else SUMMARY_SYSTEM_PROMPT

    # Build messages with typed content
    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # Apply chat template
    text = processor.apply_chat_template(  # type: ignore[attr-defined]
        messages, add_generation_prompt=True, tokenize=False
    )

    # Encode
    # Respect max_pixels by passing images_kwargs explicitly (Qwen2VLImageProcessorFast)
    _img_kwargs = {}
    try:
        ip = getattr(processor, "image_processor", None)
        if ip is not None:
            if isinstance(getattr(ip, "size", None), dict):
                min_pix = ip.size.get("shortest_edge")
                max_pix = ip.size.get("longest_edge")
            else:
                min_pix = getattr(ip, "min_pixels", None)
                max_pix = getattr(ip, "max_pixels", None)
            if min_pix is not None:
                _img_kwargs["min_pixels"] = int(min_pix)
            if max_pix is not None:
                _img_kwargs["max_pixels"] = int(max_pix)
    except Exception:
        pass
    inputs = processor(  # type: ignore[operator]
        images=[image], text=[text], return_tensors="pt", images_kwargs=_img_kwargs
    )
    inputs = cast(BatchEncoding, inputs)
    if verify:
        try:
            import hashlib

            buf = image.tobytes()
            sha = hashlib.sha256(buf).hexdigest()[:16]
            grid = inputs.get("image_grid_thw")
            merge = getattr(
                getattr(processor, "image_processor", None), "merge_size", 2
            )
            expected_tokens = (
                int((grid[0].prod() // (merge * merge)).item())
                if grid is not None
                else -1
            )
            image_token_id = getattr(processor, "image_token_id", None) or getattr(
                processor.tokenizer,  # type: ignore[attr-defined]
                "image_token_id",
                None,
            )
            text_token_count = (
                int((inputs["input_ids"][0] == image_token_id).sum().item())
                if image_token_id is not None
                else -1
            )
            logger.info(
                "[verify] size=%sx%s sha256=%s grid_thw=%s expected_image_tokens=%s text_image_tokens=%s",
                image.width,
                image.height,
                sha,
                tuple(grid[0].tolist()) if grid is not None else None,
                expected_tokens,
                text_token_count,
            )
        except Exception:
            logger.warning("[verify] logging failed", exc_info=False)
    inputs = inputs.to(model.device)
    # Debug: verify grid/token alignment
    try:
        grid = inputs.get("image_grid_thw")
        if grid is not None:
            merge = getattr(
                getattr(processor, "image_processor", None), "merge_size", 2
            )
            expected_tokens = int((grid[0].prod() // (merge * merge)).item())
            image_token_id = getattr(processor, "image_token_id", None)
            if image_token_id is None:
                image_token_id = getattr(processor.tokenizer, "image_token_id", None)  # type: ignore[attr-defined]
            text_token_count = -1
            try:
                text_token_count = (
                    int((inputs["input_ids"][0] == image_token_id).sum().item())
                    if image_token_id is not None
                    else -1
                )
            except Exception:
                pass
            logger.debug(
                "grid_thw=%s expected_image_tokens=%s text_image_tokens=%s",
                tuple(grid[0].tolist()),
                expected_tokens,
                text_token_count,
            )
    except Exception:
        pass

    generation_config = _build_generation_config(gen_config)

    # Generate
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            use_cache=True,
            generation_config=generation_config,
            pad_token_id=processor.tokenizer.pad_token_id,  # type: ignore[attr-defined]
        )

    # Decode
    start = inputs["input_ids"].shape[-1]
    gen_only = gen[:, start:]
    # Trim trailing EOS/PAD tokens only (preserve other content)
    gen_only = _trim_trailing_eos_pad(
        gen_only,
        eos_id=getattr(processor.tokenizer, "eos_token_id", None),  # type: ignore[attr-defined]
        pad_id=getattr(processor.tokenizer, "pad_token_id", None),  # type: ignore[attr-defined]
    )

    try:
        raw_text = processor.batch_decode(  # type: ignore[attr-defined]
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
    except Exception:
        raw_text = processor.tokenizer.batch_decode(  # type: ignore[attr-defined]
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

    # Keep raw content; only strip outer whitespace
    raw_text = raw_text.strip()

    try:
        clean_text = processor.batch_decode(  # type: ignore[attr-defined]
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
    except Exception:
        clean_text = processor.tokenizer.batch_decode(  # type: ignore[attr-defined]
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

    clean_text = clean_text.strip()

    # Validation: fail-fast on empty summary
    if not clean_text:
        raise ValueError(
            "Empty summary generated (clean_text is empty after stripping)"
        )

    # Sanitize single-image summary to keep only primary image content and remove redundancy
    clean_text = sanitize_single_image_summary(clean_text)
    return raw_text, clean_text


def infer_batch(
    model: Qwen3VLForConditionalGeneration,
    processor: ProcessorProtocol,
    images: list[Image.Image],
    user_text: str,
    gen_config: dict[str, object],
    verify: bool = False,
    system_prompt: str | None = None,
) -> list[tuple[str, str]]:
    """Run batched inference on multiple images.

    Args:
        model: Qwen3-VL model
        processor: ProcessorProtocol
        images: List of PIL Images
        user_text: User prompt text (same for all images)
        gen_config: Generation config dict

    Returns:
        List of (raw_text, clean_text) tuples

    Raises:
        ValueError: If any clean_text is empty after stripping
    """
    # Use provided system prompt or fallback to default
    sys_prompt = system_prompt if system_prompt is not None else SUMMARY_SYSTEM_PROMPT

    # Build messages for each image
    messages_list = [
        [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        for img in images
    ]

    # Apply chat template to each
    texts = [
        processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)  # type: ignore[attr-defined]
        for msgs in messages_list
    ]

    # Batch encode with padding
    _img_kwargs = {}
    try:
        ip = getattr(processor, "image_processor", None)
        if ip is not None:
            if isinstance(getattr(ip, "size", None), dict):
                min_pix = ip.size.get("shortest_edge")
                max_pix = ip.size.get("longest_edge")
            else:
                min_pix = getattr(ip, "min_pixels", None)
                max_pix = getattr(ip, "max_pixels", None)
            if min_pix is not None:
                _img_kwargs["min_pixels"] = int(min_pix)
            if max_pix is not None:
                _img_kwargs["max_pixels"] = int(max_pix)
    except Exception:
        pass
    inputs = processor(  # type: ignore[operator]
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
        images_kwargs=_img_kwargs,
    )
    inputs = cast(BatchEncoding, inputs)
    if verify and len(images) > 0:
        try:
            import hashlib

            grid = inputs.get("image_grid_thw")
            merge = getattr(
                getattr(processor, "image_processor", None), "merge_size", 2
            )
            exp = (
                int((grid[0].prod() // (merge * merge)).item())
                if grid is not None
                else -1
            )
            image_token_id = getattr(processor, "image_token_id", None) or getattr(
                processor.tokenizer,  # type: ignore[attr-defined]
                "image_token_id",
                None,
            )
            txt = (
                int((inputs["input_ids"][0] == image_token_id).sum().item())
                if image_token_id is not None
                else -1
            )
            sha0 = hashlib.sha256(images[0].tobytes()).hexdigest()[:16]
            logger.info(
                "[verify-batch] first.size=%sx%s sha256=%s grid_thw=%s expected_tokens=%s text_tokens=%s",
                images[0].width,
                images[0].height,
                sha0,
                tuple(grid[0].tolist()) if grid is not None else None,
                exp,
                txt,
            )
        except Exception:
            logger.warning("[verify-batch] logging failed", exc_info=False)
    inputs = inputs.to(model.device)
    # Debug: verify first-sample grid/token alignment
    try:
        if len(images) > 0:
            grid = inputs.get("image_grid_thw")
            if grid is not None:
                merge = getattr(
                    getattr(processor, "image_processor", None), "merge_size", 2
                )
                expected_tokens = int((grid[0].prod() // (merge * merge)).item())
                image_token_id = getattr(processor, "image_token_id", None)
                if image_token_id is None:
                    image_token_id = getattr(
                        processor.tokenizer,  # type: ignore[attr-defined]
                        "image_token_id",
                        None,
                    )
                text_token_count = -1
                try:
                    text_token_count = (
                        int((inputs["input_ids"][0] == image_token_id).sum().item())
                        if image_token_id is not None
                        else -1
                    )
                except Exception:
                    pass
                logger.debug(
                    "[batch] grid_thw=%s expected_image_tokens=%s text_image_tokens=%s",
                    tuple(grid[0].tolist()),
                    expected_tokens,
                    text_token_count,
                )
    except Exception:
        pass

    generation_config = _build_generation_config(gen_config)

    # Generate for batch
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            use_cache=True,
            generation_config=generation_config,
            pad_token_id=processor.tokenizer.pad_token_id,  # type: ignore[attr-defined]
        )

    # Decode each output
    start = inputs["input_ids"].shape[-1]
    outputs: list[tuple[str, str]] = []

    for i in range(len(images)):
        gen_only = gen[i : i + 1, start:]
        # Trim trailing EOS/PAD tokens only (preserve other content)
        gen_only = _trim_trailing_eos_pad(
            gen_only,
            eos_id=getattr(processor.tokenizer, "eos_token_id", None),  # type: ignore[attr-defined]
            pad_id=getattr(processor.tokenizer, "pad_token_id", None),  # type: ignore[attr-defined]
        )

        try:
            raw_text = processor.batch_decode(  # type: ignore[attr-defined]
                gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]
        except Exception:
            raw_text = processor.tokenizer.batch_decode(  # type: ignore[attr-defined]
                gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]

        # Keep raw content; only strip outer whitespace
        raw_text = raw_text.strip()

        try:
            clean_text = processor.batch_decode(  # type: ignore[attr-defined]
                gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]
        except Exception:
            clean_text = processor.tokenizer.batch_decode(  # type: ignore[attr-defined]
                gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        clean_text = clean_text.strip()

        # Validation: fail-fast on empty summary
        if not clean_text:
            raise ValueError(
                f"Empty summary generated for image {i} in batch "
                "(clean_text is empty after stripping)"
            )

        # Sanitize single-image summary to keep only primary image content and remove redundancy
        clean_text = sanitize_single_image_summary(clean_text)
        outputs.append((raw_text, clean_text))

    return outputs


def process_group(
    group_id: str,
    group_info: GroupInfo,
    model: Qwen3VLForConditionalGeneration,
    processor: ProcessorProtocol,
    mission: str | None,
    dataset: str = "bbu",
    prompt_profile: str = DEFAULT_SUMMARY_PROFILE_RUNTIME,
    gen_config: dict[str, object] | None = None,
    batch_size: int = 8,
    verify: bool = False,
) -> dict[str, object]:
    """Process a single group with batched inference.

    Args:
        group_id: Group ID
        group_info: GroupInfo with paths and metadata
        model: Qwen3-VL model
        processor: ProcessorProtocol
        mission: Mission name (for prompt building)
        dataset: Dataset type ("bbu" or "rru")
        gen_config: Generation config dict
        batch_size: Batch size for inference (1 = sequential)

    Returns:
        JSONL record dict with all required fields

    Raises:
        ValueError: If validation fails (empty summary or per-image index mismatch)
    """
    # Build mission-dependent prompts
    user_text = build_user_prompt(
        mission,
        dataset=dataset,
        profile_name=prompt_profile,
    )
    system_text = build_system_prompt(
        mission,
        dataset=dataset,
        profile_name=prompt_profile,
    )

    # Default generation config if not provided
    if gen_config is None:
        gen_config = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }

    # Load images
    images: list[Image.Image] = []
    for path in group_info.paths:
        try:
            img = apply_exif_orientation(Image.open(path))
            images.append(img)
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {path}") from e

    # Run inference with batching
    raw_texts: list[str] = []
    clean_texts: list[str] = []

    num_images = len(images)
    for i in range(0, num_images, batch_size):
        chunk = images[i : i + batch_size]

        if batch_size == 1:
            # Sequential fallback
            raw, clean = infer_one_image(
                model,
                processor,
                chunk[0],
                user_text,
                gen_config,
                verify=verify,
                system_prompt=system_text,
            )
            raw_texts.append(raw)
            clean_texts.append(clean)
        else:
            # Batched inference
            chunk_outputs = infer_batch(
                model,
                processor,
                chunk,
                user_text,
                gen_config,
                verify=verify,
                system_prompt=system_text,
            )
            for raw, clean in chunk_outputs:
                raw_texts.append(raw)
                clean_texts.append(clean)

    # Build per_image mapping with deterministic image_i keys
    per_image: dict[str, str] = {}
    for idx, clean_text in enumerate(clean_texts, start=1):
        key = f"image_{idx}"
        sanitized = sanitize_summary_by_dataset(clean_text, dataset)
        per_image[key] = sanitized

    # Strict validation: coverage check
    if len(per_image) != num_images:
        raise ValueError(
            f"per_image coverage mismatch for group {group_id}: "
            f"{len(per_image)} keys vs {num_images} images"
        )

    # Verify all keys are present
    for idx in range(1, num_images + 1):
        key = f"image_{idx}"
        if key not in per_image:
            raise ValueError(f"Missing image_{idx} in per_image for group {group_id}")

    # Build record
    # Flattened record: keep only essential fields; drop raw/clean arrays to reduce redundancy
    record: dict[str, object] = {
        "group_id": group_id,
        "mission": group_info.mission,
        "label": group_info.label,
        "images": [p.name for p in group_info.paths],
        "per_image": per_image,
    }

    return record


def run_stage_a_inference(
    checkpoint: str,
    input_dir: str,
    output_dir: str,
    mission: str,
    dataset: str = "bbu",
    prompt_profile: str = DEFAULT_SUMMARY_PROFILE_RUNTIME,
    device: str = "cuda:0",
    gen_params: dict[str, object] | None = None,
    batch_size: int = 8,
    max_pixels: int = 786432,
    verify_inputs: bool = False,
    pass_group_number: int | None = None,
    fail_group_number: int | None = None,
    sample_seed: int = 42,
    sharding_mode: str = "per_group",
    keep_intermediate_outputs: bool = False,
) -> None:
    """Run Stage-A inference on mission-based directory.

    Args:
        checkpoint: HuggingFace checkpoint path
        input_dir: Root directory with mission/label/group structure
        output_dir: Output directory for JSONL files
        mission: Mission name to process
        prompt_profile: Summary prompt profile (e.g., summary_runtime)
        device: Device string (cuda:N or cpu) - overridden by per-rank device in distributed mode
        gen_params: Generation parameters dict
        batch_size: Batch size for inference (default 8, set 1 for sequential)
        max_pixels: Maximum pixels for image resizing (default 786432 for efficiency)
        pass_group_number: Optional cap on pass groups; random sampling applied if total exceeds this.
        fail_group_number: Optional cap on fail groups; random sampling applied if total exceeds this.
        sample_seed: Random seed used when performing pass/fail sampling.
        sharding_mode: Execution strategy: per_group (group-level sharding) or per_image (image-level sharding + rank-0 merge).
        keep_intermediate_outputs: Keep intermediate per-rank per-image outputs in per_image mode (default: delete after successful merge).
    """
    # Initialize distributed mode if launched under torchrun
    init_distributed()
    world_size = get_world_size()
    rank = get_rank()
    distributed = world_size > 1

    # Override device with per-rank device in distributed mode
    if distributed and torch.cuda.is_available():
        local_rank = get_local_rank()
        device = f"cuda:{local_rank}"
        if is_main_process():
            logger.info(
                "Distributed mode: using per-rank device assignment (LOCAL_RANK=%d, device=%s)",
                local_rank,
                device,
            )

    # Default generation params
    if gen_params is None:
        gen_params = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }

    # Print config summary (only on main process to reduce log spam)
    if is_main_process() or not distributed:
        logger.info("=" * 70)
        logger.info("Stage-A Inference Configuration")
        logger.info("=" * 70)
        logger.info(f"Checkpoint: {checkpoint}")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Mission: {mission}")
        logger.info(f"Dataset: {dataset}")
        logger.info(f"Prompt profile: {prompt_profile}")
        logger.info(f"Device: {device}")
        if distributed:
            logger.info(f"Distributed mode: WORLD_SIZE={world_size}, RANK={rank}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Sharding mode: {sharding_mode}")
        logger.info(f"Max pixels: {max_pixels}")
        logger.info(f"Generation params: {gen_params}")
        logger.info("=" * 70)

    # Discover groups (all ranks discover the same groups)
    if is_main_process() or not distributed:
        logger.info("Discovering groups...")
    groups = discover_groups(Path(input_dir), mission=mission)
    original_group_count = len(groups)
    if is_main_process() or not distributed:
        logger.info(f"Found {original_group_count} groups for mission '{mission}'")

    sampled_groups: list[GroupInfo] | None = None
    sampling_stats: dict[str, int] | None = None
    if distributed:
        if is_main_process():
            sampled_groups, sampling_stats = _sample_groups(
                groups, pass_group_number, fail_group_number, sample_seed
            )
        sampled_groups = cast(
            list[GroupInfo] | None, broadcast_object(sampled_groups)
        )
        groups = sampled_groups if sampled_groups is not None else []
    else:
        groups, sampling_stats = _sample_groups(
            groups, pass_group_number, fail_group_number, sample_seed
        )

    if sampling_stats and (is_main_process() or not distributed):
        logger.info(
            "Sampling result: pass %d/%d, fail %d/%d, selected %d/%d groups (seed=%d)",
            sampling_stats["pass_selected"],
            sampling_stats["pass_total"],
            sampling_stats["fail_selected"],
            sampling_stats["fail_total"],
            len(groups),
            original_group_count,
            sample_seed,
        )

    if not groups:
        if is_main_process() or not distributed:
            logger.warning("No groups found; exiting")
        return

    if sharding_mode not in {"per_group", "per_image"}:
        raise ValueError(f"Unsupported sharding_mode: {sharding_mode!r}")

    # Load model and processor (all ranks).
    model, processor = load_model_processor(checkpoint, device, max_pixels=max_pixels)

    if sharding_mode == "per_group":
        # Shard groups by rank in distributed mode.
        if distributed:
            my_groups = groups[rank::world_size]
            if is_main_process():
                logger.info(
                    "Distributed sharding (per_group): rank %d processing %d/%d groups",
                    rank,
                    len(my_groups),
                    len(groups),
                )
        else:
            my_groups = groups

        # In distributed mode, each rank writes to a temp file, then rank 0 merges.
        if distributed:
            output_path = Path(output_dir) / f"{mission}_stage_a.rank{rank}.jsonl"
        else:
            output_path = Path(output_dir) / f"{mission}_stage_a.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Optional progress bar (only on main process in distributed mode).
        pbar = None
        if is_main_process() or not distributed:
            try:
                from tqdm import tqdm

                pbar = tqdm(
                    total=len(my_groups), desc="Processing groups", unit="group"
                )
            except ImportError:
                pass

        processed, errors = _run_per_group(
            groups=my_groups,
            model=model,
            processor=processor,
            mission=mission,
            dataset=dataset,
            prompt_profile=prompt_profile,
            gen_config=gen_params,
            batch_size=batch_size,
            verify_inputs=verify_inputs,
            output_path=output_path,
            pbar=pbar,
            distributed=distributed,
        )

        if pbar is not None:
            pbar.close()

        # In distributed mode, wait for all ranks to finish, then merge on rank 0.
        if distributed:
            barrier()  # Wait for all ranks to finish processing

            if is_main_process():
                final_output_path = Path(output_dir) / f"{mission}_stage_a.jsonl"
                logger.info("Merging per-rank outputs into %s", final_output_path)

                total_processed = 0
                with final_output_path.open("w", encoding="utf-8") as f_out:
                    for r in range(world_size):
                        rank_file = (
                            Path(output_dir) / f"{mission}_stage_a.rank{r}.jsonl"
                        )
                        if rank_file.exists():
                            with rank_file.open("r", encoding="utf-8") as f_in:
                                for line in f_in:
                                    f_out.write(line)
                                    total_processed += 1
                            rank_file.unlink()
                        else:
                            logger.warning(
                                "Rank %d output file not found: %s",
                                r,
                                rank_file,
                            )

                logger.info("=" * 70)
                logger.info("Stage-A Inference Complete (Distributed, per_group)")
                logger.info(f"Total processed: {total_processed} groups")
                logger.info(f"Output: {final_output_path}")
                logger.info("=" * 70)
            else:
                logger.info(
                    "Rank %d complete (per_group): processed %d/%d groups, errors: %d",
                    rank,
                    processed,
                    len(my_groups),
                    errors,
                )
        else:
            logger.info("=" * 70)
            logger.info("Stage-A Inference Complete (per_group)")
            logger.info(f"Processed: {processed}/{len(groups)} groups")
            logger.info(f"Errors: {errors}")
            logger.info(f"Output: {output_path}")
            logger.info("Results written incrementally (can tail -f to monitor)")
            logger.info("=" * 70)
        return

    # per_image mode: shard at the image level; merge on rank 0 at the end.
    jobs = _build_image_jobs(groups)
    if distributed:
        my_jobs = jobs[rank::world_size]
        if is_main_process():
            logger.info(
                "Distributed sharding (per_image): rank %d processing %d/%d images",
                rank,
                len(my_jobs),
                len(jobs),
            )
    else:
        my_jobs = jobs

    per_image_path = Path(output_dir) / f"{mission}_stage_a.images.rank{rank}.jsonl"
    per_image_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = None
    if is_main_process() or not distributed:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(my_jobs), desc="Processing images", unit="image")
        except ImportError:
            pass

    processed_images, errors_images = _run_per_image_jobs(
        jobs=my_jobs,
        groups=groups,
        model=model,
        processor=processor,
        mission=mission,
        dataset=dataset,
        prompt_profile=prompt_profile,
        gen_config=gen_params,
        batch_size=batch_size,
        verify_inputs=verify_inputs,
        output_path=per_image_path,
        pbar=pbar,
        distributed=distributed,
    )

    if pbar is not None:
        pbar.close()

    if distributed:
        barrier()  # Ensure all per-rank intermediates exist before merge.

    if is_main_process():
        final_output_path = Path(output_dir) / f"{mission}_stage_a.jsonl"
        logger.info("Merging per-image outputs into %s", final_output_path)
        merged_groups, failed_groups = _merge_per_image_outputs(
            groups=groups,
            output_dir=Path(output_dir),
            mission=mission,
            world_size=world_size if distributed else 1,
            keep_intermediate_outputs=keep_intermediate_outputs,
            dataset=dataset,
        )

        logger.info("=" * 70)
        header = (
            "Stage-A Inference Complete (Distributed, per_image)"
            if distributed
            else "Stage-A Inference Complete (per_image)"
        )
        logger.info(header)
        logger.info(f"Total processed: {merged_groups} groups")
        logger.info(f"Group failures: {failed_groups}")
        logger.info(f"Output: {final_output_path}")
        logger.info("=" * 70)
    else:
        logger.info(
            "Rank %d complete (per_image): processed %d/%d images, errors: %d",
            rank,
            processed_images,
            len(my_jobs),
            errors_images,
        )
    return


def _run_per_group(
    *,
    groups: list[GroupInfo],
    model: Qwen3VLForConditionalGeneration,
    processor: ProcessorProtocol,
    mission: str,
    dataset: str,
    prompt_profile: str,
    gen_config: dict[str, object],
    batch_size: int,
    verify_inputs: bool,
    output_path: Path,
    pbar: object | None,
    distributed: bool,
) -> tuple[int, int]:
    processed = 0
    errors = 0

    with output_path.open("w", encoding="utf-8") as f_out:
        for group_info in groups:
            group_id = group_info.group_id
            try:
                record = process_group(
                    group_id=group_id,
                    group_info=group_info,
                    model=model,
                    processor=processor,
                    mission=mission,
                    dataset=dataset,
                prompt_profile=prompt_profile,
                gen_config=gen_config,
                batch_size=batch_size,
                verify=verify_inputs,
            )

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                processed += 1

                if pbar is not None:
                    _safe_pbar_update(pbar, 1)

                if (is_main_process() or not distributed) and processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(groups)} groups")
            except Exception as exc:
                logger.error(f"Failed to process group {group_id}: {exc}")
                errors += 1

    return processed, errors


@dataclass
class _GroupAccum:
    seq: int
    info: GroupInfo
    expected_images: int
    per_image: list[str | None]
    done: bool = False
    failed: bool = False


@dataclass(frozen=True)
class _ImageJob:
    group_seq: int
    image_index: int  # 1-based index within the group
    path: Path


def _build_image_jobs(groups: list[GroupInfo]) -> list[_ImageJob]:
    """Flatten groups into a deterministic list of per-image jobs."""
    jobs: list[_ImageJob] = []
    for group_seq, info in enumerate(groups):
        for image_index, path in enumerate(info.paths, start=1):
            jobs.append(
                _ImageJob(group_seq=group_seq, image_index=image_index, path=path)
            )
    return jobs


def _run_per_image_jobs(
    *,
    jobs: list[_ImageJob],
    groups: list[GroupInfo],
    model: Qwen3VLForConditionalGeneration,
    processor: ProcessorProtocol,
    mission: str,
    dataset: str,
    prompt_profile: str,
    gen_config: dict[str, object],
    batch_size: int,
    verify_inputs: bool,
    output_path: Path,
    pbar: object | None,
    distributed: bool,
) -> tuple[int, int]:
    """Run Stage-A in per-image mode and write per-image intermediate outputs.

    Each job produces exactly one JSONL line:
    - success: (group_seq, image_index) -> summary
    - failure: (group_seq, image_index) -> error
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    user_text = build_user_prompt(
        mission,
        dataset=dataset,
        profile_name=prompt_profile,
    )
    system_text = build_system_prompt(
        mission,
        dataset=dataset,
        profile_name=prompt_profile,
    )

    processed = 0
    errors = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start : start + batch_size]

            loaded_jobs: list[_ImageJob] = []
            images: list[Image.Image] = []

            def _write_failure(job: _ImageJob, *, error: str) -> None:
                nonlocal processed, errors
                info = groups[job.group_seq]
                payload: dict[str, object] = {
                    "group_seq": job.group_seq,
                    "group_id": info.group_id,
                    "label": info.label,
                    "image_index": job.image_index,
                    "image_name": job.path.name,
                    "ok": False,
                    "error": error,
                }
                f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                f_out.flush()
                processed += 1
                errors += 1
                if pbar is not None:
                    _safe_pbar_update(pbar, 1)

            def _write_success(job: _ImageJob, *, summary: str) -> None:
                nonlocal processed
                info = groups[job.group_seq]
                sanitized = sanitize_summary_by_dataset(summary, dataset)
                payload: dict[str, object] = {
                    "group_seq": job.group_seq,
                    "group_id": info.group_id,
                    "label": info.label,
                    "image_index": job.image_index,
                    "image_name": job.path.name,
                    "ok": True,
                    "summary": sanitized,
                }
                f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                f_out.flush()
                processed += 1
                if pbar is not None:
                    _safe_pbar_update(pbar, 1)

            # Decode images (bounded by batch_size).
            for job in chunk:
                try:
                    img = apply_exif_orientation(Image.open(job.path))
                except Exception as exc:
                    logger.error(
                        "Failed to open image %s (group_seq=%d, group_id=%s): %s",
                        job.path,
                        job.group_seq,
                        groups[job.group_seq].group_id,
                        exc,
                    )
                    _write_failure(job, error=str(exc))
                    continue

                loaded_jobs.append(job)
                images.append(img)

            if not loaded_jobs:
                continue

            # Infer for this batch. If batch inference fails, fall back to per-image.
            outputs: list[tuple[str, str]] = []
            per_job_error: dict[int, str] = {}

            if len(images) == 1:
                try:
                    outputs = [
                        infer_one_image(
                            model,
                            processor,
                            images[0],
                            user_text,
                            gen_config,
                            verify=verify_inputs,
                            system_prompt=system_text,
                        )
                    ]
                except Exception as exc:
                    per_job_error[0] = str(exc)
                    outputs = [("", "")]
            else:
                try:
                    outputs = infer_batch(
                        model,
                        processor,
                        images,
                        user_text,
                        gen_config,
                        verify=verify_inputs,
                        system_prompt=system_text,
                    )
                except Exception as exc:
                    logger.error(
                        "Batch inference failed (len=%d); falling back to per-image: %s",
                        len(images),
                        exc,
                    )
                    outputs = []
                    for idx, (job, img) in enumerate(zip(loaded_jobs, images)):
                        try:
                            outputs.append(
                                infer_one_image(
                                    model,
                                    processor,
                                    img,
                                    user_text,
                                    gen_config,
                                    verify=verify_inputs,
                                    system_prompt=system_text,
                                )
                            )
                        except Exception as per_exc:
                            per_job_error[idx] = str(per_exc)
                            outputs.append(("", ""))

            # Drop images eagerly to keep memory bounded to the current batch.
            for img in images:
                try:
                    img.close()
                except Exception:
                    pass
            del images

            if len(outputs) != len(loaded_jobs):
                logger.error(
                    "Infer returned mismatched response count: %d outputs vs %d jobs",
                    len(outputs),
                    len(loaded_jobs),
                )
                for job in loaded_jobs:
                    _write_failure(job, error="mismatched infer output count")
                continue

            for idx, (job, (_raw, clean)) in enumerate(zip(loaded_jobs, outputs)):
                if clean:
                    _write_success(job, summary=clean)
                    continue
                err = per_job_error.get(idx, "empty summary")
                _write_failure(job, error=err)

    if (is_main_process() or not distributed) and processed:
        logger.info(
            "Per-image intermediates written: %s (%d images, %d errors)",
            output_path,
            processed,
            errors,
        )

    return processed, errors


def _merge_per_image_outputs(
    *,
    groups: list[GroupInfo],
    output_dir: Path,
    mission: str,
    world_size: int,
    keep_intermediate_outputs: bool,
    dataset: str,
) -> tuple[int, int]:
    """Merge per-rank per-image outputs into group-level Stage-A JSONL.

    Returns:
        (merged_groups, failed_groups)
    """
    final_output_path = output_dir / f"{mission}_stage_a.jsonl"

    # Prepare per-group buffers.
    per_group: list[list[str | None]] = [
        [None for _ in range(len(info.paths))] for info in groups
    ]
    failed: list[bool] = [False for _ in groups]
    seen: set[tuple[int, int]] = set()

    def _mark_failed(group_seq: int, *, reason: str) -> None:
        if group_seq < 0 or group_seq >= len(groups):
            logger.error("Invalid group_seq=%d (reason=%s)", group_seq, reason)
            return
        if not failed[group_seq]:
            logger.error(
                "Group failed at merge: group_seq=%d group_id=%s (%s)",
                group_seq,
                groups[group_seq].group_id,
                reason,
            )
        failed[group_seq] = True

    # Load all intermediates.
    for r in range(world_size):
        path = output_dir / f"{mission}_stage_a.images.rank{r}.jsonl"
        if not path.exists():
            logger.warning("Per-image output file not found for rank %d: %s", r, path)
            continue
        with path.open("r", encoding="utf-8") as f_in:
            for line_number, line in enumerate(f_in, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    logger.error(
                        "Invalid JSON at %s:%d (%s)", path.name, line_number, exc
                    )
                    continue

                try:
                    group_seq = int(payload["group_seq"])
                    image_index = int(payload["image_index"])
                except Exception as exc:
                    logger.error(
                        "Invalid per-image keys at %s:%d (%s)",
                        path.name,
                        line_number,
                        exc,
                    )
                    continue

                key = (group_seq, image_index)
                if key in seen:
                    _mark_failed(group_seq, reason=f"duplicate result for {key}")
                    continue
                seen.add(key)

                if group_seq < 0 or group_seq >= len(groups):
                    logger.error(
                        "Out-of-range group_seq=%d at %s:%d",
                        group_seq,
                        path.name,
                        line_number,
                    )
                    continue

                expected = len(groups[group_seq].paths)
                if image_index < 1 or image_index > expected:
                    _mark_failed(
                        group_seq,
                        reason=f"out-of-range image_index={image_index} expected=1..{expected}",
                    )
                    continue

                ok = bool(payload.get("ok", False))
                if not ok:
                    _mark_failed(group_seq, reason=str(payload.get("error", "error")))
                    continue

                summary = str(payload.get("summary", "")).strip()
                if not summary:
                    _mark_failed(group_seq, reason="empty summary")
                    continue

                sanitized = sanitize_summary_by_dataset(summary, dataset)
                per_group[group_seq][image_index - 1] = sanitized

    merged_groups = 0
    failed_groups = 0

    with final_output_path.open("w", encoding="utf-8") as f_out:
        for group_seq, info in enumerate(groups):
            missing = [
                idx for idx, v in enumerate(per_group[group_seq], start=1) if v is None
            ]
            if failed[group_seq] or missing:
                if not failed[group_seq] and missing:
                    _mark_failed(group_seq, reason=f"missing image indices: {missing}")
                failed_groups += 1
                continue

            per_image_map: dict[str, str] = {}
            for idx, text in enumerate(per_group[group_seq], start=1):
                if text is None:
                    # Defensive: should have been caught above.
                    _mark_failed(group_seq, reason="missing summary at write time")
                    failed_groups += 1
                    break
                per_image_map[f"image_{idx}"] = text
            else:
                record = {
                    "group_id": info.group_id,
                    "mission": info.mission,
                    "label": info.label,
                    "images": [p.name for p in info.paths],
                    "per_image": per_image_map,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                merged_groups += 1

    # Cleanup intermediates by default.
    if not keep_intermediate_outputs:
        for r in range(world_size):
            path = output_dir / f"{mission}_stage_a.images.rank{r}.jsonl"
            try:
                if path.exists():
                    path.unlink()
            except Exception as exc:
                logger.warning("Failed to delete intermediate file %s: %s", path, exc)

    return merged_groups, failed_groups


def _run_cross_group_batches(
    *,
    groups: list[GroupInfo],
    model: Qwen3VLForConditionalGeneration,
    processor: ProcessorProtocol,
    mission: str,
    dataset: str,
    gen_config: dict[str, object],
    batch_size: int,
    verify_inputs: bool,
    prompt_profile: str,
    output_path: Path,
    pbar: object | None,
    distributed: bool,
) -> tuple[int, int]:
    """Run Stage-A using cross-group image batching while preserving per-group outputs.

    Constraints:
    - At most `batch_size` images are decoded/in-flight per batch.
    - Groups are flushed in discovery order per rank; failures count as complete for ordering.
    - Failures do not emit partial group records and do not stop other groups.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    user_text = build_user_prompt(
        mission,
        dataset=dataset,
        profile_name=prompt_profile,
    )
    system_text = build_system_prompt(
        mission,
        dataset=dataset,
        profile_name=prompt_profile,
    )

    accums: list[_GroupAccum] = []
    for seq, info in enumerate(groups):
        expected = len(info.paths)
        accums.append(
            _GroupAccum(
                seq=seq,
                info=info,
                expected_images=expected,
                per_image=[None for _ in range(expected)],
            )
        )

    processed = 0
    errors = 0

    # Completed group outputs keyed by seq; value None means "failed, no record".
    completed: dict[int, dict[str, object] | None] = {}
    next_flush_seq = 0

    # Sequential cursor over groups and per-group image index.
    group_cursor = 0
    image_cursor = 0

    def _mark_group_failed(group_seq: int) -> None:
        nonlocal errors
        acc = accums[group_seq]
        if acc.done:
            return
        acc.failed = True
        acc.done = True
        acc.per_image = [None for _ in range(acc.expected_images)]
        completed[group_seq] = None
        errors += 1
        if pbar is not None:
            _safe_pbar_update(pbar, 1)

    def _maybe_finish_group(group_seq: int) -> None:
        acc = accums[group_seq]
        if acc.done:
            return
        if any(v is None for v in acc.per_image):
            return

        # Build record with strict coverage (fail-fast defensive checks).
        per_image_map: dict[str, str] = {}
        for idx, text in enumerate(acc.per_image, start=1):
            if text is None:
                raise ValueError("Internal error: per_image missing after completion")
            per_image_map[f"image_{idx}"] = text

        record = {
            "group_id": acc.info.group_id,
            "mission": acc.info.mission,
            "label": acc.info.label,
            "images": [p.name for p in acc.info.paths],
            "per_image": per_image_map,
        }

        acc.done = True
        completed[group_seq] = record
        if pbar is not None:
            _safe_pbar_update(pbar, 1)

    def _flush_ready(f_out) -> None:
        nonlocal next_flush_seq, processed
        while next_flush_seq in completed:
            record = completed.pop(next_flush_seq)
            if record is not None:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                processed += 1
                if (is_main_process() or not distributed) and processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(groups)} groups")
            next_flush_seq += 1

    with output_path.open("w", encoding="utf-8") as f_out:
        while group_cursor < len(groups):
            # Skip already-done groups (can happen after marking failed).
            if group_cursor < len(accums) and accums[group_cursor].done:
                group_cursor += 1
                image_cursor = 0
                continue

            # Assemble a batch of jobs from successive groups.
            jobs: list[_ImageJob] = []
            images: list[Image.Image] = []

            local_group_cursor = group_cursor
            local_image_cursor = image_cursor

            while len(jobs) < batch_size and local_group_cursor < len(groups):
                acc = accums[local_group_cursor]
                if acc.done:
                    local_group_cursor += 1
                    local_image_cursor = 0
                    continue

                if local_image_cursor >= acc.expected_images:
                    local_group_cursor += 1
                    local_image_cursor = 0
                    continue

                path = acc.info.paths[local_image_cursor]
                job = _ImageJob(
                    group_seq=acc.seq,
                    image_index=local_image_cursor + 1,
                    path=path,
                )

                try:
                    img = apply_exif_orientation(Image.open(path))
                except Exception as exc:
                    logger.error(
                        "Failed to open image %s (group=%s): %s",
                        path,
                        acc.info.group_id,
                        exc,
                    )
                    _mark_group_failed(acc.seq)
                    local_group_cursor += 1
                    local_image_cursor = 0
                    continue

                jobs.append(job)
                images.append(img)
                local_image_cursor += 1
                if local_image_cursor >= acc.expected_images:
                    local_group_cursor += 1
                    local_image_cursor = 0

            # If we couldn't enqueue anything (all remaining groups failed), stop.
            if not jobs:
                break

            # Commit cursor advancement.
            group_cursor = local_group_cursor
            image_cursor = local_image_cursor

            # Run inference for the batch. If batch inference fails, fallback to per-image.
            outputs: list[tuple[str, str]] = []
            if batch_size == 1 and len(images) == 1:
                try:
                    outputs = [
                        infer_one_image(
                            model,
                            processor,
                            images[0],
                            user_text,
                            gen_config,
                            verify=verify_inputs,
                            system_prompt=system_text,
                        )
                    ]
                except Exception as exc:
                    logger.error("Inference failed for %s: %s", jobs[0].path, exc)
                    _mark_group_failed(jobs[0].group_seq)
                    outputs = []
            else:
                try:
                    outputs = infer_batch(
                        model,
                        processor,
                        images,
                        user_text,
                        gen_config,
                        verify=verify_inputs,
                        system_prompt=system_text,
                    )
                except Exception as exc:
                    logger.error(
                        "Batch inference failed (len=%d); falling back to per-image: %s",
                        len(images),
                        exc,
                    )
                    outputs = []
                    for job, img in zip(jobs, images):
                        if accums[job.group_seq].done:
                            outputs.append(("", ""))
                            continue
                        try:
                            outputs.append(
                                infer_one_image(
                                    model,
                                    processor,
                                    img,
                                    user_text,
                                    gen_config,
                                    verify=verify_inputs,
                                    system_prompt=system_text,
                                )
                            )
                        except Exception as per_exc:
                            logger.error(
                                "Inference failed for %s: %s", job.path, per_exc
                            )
                            _mark_group_failed(job.group_seq)
                            outputs.append(("", ""))

            # Drop images eagerly to keep memory bounded to the current batch.
            for img in images:
                try:
                    img.close()
                except Exception:
                    pass
            del images

            # Re-aggregate outputs back into per-group buffers.
            if outputs and len(outputs) != len(jobs):
                # Defensive: if mismatch, fail all involved groups to avoid corrupt alignment.
                logger.error(
                    "Sampler returned mismatched response count: %d outputs vs %d jobs",
                    len(outputs),
                    len(jobs),
                )
                for job in jobs:
                    _mark_group_failed(job.group_seq)
                _flush_ready(f_out)
                continue

            for job, (_raw, clean) in zip(jobs, outputs):
                acc = accums[job.group_seq]
                if acc.done:
                    continue
                if not clean:
                    _mark_group_failed(job.group_seq)
                    continue
                slot = job.image_index - 1
                if slot < 0 or slot >= acc.expected_images:
                    _mark_group_failed(job.group_seq)
                    continue
                sanitized = sanitize_summary_by_dataset(clean, dataset)
                acc.per_image[slot] = sanitized
                _maybe_finish_group(job.group_seq)

            _flush_ready(f_out)

        # Flush any remaining completed groups (including trailing failures).
        _flush_ready(f_out)

    return processed, errors
