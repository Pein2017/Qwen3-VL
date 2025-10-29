#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core inference engine for Stage-A per-image summarization.

This module discovers groups from mission-based directories, runs batched
inference on images, and outputs grouped JSONL records with strict validation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .prompts import SUMMARY_SYSTEM_PROMPT, build_user_prompt
from ..utils import get_logger

logger = get_logger(__name__)

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

    paths: List[Path]
    label: str
    mission: str
    group_id: str


def _natural_key(s: str) -> List[Any]:
    """Natural sort key for filenames with numbers."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _maybe_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
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


def sanitize_single_image_summary(text: str) -> str:
    """Extract and sanitize the summary for a single image.

    - If `text` is a JSON object containing keys like 图片_1/图片_2, keep only 图片_1's value
    - Otherwise use the raw `text`
    - Do not alter content beyond trimming outer whitespace
    """
    summary_text = text.strip()

    obj = _maybe_parse_json_object(summary_text)
    if obj is not None:
        val = obj.get("图片_1")
        if isinstance(val, str):
            summary_text = val.strip()
        else:
            # Fallback: join all string values if 图片_1 missing
            collected: List[str] = []
            for k, v in obj.items():
                if isinstance(v, str):
                    collected.append(v.strip())
            if collected:
                summary_text = "，".join(collected)

    # Keep raw content (no count-marker removal or dedupe). Only strip surrounding whitespace.
    summary_text = summary_text.strip()
    return summary_text


def _trim_trailing_eos_pad(
    gen_ids: torch.Tensor, eos_id: Optional[int], pad_id: Optional[int]
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


def discover_groups(root: Path, mission: Optional[str] = None) -> Dict[str, GroupInfo]:
    """Discover and group images from mission-based directory structure.

    Expected structure:
        <root>/<mission>/{审核通过|审核不通过}/<group_id>/*.{jpg,jpeg,png}

    Args:
        root: Root directory path
        mission: Mission name to filter (processes only this mission)

    Returns:
        Mapping from group_id to GroupInfo

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

    groups: Dict[str, GroupInfo] = {}

    for mission_dir in mission_dirs:
        mission_name = mission_dir.name

        # Scan label directories (审核通过/审核不通过)
        for label_dir in mission_dir.iterdir():
            if not label_dir.is_dir():
                continue

            label_str = LABEL_DIR_MAP.get(label_dir.name)
            if label_str is None:
                logger.warning(f"Skipping unknown label directory: {label_dir.name}")
                continue

            # Scan group directories
            for group_dir in label_dir.iterdir():
                if not group_dir.is_dir():
                    continue

                # Discover images in group
                image_paths: List[Path] = []
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

                # Store group info
                if group_id in groups:
                    logger.warning(
                        f"Duplicate group_id {group_id}; using first occurrence"
                    )
                else:
                    groups[group_id] = GroupInfo(
                        paths=image_paths,
                        label=label_str,
                        mission=mission_name,
                        group_id=group_id,
                    )

    return groups


def load_model_processor(
    checkpoint: str, device: str, max_pixels: int = 786432
) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
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
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    # Configure max_pixels for compute efficiency
    # Lower values = faster inference but lower image quality
    # 786432 = 1024×768 equivalent (good balance)
    # Default Qwen3-VL is 12845056 (very high resolution)
    if hasattr(processor, "image_processor") and hasattr(
        processor.image_processor, "max_pixels"
    ):
        processor.image_processor.max_pixels = max_pixels
        logger.info(f"Set max_pixels={max_pixels} for image preprocessing")
    else:
        logger.warning(f"Could not set max_pixels (processor may not support it)")
    # Log critical image processor settings for alignment diagnostics
    try:
        ip = getattr(processor, "image_processor", None)
        logger.info(
            "ImageProcessor settings: do_resize=%s, patch_size=%s, merge_size=%s, min_pixels=%s, max_pixels=%s",
            getattr(ip, "do_resize", None),
            getattr(ip, "patch_size", None),
            getattr(ip, "merge_size", None),
            getattr(ip, "min_pixels", None),
            getattr(ip, "max_pixels", None),
        )
    except Exception:
        pass

    logger.info(f"Model loaded on {device}")
    return model, processor


def infer_one_image(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
    user_text: str,
    gen_config: Dict[str, Any],
    verify: bool = False,
) -> Tuple[str, str]:
    """Run inference on a single image.

    Args:
        model: Qwen3-VL model
        processor: AutoProcessor
        image: PIL Image
        user_text: User prompt text
        gen_config: Generation config dict

    Returns:
        Tuple of (raw_text, clean_text)

    Raises:
        ValueError: If clean_text is empty after stripping
    """
    # Build messages with typed content
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    # Encode
    # Respect max_pixels by passing images_kwargs explicitly (Qwen2VLImageProcessorFast)
    _img_kwargs = {}
    try:
        mp = getattr(getattr(processor, "image_processor", None), "max_pixels", None)
        if mp is not None:
            _img_kwargs["max_pixels"] = int(mp)
    except Exception:
        pass
    inputs = processor(
        images=[image], text=[text], return_tensors="pt", images_kwargs=_img_kwargs
    )
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
                processor.tokenizer, "image_token_id", None
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
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
    }
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
                image_token_id = getattr(processor.tokenizer, "image_token_id", None)
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

    # Generate
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            **gen_config,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Decode
    start = inputs["input_ids"].shape[-1]
    gen_only = gen[:, start:]
    # Trim trailing EOS/PAD tokens only (preserve other content)
    gen_only = _trim_trailing_eos_pad(
        gen_only,
        eos_id=getattr(processor.tokenizer, "eos_token_id", None),
        pad_id=getattr(processor.tokenizer, "pad_token_id", None),
    )

    try:
        raw_text = processor.batch_decode(
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
    except Exception:
        raw_text = processor.tokenizer.batch_decode(
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

    # Keep raw content; only strip outer whitespace
    raw_text = raw_text.strip()

    try:
        clean_text = processor.batch_decode(
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
    except Exception:
        clean_text = processor.tokenizer.batch_decode(
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

    clean_text = clean_text.strip()

    # Validation: fail-fast on empty summary
    if not clean_text:
        raise ValueError(
            "Empty summary generated (clean_text is empty after stripping)"
        )

    # Sanitize single-image summary to keep only 图片_1 content and remove redundancy
    clean_text = sanitize_single_image_summary(clean_text)
    return raw_text, clean_text


def infer_batch(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    images: List[Image.Image],
    user_text: str,
    gen_config: Dict[str, Any],
    verify: bool = False,
) -> List[Tuple[str, str]]:
    """Run batched inference on multiple images.

    Args:
        model: Qwen3-VL model
        processor: AutoProcessor
        images: List of PIL Images
        user_text: User prompt text (same for all images)
        gen_config: Generation config dict

    Returns:
        List of (raw_text, clean_text) tuples

    Raises:
        ValueError: If any clean_text is empty after stripping
    """
    # Build messages for each image
    messages_list = [
        [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
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
        processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        for msgs in messages_list
    ]

    # Batch encode with padding
    _img_kwargs = {}
    try:
        mp = getattr(getattr(processor, "image_processor", None), "max_pixels", None)
        if mp is not None:
            _img_kwargs["max_pixels"] = int(mp)
    except Exception:
        pass
    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
        images_kwargs=_img_kwargs,
    )
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
                processor.tokenizer, "image_token_id", None
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
    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()
    }
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
                        processor.tokenizer, "image_token_id", None
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

    # Generate for batch
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            **gen_config,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Decode each output
    start = inputs["input_ids"].shape[-1]
    outputs: List[Tuple[str, str]] = []

    for i in range(len(images)):
        gen_only = gen[i : i + 1, start:]
        # Trim trailing EOS/PAD tokens only (preserve other content)
        gen_only = _trim_trailing_eos_pad(
            gen_only,
            eos_id=getattr(processor.tokenizer, "eos_token_id", None),
            pad_id=getattr(processor.tokenizer, "pad_token_id", None),
        )

        try:
            raw_text = processor.batch_decode(
                gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]
        except Exception:
            raw_text = processor.tokenizer.batch_decode(
                gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]

        # Keep raw content; only strip outer whitespace
        raw_text = raw_text.strip()

        try:
            clean_text = processor.batch_decode(
                gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]
        except Exception:
            clean_text = processor.tokenizer.batch_decode(
                gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        clean_text = clean_text.strip()

        # Validation: fail-fast on empty summary
        if not clean_text:
            raise ValueError(
                f"Empty summary generated for image {i} in batch "
                "(clean_text is empty after stripping)"
            )

        # Sanitize single-image summary to keep only 图片_1 content and remove redundancy
        clean_text = sanitize_single_image_summary(clean_text)
        outputs.append((raw_text, clean_text))

    return outputs


def process_group(
    group_id: str,
    group_info: GroupInfo,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    mission: Optional[str],
    gen_config: Dict[str, Any],
    batch_size: int = 8,
    include_mission_focus: bool = True,
    verify: bool = False,
) -> Dict[str, Any]:
    """Process a single group with batched inference.

    Args:
        group_id: Group ID
        group_info: GroupInfo with paths and metadata
        model: Qwen3-VL model
        processor: AutoProcessor
        mission: Mission name (for prompt building)
        gen_config: Generation config dict
        batch_size: Batch size for inference (1 = sequential)

    Returns:
        JSONL record dict with all required fields

    Raises:
        ValueError: If validation fails (empty summary or 图片_{i} mismatch)
    """
    # Build mission-dependent user prompt
    user_text = build_user_prompt(mission if include_mission_focus else None)

    # Load images
    images: List[Image.Image] = []
    for path in group_info.paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {path}") from e

    # Run inference with batching
    raw_texts: List[str] = []
    clean_texts: List[str] = []

    num_images = len(images)
    for i in range(0, num_images, batch_size):
        chunk = images[i : i + batch_size]

        if batch_size == 1:
            # Sequential fallback
            raw, clean = infer_one_image(
                model, processor, chunk[0], user_text, gen_config, verify=verify
            )
            raw_texts.append(raw)
            clean_texts.append(clean)
        else:
            # Batched inference
            chunk_outputs = infer_batch(
                model, processor, chunk, user_text, gen_config, verify=verify
            )
            for raw, clean in chunk_outputs:
                raw_texts.append(raw)
                clean_texts.append(clean)

    # Build per_image mapping with 图片_{i} keys
    # Each single-image summary originally refers to 图片_1; we rewrite to 图片_{i}
    per_image: Dict[str, str] = {}
    for idx, clean_text in enumerate(clean_texts, start=1):
        key = f"图片_{idx}"
        per_image[key] = clean_text

    # Strict validation: 图片_{i} coverage
    if len(per_image) != num_images:
        raise ValueError(
            f"图片_{{i}} coverage mismatch for group {group_id}: "
            f"{len(per_image)} keys vs {num_images} images"
        )

    # Verify all keys are present
    for idx in range(1, num_images + 1):
        key = f"图片_{idx}"
        if key not in per_image:
            raise ValueError(f"Missing 图片_{idx} in per_image for group {group_id}")

    # Build record
    # Flattened record: keep only essential fields; drop raw/clean arrays to reduce redundancy
    record = {
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
    device: str = "cuda:0",
    gen_params: Optional[Dict[str, Any]] = None,
    batch_size: int = 8,
    max_pixels: int = 786432,
    include_mission_focus: bool = True,
    verify_inputs: bool = False,
) -> None:
    """Run Stage-A inference on mission-based directory.

    Args:
        checkpoint: HuggingFace checkpoint path
        input_dir: Root directory with mission/label/group structure
        output_dir: Output directory for JSONL files
        mission: Mission name to process
        device: Device string (cuda:N or cpu)
        gen_params: Generation parameters dict
        batch_size: Batch size for inference (default 8, set 1 for sequential)
        max_pixels: Maximum pixels for image resizing (default 786432 for efficiency)
    """
    # Default generation params
    if gen_params is None:
        gen_params = {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        }

    # Print config summary
    logger.info("=" * 70)
    logger.info("Stage-A Inference Configuration")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mission: {mission}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max pixels: {max_pixels}")
    logger.info(f"Generation params: {gen_params}")
    logger.info("=" * 70)

    # Discover groups
    logger.info("Discovering groups...")
    groups = discover_groups(Path(input_dir), mission=mission)
    logger.info(f"Found {len(groups)} groups for mission '{mission}'")

    if not groups:
        logger.warning("No groups found; exiting")
        return

    # Load model and processor
    model, processor = load_model_processor(checkpoint, device, max_pixels=max_pixels)

    # Prepare output path
    output_path = Path(output_dir) / f"{mission}_stage_a.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional progress bar
    pbar = None
    try:
        from tqdm import tqdm

        pbar = tqdm(total=len(groups), desc="Processing groups", unit="group")
    except ImportError:
        pass

    # Process groups and write immediately (streaming mode)
    processed = 0
    errors = 0

    # Open output file in append mode for immediate writing
    with output_path.open("w", encoding="utf-8") as f_out:
        for group_id, group_info in groups.items():
            try:
                record = process_group(
                    group_id=group_id,
                    group_info=group_info,
                    model=model,
                    processor=processor,
                    mission=mission,
                    gen_config=gen_params,
                    batch_size=batch_size,
                    include_mission_focus=include_mission_focus,
                    verify=verify_inputs,
                )

                # Write immediately after processing each group
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()  # Force write to disk immediately

                processed += 1

                if pbar is not None:
                    pbar.update(1)

                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{len(groups)} groups")

            except Exception as e:
                logger.error(f"Failed to process group {group_id}: {e}")
                errors += 1
                # Continue processing other groups

    if pbar is not None:
        pbar.close()

    # Final summary
    logger.info("=" * 70)
    logger.info("Stage-A Inference Complete")
    logger.info(f"Processed: {processed}/{len(groups)} groups")
    logger.info(f"Errors: {errors}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Results written incrementally (can tail -f to monitor)")
    logger.info("=" * 70)
