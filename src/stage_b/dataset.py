#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-B dataset loader for GRPO training.

Loads Stage-A JSONL outputs and builds GRPO-ready prompts dynamically.
The model generates Stage-B responses (two-line verdicts) during GRPO rollout.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset

from .prompts import MISSION_FOCUS_MAP, build_stage_b_messages

logger = logging.getLogger("stage_b.dataset")


def load_stage_a_for_grpo(
    stage_a_jsonl_paths: List[str],
    limit: Optional[int] = None,
) -> Dataset:
    """Load Stage-A JSONL outputs as GRPO training dataset.
    
    Args:
        stage_a_jsonl_paths: List of Stage-A JSONL file paths (one or more missions)
        limit: Optional limit on number of groups to load (for debugging)
        
    Returns:
        HuggingFace Dataset with columns:
            - group_id: str
            - task_type: str (mission name)
            - group_label: str ("通过" | "不通过")
            - stage_a_summaries: dict {图片_i: summary_text}
            - messages: list[dict] (system + user prompt, no assistant)
            
    Notes:
        - Messages are built with system + user prompts only
        - GRPO will generate assistant responses during rollout
        - GT label (group_label) is used for reward computation
        - stage_a_summaries passed to reward functions for consistency checks
    """
    records = []
    
    # Convert label: pass/fail → 通过/不通过
    label_map = {"pass": "通过", "fail": "不通过"}
    
    for jsonl_path in stage_a_jsonl_paths:
        path = Path(jsonl_path)
        if not path.exists():
            logger.warning(f"Skipping non-existent file: {path}")
            continue
        
        logger.info(f"Loading {path.name}...")
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    stage_a_record = json.loads(line)
                    
                    # Extract required fields
                    group_id = stage_a_record["group_id"]
                    mission = stage_a_record["mission"]
                    label = stage_a_record["label"]
                    per_image = stage_a_record["per_image"]
                    
                    # Validate mission
                    if mission not in MISSION_FOCUS_MAP:
                        logger.warning(
                            f"Unknown mission '{mission}' in {path.name}:{line_num}, skipping"
                        )
                        continue
                    
                    # Convert label
                    if label not in label_map:
                        logger.warning(
                            f"Invalid label '{label}' in {path.name}:{line_num}, skipping"
                        )
                        continue
                    
                    group_label = label_map[label]
                    
                    # Build messages (system + user, no assistant)
                    messages = build_stage_b_messages(
                        stage_a_summaries=per_image,
                        task_type=mission,
                    )
                    
                    # Create GRPO record
                    grpo_record = {
                        "group_id": group_id,
                        "task_type": mission,
                        "group_label": group_label,
                        "stage_a_summaries": per_image,
                        "messages": messages,
                    }
                    
                    records.append(grpo_record)
                    
                    # Check limit
                    if limit and len(records) >= limit:
                        logger.info(f"Reached limit of {limit} groups")
                        break
                
                except (KeyError, json.JSONDecodeError) as e:
                    logger.error(f"Error parsing {path.name}:{line_num}: {e}")
                    continue
            
            if limit and len(records) >= limit:
                break
    
    if not records:
        raise ValueError("No valid records loaded from Stage-A JSONL files")
    
    logger.info(f"✓ Loaded {len(records)} groups from {len(stage_a_jsonl_paths)} file(s)")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(records)
    
    # Print statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"  Total groups: {len(dataset)}")
    
    # Count by task type
    task_counts = {}
    for rec in records:
        task = rec["task_type"]
        task_counts[task] = task_counts.get(task, 0) + 1
    
    for task, count in sorted(task_counts.items()):
        logger.info(f"  {task}: {count}")
    
    # Count by label
    label_counts = {"通过": 0, "不通过": 0}
    for rec in records:
        label_counts[rec["group_label"]] += 1
    
    logger.info(f"  Labels: 通过={label_counts['通过']}, 不通过={label_counts['不通过']}")
    
    return dataset


def prepare_grpo_batch(batch: Dict[str, List[Any]], processor) -> Dict[str, Any]:
    """Prepare a batch for GRPO forward pass.
    
    Args:
        batch: Dict with keys from dataset (group_id, messages, etc.)
        processor: Qwen2VLProcessor instance
        
    Returns:
        Dict with tokenized inputs ready for model.generate()
        
    Notes:
        - Applies chat template to messages (system + user)
        - Tokenizes prompts without images (text-only)
        - Returns input_ids and attention_mask
    """
    prompts = []
    
    for messages in batch["messages"]:
        # Apply chat template (system + user turns only, no assistant)
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # Adds <|im_start|>assistant\n
        )
        prompts.append(prompt_text)
    
    # Tokenize (text-only, no images)
    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,  # Adjust based on summary lengths
    )
    
    return inputs


def save_grpo_dataset_preview(
    dataset: Dataset,
    output_path: str,
    num_samples: int = 3,
):
    """Save a preview of GRPO dataset for inspection.
    
    Args:
        dataset: Loaded GRPO dataset
        output_path: Path to save preview JSON
        num_samples: Number of samples to save
    """
    samples = []
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        samples.append({
            "group_id": sample["group_id"],
            "task_type": sample["task_type"],
            "group_label": sample["group_label"],
            "num_images": len(sample["stage_a_summaries"]),
            "system_prompt": sample["messages"][0]["content"][:200] + "...",
            "user_prompt": sample["messages"][1]["content"][:400] + "...",
        })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Saved dataset preview to {output_path}")

