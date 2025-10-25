#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRPO training launcher for Stage-B group-level judgment.

Python launcher that integrates with ms-swift GRPO trainer:
- LLM-only LoRA on last-K transformer blocks
- Frozen vision encoder and aligner
- Custom reward functions (label + format)
- Text-only dataset (no images)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stage_b.rewards import label_reward, format_reward

logger = logging.getLogger("run_grpo")


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    
    # Required fields first (no defaults)
    model_path: str
    train_dataset_path: str  # Path or list of paths to Stage-A JSONL files
    
    # Optional fields (with defaults)
    adapter_path: Optional[str] = None  # Load existing adapter (e.g., from Stage-A)
    val_dataset_path: Optional[str] = None
    
    # LoRA configuration
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_last_k_blocks: int = 4  # Only tune last K LLM blocks
    
    # Freezing
    freeze_vit: bool = True
    freeze_aligner: bool = True
    
    # GRPO parameters
    num_generations: int = 4  # Must be >= 2
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Rewards
    reward_weights: Optional[List[float]] = None  # [label_weight, format_weight]
    
    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    warmup_steps: int = 100
    
    # Output
    output_dir: str = "output_post/grpo"
    save_steps: int = 100
    logging_steps: int = 10
    
    # Device
    device: str = "cuda:0"
    
    # Dry run controls
    dry_run: bool = False
    dry_run_samples: int = 2  # number of dataset samples to generate during dry-run
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_generations < 2:
            raise ValueError("num_generations must be >= 2 for GRPO")
        
        if self.reward_weights is None:
            self.reward_weights = [1.0, 0.2]  # Default: emphasize label over format
        
        if len(self.reward_weights) != 2:
            raise ValueError(f"reward_weights must have length 2, got {len(self.reward_weights)}")


from src.stage_b.dataset import load_stage_a_for_grpo


def get_llm_layer_names(model, last_k: int) -> List[str]:
    """Get names of last-K LLM transformer blocks for LoRA.
    
    Args:
        model: Qwen3VLForConditionalGeneration model
        last_k: Number of final blocks to target
        
    Returns:
        List of module name patterns for LoRA target_modules
        
    Example:
        For Qwen2VL, returns patterns like:
        ["model.layers.27", "model.layers.28", ...] for last 4 blocks
    """
    # Qwen2VL structure: model.layers[0..N-1]
    total_layers = len(model.model.layers)
    start_layer = max(0, total_layers - last_k)
    
    target_patterns = []
    for i in range(start_layer, total_layers):
        # Target all linear layers in these blocks
        target_patterns.append(f"model.layers.{i}")
    
    logger.info(f"Targeting last {last_k} LLM blocks: layers {start_layer} to {total_layers-1}")
    return target_patterns


def freeze_modules(model, freeze_vit: bool = True, freeze_aligner: bool = True):
    """Freeze vision encoder and/or aligner modules.
    
    Args:
        model: Qwen3VLForConditionalGeneration model
        freeze_vit: Whether to freeze vision encoder
        freeze_aligner: Whether to freeze aligner (MLP projector)
    """
    if freeze_vit:
        # Freeze visual encoder
        if hasattr(model, 'visual'):
            for param in model.visual.parameters():
                param.requires_grad = False
            logger.info("✓ Froze vision encoder (visual)")
        else:
            logger.warning("Model has no 'visual' attribute to freeze")
    
    if freeze_aligner:
        # Freeze aligner/merger (multimodal projector)
        if hasattr(model, 'merger'):
            for param in model.merger.parameters():
                param.requires_grad = False
            logger.info("✓ Froze aligner (merger)")
        else:
            logger.warning("Model has no 'merger' attribute to freeze")


def create_reward_function(config: GRPOConfig):
    """Create combined reward function for GRPO.
    
    Args:
        config: GRPO configuration with reward_weights
        
    Returns:
        Callable reward function that computes weighted sum
    """
    label_weight = config.reward_weights[0]
    format_weight = config.reward_weights[1]
    
    logger.info(f"Reward weights: label={label_weight}, format={format_weight}")
    
    def combined_reward(responses: List[str], row: Dict[str, Any], **kwargs) -> List[float]:
        """Combined reward: weighted sum of label and format rewards."""
        label_rewards = label_reward(responses, row, **kwargs)
        format_rewards = format_reward(responses, row, **kwargs)
        
        combined = [
            label_weight * lr + format_weight * fr
            for lr, fr in zip(label_rewards, format_rewards)
        ]
        return combined
    
    return combined_reward


def run_grpo_training(config: GRPOConfig):
    """Run GRPO training with ms-swift integration.
    
    Args:
        config: GRPO configuration
        
    Note:
        This is a simplified launcher. Full integration requires:
        1. Installing ms-swift with GRPO support
        2. Adapting to ms-swift's exact API (GRPOTrainer, RLHFArguments)
        3. Proper LoRA setup with last-K block targeting
        
        The implementation below shows the intended structure and can be
        completed once ms-swift GRPO API is confirmed.
    """
    logger.info("="*60)
    logger.info("GRPO Training Launcher (Stage-B)")
    logger.info("="*60)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Dataset: {config.train_dataset_path}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Device: {config.device}")
    logger.info("="*60)
    
    # Load dataset (from Stage-A JSONL outputs)
    logger.info("\n[1/6] Loading Stage-A outputs for GRPO...")
    
    # Handle both single path and list of paths
    if isinstance(config.train_dataset_path, str):
        train_paths = [config.train_dataset_path]
    elif isinstance(config.train_dataset_path, list):
        train_paths = config.train_dataset_path
    else:
        raise TypeError(f"train_dataset_path must be str or list, got {type(config.train_dataset_path)}")
    
    logger.info(f"Loading {len(train_paths)} Stage-A JSONL file(s):")
    for path in train_paths:
        logger.info(f"  - {path}")
    
    train_dataset = load_stage_a_for_grpo(train_paths)
    logger.info(f"✓ Loaded {len(train_dataset)} training groups")
    
    val_dataset = None
    if config.val_dataset_path:
        val_paths = [config.val_dataset_path] if isinstance(config.val_dataset_path, str) else config.val_dataset_path
        logger.info(f"Loading {len(val_paths)} validation JSONL file(s)")
        val_dataset = load_stage_a_for_grpo(val_paths)
        logger.info(f"✓ Loaded {len(val_dataset)} validation groups")
    
    # Load model and processor
    logger.info("\n[2/6] Loading model and processor...")
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )
    
    # Load Qwen3-VL model and move to device
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        trust_remote_code=True
    )
    model.to(config.device)
    
    # Freeze modules
    logger.info("\n[3/6] Freezing vision and aligner modules...")
    freeze_modules(model, config.freeze_vit, config.freeze_aligner)
    
    # Setup LoRA on last-K blocks
    logger.info("\n[4/6] Setting up LoRA on last-K LLM blocks...")
    target_modules = get_llm_layer_names(model, config.lora_last_k_blocks)
    
    logger.warning(
        "\n⚠️  Full LoRA + GRPO integration requires ms-swift GRPOTrainer.\n"
        "   This launcher provides the architecture and configuration.\n"
        "   To complete integration:\n"
        "   1. Install ms-swift with GRPO support\n"
        "   2. Use swift.trainers.rlhf_trainer.grpo_trainer.GRPOTrainer\n"
        "   3. Pass reward_fn via external_plugins or direct callback\n"
        "   4. See /data/ms-swift/examples/train/grpo/ for reference\n"
    )
    
    # Create reward function
    logger.info("\n[5/6] Creating reward function...")
    reward_fn = create_reward_function(config)
    
    # Placeholder for GRPO trainer initialization
    logger.info("\n[6/6] GRPO trainer setup...")
    logger.info("✓ Configuration validated")
    logger.info(f"✓ Train samples: {len(train_dataset)}")
    logger.info(f"✓ Reward function: label + format (weights: {config.reward_weights})")
    logger.info(f"✓ LoRA targets: {len(target_modules)} blocks")
    logger.info(f"✓ Generation config: num_gen={config.num_generations}, max_tokens={config.max_new_tokens}")
    
    # Optional dry-run completions generation
    if config.dry_run:
        try:
            _generate_dry_run_completions(
                model=model,
                processor=processor,
                dataset=train_dataset,
                reward_fn=reward_fn,
                output_dir=config.output_dir,
                num_generations=config.num_generations,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                sample_count=config.dry_run_samples,
            )
        except Exception as e:
            logger.error(f"Dry-run completions generation failed: {e}")
    
    # TODO: Complete integration with ms-swift GRPOTrainer
    # Example structure (pseudo-code):
    # from swift.trainers.rlhf_trainer.grpo_trainer import GRPOTrainer
    # from swift.trainers.rlhf_arguments import GRPOConfig as SwiftGRPOConfig
    #
    # swift_config = SwiftGRPOConfig(
    #     num_generations=config.num_generations,
    #     max_new_tokens=config.max_new_tokens,
    #     temperature=config.temperature,
    #     ...
    # )
    #
    # trainer = GRPOTrainer(
    #     model=model,
    #     tokenizer=processor.tokenizer,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     reward_fn=reward_fn,
    #     args=swift_config,
    # )
    #
    # trainer.train()
    # trainer.save_model(config.output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Dry-run complete. Ready for full GRPO integration.")
    logger.info("="*60)
    
    return model, processor, train_dataset


def _generate_dry_run_completions(
    *,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    dataset: Dataset,
    reward_fn,
    output_dir: str,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    sample_count: int,
) -> None:
    """Generate a small set of completions and write to completions.jsonl.
    
    Writes JSONL records with fields:
      {group_id, task_type, group_label, generation_index, response, rewards{combined,label,format}}
    """
    out_path = Path(output_dir) / "completions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating dry-run completions → {out_path} (samples={sample_count}, gens/sample={num_generations})")
    
    total = min(sample_count, len(dataset))
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(total):
            row = dataset[idx]
            messages = row["messages"]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=[prompt], return_tensors="pt")
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            
            responses: List[str] = []
            for _ in range(num_generations):
                with torch.inference_mode():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        use_cache=True,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )
                start = inputs["input_ids"].shape[-1]
                gen_only = gen[:, start:]
                try:
                    text = processor.batch_decode(
                        gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                except Exception:
                    text = processor.tokenizer.batch_decode(
                        gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                responses.append(text.strip())
            
            # Compute rewards
            combined = reward_fn(responses, row)
            labels = label_reward(responses, row)
            formats = format_reward(responses, row)
            
            for j, resp in enumerate(responses):
                rec = {
                    "group_id": row.get("group_id"),
                    "task_type": row.get("task_type"),
                    "group_label": row.get("group_label"),
                    "generation_index": j,
                    "response": resp,
                    "rewards": {
                        "combined": float(combined[j]),
                        "label": float(labels[j]),
                        "format": float(formats[j]),
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    logger.info(f"✓ Wrote {written} completion record(s) to {out_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GRPO training launcher for Stage-B group-level judgment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to base Qwen3-VL model checkpoint",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to Stage-A JSONL file(s) (e.g., output_post/stage_a/*.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_post/grpo",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--lora_last_k",
        type=int,
        default=4,
        help="Number of last LLM blocks to tune with LoRA. Default: 4",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Number of generations per sample (must be >= 2). Default: 4",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for training. Default: cuda:0",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate config and setup without training",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    
    # Build config
    # Handle train_dataset: if single path provided, use as string; if multiple, use as list
    train_path = args.train_dataset[0] if len(args.train_dataset) == 1 else args.train_dataset
    
    config = GRPOConfig(
        model_path=args.model,
        train_dataset_path=train_path,
        output_dir=args.output_dir,
        lora_last_k_blocks=args.lora_last_k,
        num_generations=args.num_generations,
        device=args.device,
        dry_run=args.dry_run,
    )
    
    try:
        run_grpo_training(config)
        logger.info("\n✅ Success!")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

