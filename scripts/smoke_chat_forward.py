#!/usr/bin/env python3
"""
Manual smoke test: build a chat-only dataset sample and run one forward+backward step.

Usage:
  conda run -n ms python scripts/smoke_chat_forward.py \
    --chat-jsonl public_data/coig_cqia/coig_cqia_merged.jsonl \
    --model model_cache/models/Qwen/Qwen3-VL-4B-Instruct \
    --device cuda:0

Notes:
- Loads only a single sample (configurable via --sample-limit).
- Uses ms-swift's SwiftSft to get the tokenizer/template for chatml.
- Backward is run to verify gradients wire up; no optimizer step is taken.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import torch
from swift.llm import TrainArguments
from swift.llm.train.sft import SwiftSft

from src.datasets.dense_caption import BaseCaptionDataset
from src.datasets.utils import load_jsonl


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat fusion smoke forward pass")
    parser.add_argument(
        "--chat-jsonl",
        type=Path,
        required=True,
        help="Path to chat JSONL (messages-only, e.g., COIG-CQIA merged)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model_cache/models/Qwen/Qwen3-VL-4B-Instruct",
        help="Path or hub ID of the base model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for forward pass",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=1,
        help="Number of chat samples to load (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/chat_smoke",
        help="Where to write temporary checkpoints/logs if any (not used here)",
    )
    return parser.parse_args()


def _is_numeric_list(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) == 0:
        return False
    return all(isinstance(x, (int, float)) for x in value)


def to_batch(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0).to(device)
        elif _is_numeric_list(v):
            tensor = torch.tensor(v)
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            batch[k] = tensor.unsqueeze(0).to(device)
        else:
            # skip non-tensor / nested dict fields (messages, metadata, etc.)
            continue

    # Ensure labels are long dtype if present
    if "labels" in batch and batch["labels"].dtype != torch.long:
        batch["labels"] = batch["labels"].long()
    return batch


def main() -> None:
    args = build_args()
    device = torch.device(args.device)

    if not args.chat_jsonl.is_file():
        raise FileNotFoundError(f"Chat JSONL not found: {args.chat_jsonl}")

    records = load_jsonl(str(args.chat_jsonl), resolve_relative=True)
    if args.sample_limit > 0:
        records = records[: args.sample_limit]
    if not records:
        raise ValueError("No records loaded from chat JSONL; cannot run smoke test.")

    # Minimal TrainArguments to get tokenizer/template/model via SwiftSft
    train_args = TrainArguments(
        model=args.model,
        template="chatml",
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        dataset=["dummy"],  # ms-swift requires a placeholder
        train_type="lora",  # avoids full-state save; we won't step optimizer
    )
    sft = SwiftSft(train_args)
    model_device = next(sft.model.parameters()).device

    dataset = BaseCaptionDataset(
        base_records=records,
        template=sft.template,
        user_prompt="Answer the instruction.",
        emit_norm="none",
        json_format="standard",
        use_summary=False,
        bypass_prob=1.0,  # no aug
        seed=42,
        allow_empty=False,
    )
    sample = dataset[0]
    batch = to_batch(sample, model_device)

    if "labels" not in batch:
        raise RuntimeError("Encoded batch missing 'labels'; template likely mis-configured.")

    # Verify model is in train mode and has trainable parameters
    sft.model.train()
    trainable_params = [p for p in sft.model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found! Model may not be properly configured for training.")
    
    print(f"[INFO] Found {len(trainable_params)} trainable parameter tensors")
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"[INFO] Total trainable parameters: {total_trainable:,}")
    
    # Clear any existing gradients
    sft.model.zero_grad()
    
    # Forward pass
    outputs = sft.model(**batch)
    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
    
    if not torch.isfinite(loss):
        raise RuntimeError(f"Loss is not finite: {loss.item()}")
    
    print(f"[INFO] Forward pass completed. loss={loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Verify gradients were computed
    params_with_grad = [p for p in trainable_params if p.grad is not None]
    if not params_with_grad:
        raise RuntimeError("No gradients computed! Backward pass may have failed.")
    
    # Check for non-zero gradients
    non_zero_grads = [p for p in params_with_grad if p.grad.abs().max() > 0]
    if not non_zero_grads:
        raise RuntimeError("All gradients are zero! This may indicate a problem with the backward pass.")
    
    print(f"[INFO] Gradients computed for {len(params_with_grad)}/{len(trainable_params)} trainable parameters")
    print(f"[INFO] Non-zero gradients in {len(non_zero_grads)} parameters")
    
    # Compute gradient statistics
    grad_norms = [p.grad.norm().item() for p in params_with_grad if p.grad is not None]
    if grad_norms:
        print(f"[INFO] Gradient norm stats: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={sum(grad_norms)/len(grad_norms):.6f}")
    
    print(f"[OK] Forward+backward succeeded. loss={loss.item():.4f}")


if __name__ == "__main__":
    # Safer defaults for CUDA memory fragmentation in a quick smoke
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
