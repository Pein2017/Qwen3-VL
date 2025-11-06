"""
Utility to auto-detect aligner module names for any Qwen3-VL model size.

This helps generate the correct modules_to_save list for different model variants.
"""

import torch.nn as nn
from typing import List

from . import get_logger

logger = get_logger(__name__)


def detect_aligner_modules(model: nn.Module) -> List[str]:
    """
    Auto-detect aligner module names from a Qwen3-VL model.

    Works for any model size (2B, 4B, 8B, etc.) by introspecting the model structure.

    Args:
        model: A Qwen3-VL model instance

    Returns:
        List of module names that should be in modules_to_save for aligner training

    Example:
        >>> from transformers import Qwen2_5_VLForConditionalGeneration
        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        >>> aligners = detect_aligner_modules(model)
        >>> print(aligners)
        ['model.visual.merger', 'model.visual.deepstack_merger_list.0', ...]
    """
    aligner_modules = []

    try:
        # Check if model has visual component
        if not hasattr(model, "model") or not hasattr(model.model, "visual"):
            return []

        visual = model.model.visual

        # Detect merger (single module)
        if hasattr(visual, "merger"):
            aligner_modules.append("model.visual.merger")

        # Detect deepstack_merger_list (ModuleList)
        if hasattr(visual, "deepstack_merger_list"):
            merger_list = visual.deepstack_merger_list
            if isinstance(merger_list, nn.ModuleList):
                for i in range(len(merger_list)):
                    aligner_modules.append(f"model.visual.deepstack_merger_list.{i}")

    except Exception as e:
        logger.warning(f"Could not auto-detect aligners: {e}")

    return aligner_modules


def print_model_structure(model: nn.Module, max_depth: int = 2) -> None:
    """
    Print model structure to help understand module hierarchy.

    Args:
        model: Model to inspect
        max_depth: Maximum depth to print (default 2)
    """

    def _print_recursive(module, prefix="", depth=0):
        if depth > max_depth:
            return

        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            num_params = sum(p.numel() for p in child.parameters(recurse=False))
            total_params = sum(p.numel() for p in child.parameters(recurse=True))

            logger.info(
                f"{'  ' * depth}{name:30s} | {type(child).__name__:20s} | "
                f"Direct: {num_params:>12,d} | Total: {total_params:>12,d}"
            )

            _print_recursive(child, full_name, depth + 1)

    logger.info("=" * 100)
    logger.info("Model Structure")
    logger.info("=" * 100)
    _print_recursive(model)
    logger.info("=" * 100)


def generate_modules_to_save_yaml(model: nn.Module) -> str:
    """
    Generate YAML snippet for modules_to_save.

    Args:
        model: Model to inspect

    Returns:
        YAML string ready to paste into config
    """
    aligners = detect_aligner_modules(model)

    if not aligners:
        return "  modules_to_save: []  # No aligners detected"

    yaml_lines = ["  modules_to_save:"]
    for module_name in aligners:
        yaml_lines.append(f"    - {module_name}")

    return "\n".join(yaml_lines)


if __name__ == "__main__":
    """
    Usage:
        python -m src.utils.auto_detect_aligners <model_name_or_path>
    
    Example:
        python -m src.utils.auto_detect_aligners Qwen/Qwen3-VL-4B-Instruct
        python -m src.utils.auto_detect_aligners output/stage_1_full_aligner_only/best/checkpoint-200
    """
    import sys

    if len(sys.argv) < 2:
        logger.info("Usage: python -m src.utils.auto_detect_aligners <model_path>")
        logger.info("\nExample:")
        logger.info(
            "  python -m src.utils.auto_detect_aligners Qwen/Qwen3-VL-4B-Instruct"
        )
        logger.info(
            "  python -m src.utils.auto_detect_aligners output/stage_1/checkpoint-100"
        )
        sys.exit(1)

    model_path = sys.argv[1]

    logger.info(f"Loading model from: {model_path}")
    logger.info("(This may take a moment...)\n")

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        import torch

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Keep on CPU for inspection
        )

        logger.info("\n" + "=" * 100)
        logger.info("DETECTED ALIGNER MODULES")
        logger.info("=" * 100)

        aligners = detect_aligner_modules(model)

        if aligners:
            logger.info("\nFound {} aligner module(s):".format(len(aligners)))
            for module_name in aligners:
                logger.info(f"  - {module_name}")

            logger.info("\n" + "=" * 100)
            logger.info("YAML CONFIG SNIPPET (copy to your config file)")
            logger.info("=" * 100)
            logger.info(generate_modules_to_save_yaml(model))
            logger.info("=" * 100)
        else:
            logger.info("No aligner modules detected!")

        # Optional: Print full structure
        if "--verbose" in sys.argv:
            logger.info("\n")
            print_model_structure(model, max_depth=2)

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
