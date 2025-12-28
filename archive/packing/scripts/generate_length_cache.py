#!/usr/bin/env python
"""Generate cached lengths for packing using a training config."""

import argparse
import os
from typing import Any, Mapping, Optional

from swift.llm.train.sft import SwiftSft
from tqdm import tqdm

from src.config.loader import ConfigLoader
from src.datasets import BaseCaptionDataset
from src.datasets.augmentation import ops as _register_ops  # noqa: F401
from src.datasets.augmentation.builder import build_compose_from_config
from src.datasets.fusion import FusionConfig
from src.datasets.length_cache import (
    LengthCache,
    build_cache_fingerprint,
    fingerprint_from_paths,
)
from src.datasets.unified_fusion_dataset import FusionCaptionDataset
from src.packing.grouped_packing import _resolve_length
from src.sft import get_logger

logger = get_logger(__name__)


def _build_augmenter(custom_cfg: Any) -> tuple[Optional[Any], float]:
    aug_cfg = custom_cfg.augmentation
    augmenter = None
    bypass_prob = float(custom_cfg.bypass_prob)
    if isinstance(aug_cfg, Mapping) and aug_cfg.get("enabled", True):
        augmenter = build_compose_from_config(dict(aug_cfg))
        bypass_prob = float(aug_cfg.get("bypass_prob", custom_cfg.bypass_prob))
    return augmenter, bypass_prob


def _build_datasets(train_args, training_cfg, template):
    custom_cfg = training_cfg.custom
    use_summary = bool(custom_cfg.use_summary)

    # Auto ROOT_IMAGE_DIR
    train_jsonl = custom_cfg.train_jsonl or custom_cfg.extra.get("jsonl")
    if not train_jsonl:
        raise ValueError("custom.train_jsonl is required")
    if os.environ.get("ROOT_IMAGE_DIR") in (None, ""):
        root_dir = os.path.abspath(os.path.dirname(train_jsonl))
        os.environ["ROOT_IMAGE_DIR"] = root_dir
        logger.info(f"Set ROOT_IMAGE_DIR={root_dir}")

    augmenter, bypass_prob = _build_augmenter(custom_cfg)

    curriculum_cfg = custom_cfg.augmentation_curriculum
    curriculum_state = curriculum_cfg if curriculum_cfg else None



    system_prompt_dense = getattr(train_args, "system_prompt", None)
    system_prompt_summary = getattr(custom_cfg, "system_prompt_summary", None)
    dataset_seed = 42

    if custom_cfg.fusion_config:
        fusion_config_obj = FusionConfig.from_file(custom_cfg.fusion_config)
        dataset = FusionCaptionDataset(
            fusion_config=fusion_config_obj,
            base_template=template,
            user_prompt=custom_cfg.user_prompt,
            emit_norm=custom_cfg.emit_norm,
            json_format=custom_cfg.json_format,
            augmenter=augmenter,
            bypass_prob=bypass_prob,
            curriculum_state=curriculum_state,
            use_summary=use_summary,
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=system_prompt_summary,
            seed=dataset_seed,
            sample_limit=custom_cfg.train_sample_limit or custom_cfg.sample_limit,
            split="train",
        )
    else:
        dataset = BaseCaptionDataset.from_jsonl(
            train_jsonl,
            template=template,
            user_prompt=custom_cfg.user_prompt,
            emit_norm=custom_cfg.emit_norm,
            json_format=custom_cfg.json_format,
            augmenter=augmenter,
            bypass_prob=bypass_prob,
            curriculum_state=curriculum_state,
            sample_limit=custom_cfg.train_sample_limit or custom_cfg.sample_limit,
            use_summary=use_summary,
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=system_prompt_summary,
            seed=dataset_seed,
        )
    return dataset


def _compute_lengths(dataset) -> dict[int, int]:
    lengths: dict[int, int] = {}
    dataset.set_epoch(0)
    dataset_len = len(dataset)
    import sys

    pbar = tqdm(
        range(dataset_len), desc="Computing lengths", mininterval=0.5, file=sys.stdout
    )
        for idx in pbar:
            if hasattr(dataset, "_schedule") and hasattr(dataset, "_record_pools"):
                dataset_name, base_idx = dataset._schedule[idx]
            else:
                dataset_name = getattr(dataset, "dataset_name", "dataset")
                perm = getattr(dataset, "_index_perm", list(range(dataset_len)))
                base_idx = perm[idx] if idx < len(perm) else idx
            sample_id = BaseCaptionDataset._make_sample_id(dataset_name, base_idx)
            row = dataset[idx]
            lengths[sample_id] = _resolve_length(row)
            if (idx + 1) % 100 == 0:
                pbar.refresh()
    finally:
        pbar.close()
    return lengths


def main():
    parser = argparse.ArgumentParser(description="Generate cached lengths for packing.")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--base-config", default=None, help="Optional base config YAML")
    parser.add_argument(
        "--cache-path", required=True, help="Where to write the cache JSON"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    cache_dir = os.path.dirname(os.path.abspath(args.cache_path))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    train_args, training_cfg = ConfigLoader.load_training_config(
        args.config, base_config_path=args.base_config
    )
    sft = SwiftSft(train_args)
    template = sft.template
    dataset = _build_datasets(train_args, training_cfg, template)

    lengths = _compute_lengths(dataset)
    template_id = getattr(train_args, "template", None) or getattr(
        training_cfg.template, "name", "template"
    )
    data_paths = [training_cfg.custom.train_jsonl]
    if training_cfg.custom.fusion_config:
        data_paths.append(training_cfg.custom.fusion_config)
    fingerprint = build_cache_fingerprint(
        augmentation_cfg=training_cfg.custom.augmentation
        if isinstance(training_cfg.custom.augmentation, Mapping)
        else None,
        template_id=str(template_id),
        data_paths=data_paths,
    )

    meta = {
        "data_fingerprint": fingerprint_from_paths(data_paths),
        "config": args.config,
        "base_config": args.base_config,
    }
    LengthCache(fingerprint=fingerprint, lengths=lengths, meta=meta).save(
        args.cache_path
    )
    logger.info("Length cache saved to %s (entries=%d)", args.cache_path, len(lengths))
    logger.info("Finished generating length cache.")


if __name__ == "__main__":
    main()
