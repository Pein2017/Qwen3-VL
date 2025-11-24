#!/usr/bin/env python3
"""Quick sanity check for fusion masking with shared vs cloned templates.

This script prints mask ratios for the unified fusion loader (shared template)
and demonstrates how a manually corrupted clone (jinja backend or train-all loss scale)
raises the unmasked token ratio.

Usage:
    conda run -n ms python scripts/debug_fusion_template_clone.py
"""

from __future__ import annotations

import copy
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol

# Define repo root first
REPO_ROOT = Path(__file__).resolve().parent.parent

# Keep ms-swift ahead of project modules (prevents huggingface.datasets shadowing)
# Use relative path: ms-swift is typically at ../ms-swift from repo root
# If not found, try absolute path as fallback (for environments where it's installed elsewhere)
MS_SWIFT_PATH = REPO_ROOT.parent / "ms-swift"
if not MS_SWIFT_PATH.exists():
    # Fallback: try common installation location
    MS_SWIFT_PATH = Path("/data/ms-swift")
if MS_SWIFT_PATH.exists() and str(MS_SWIFT_PATH) not in sys.path:
    sys.path.insert(0, str(MS_SWIFT_PATH))

from swift.llm import get_template  # noqa: E402
from swift.llm.model.register import get_model_tokenizer  # noqa: E402
from swift.plugin.loss_scale.loss_scale import TrainAllLossScale  # noqa: E402

SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src.config.prompts import get_template_prompts  # noqa: E402
from src.datasets.dense_caption import BaseCaptionDataset  # noqa: E402
from src.datasets.fusion import FusionConfig  # noqa: E402
from src.datasets.unified_fusion_dataset import FusionCaptionDataset  # noqa: E402
from src.datasets.utils import load_jsonl  # noqa: E402


class DatasetProtocol(Protocol):
    """Protocol for dataset-like objects."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict[str, Any]: ...


def _mask_ratio(labels: Iterable[int]) -> tuple[float, int, int]:
    labels_list = list(labels)
    total = len(labels_list)
    if total == 0:
        return 0.0, 0, 0
    unmasked = sum(1 for v in labels_list if v != -100)
    return unmasked / total, total, unmasked


def summarize_dataset(name: str, dataset: DatasetProtocol) -> None:
    schedule = getattr(dataset, "_schedule", None)
    ratios: list[float] = []
    per_source: dict[str | None, list[float]] = defaultdict(list)

    print(f"\n=== {name} (len={len(dataset)}) ===")
    for i in range(len(dataset)):
        source = schedule[i][0] if schedule and i < len(schedule) else None
        sample = dataset[i]
        labels: Any = sample.get("labels") or []
        if hasattr(labels, "tolist"):
            labels = labels.tolist()  # type: ignore[attr-defined]
        ratio, _, _ = _mask_ratio(labels)
        ratios.append(ratio)
        per_source[source].append(ratio)

    if not ratios:
        print("no samples")
        return

    avg = sum(ratios) / len(ratios)
    print(
        f"overall mask ratio: {avg * 100:.1f}% (min={min(ratios) * 100:.1f}%, max={max(ratios) * 100:.1f}%)"
    )
    for source, values in per_source.items():
        if not values:
            continue
        src_avg = sum(values) / len(values)
        print(
            f"  - {source or 'unknown'}: {src_avg * 100:.1f}% over {len(values)} samples"
        )


def demo_corrupted_clone(base_template: Any, fusion_cfg: FusionConfig) -> None:
    record = load_jsonl(str(fusion_cfg.target.train_jsonl), resolve_relative=True)[0]
    system_prompt, user_prompt = get_template_prompts(fusion_cfg.target.template)

    def run(label: str, template_obj: Any) -> None:
        ds = BaseCaptionDataset(
            base_records=[record],
            template=template_obj,
            user_prompt=user_prompt or "",
            emit_norm="norm1000",
            json_format="standard",
            augmenter=None,
            preprocessor=None,
            bypass_prob=1.0,
            curriculum_state=None,
            use_summary=False,
            system_prompt_dense=system_prompt,
            system_prompt_summary=None,
            seed=123,
            dataset_name=label,
        )
        sample = ds[0]
        labels: Any = sample.get("labels") or []
        if hasattr(labels, "tolist"):
            labels = labels.tolist()  # type: ignore[attr-defined]
        ratio, total, unmasked = _mask_ratio(labels)
        print(
            f"  - {label}: backend={getattr(template_obj, 'template_backend', None)}, "
            f"loss_scale={type(getattr(template_obj, 'loss_scale', object())).__name__}, "
            f"unmasked={unmasked}/{total} ({ratio * 100:.1f}%)"
        )

    print("\n=== Single-record masking sanity check ===")
    run("shared-template (expected)", base_template)

    jinja_clone = copy.deepcopy(base_template)
    setattr(jinja_clone, "template_backend", "jinja")
    run("corrupted clone (jinja backend)", jinja_clone)

    all_loss_clone = copy.deepcopy(base_template)
    all_loss_clone.loss_scale = TrainAllLossScale()  # type: ignore[attr-defined]
    run("corrupted clone (train-all loss scale)", all_loss_clone)


def main() -> None:
    fusion_path = REPO_ROOT / "configs/fusion/bbu_with_lvis_tiny.yaml"
    fusion_cfg = FusionConfig.from_file(str(fusion_path))

    target_dir = fusion_cfg.target.train_jsonl.parent.resolve()
    os.environ.setdefault("ROOT_IMAGE_DIR", str(target_dir))

    _, processor = get_model_tokenizer(
        model_id_or_path="model_cache/models/Qwen/Qwen3-VL-8B-Instruct",
        model_type="qwen3_vl",
        load_model=False,
    )
    template = get_template(
        "qwen3_vl", processor, max_length=12000, loss_scale="last_round"
    )
    template.mode = "train"

    print(
        f"base template id={id(template)}, backend={template.template_backend}, loss_scale={type(template.loss_scale).__name__}"
    )

    unified = FusionCaptionDataset(
        fusion_config=fusion_cfg,
        base_template=template,
        user_prompt="",
        emit_norm="norm1000",
        json_format="standard",
        augmenter=None,
        bypass_prob=1.0,
        curriculum_state=None,
        use_summary=False,
        system_prompt_dense=None,
        system_prompt_summary=None,
        seed=42,
        shuffle=True,
        sample_limit=None,
        split="train",
    )

    print(
        f"unified template id={id(unified.template)} (shared? {id(unified.template) == id(template)})"
    )

    summarize_dataset("UnifiedFusionDataset", unified)
    demo_corrupted_clone(template, fusion_cfg)


if __name__ == "__main__":
    main()
