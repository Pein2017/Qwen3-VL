"""Online fusion dataset that mixes multiple JSONL sources per epoch."""

from __future__ import annotations

import copy
import random
import math
import zlib
import os
from pathlib import Path
from typing import Any, MutableMapping, Optional, Tuple

from torch.utils.data import Dataset

from ..config.prompts import get_template_prompts
from .dense_caption import DenseCaptionDataset
from .fusion import FusionConfig, prepare_record_for_dataset
from .fusion_types import DatasetSpec
from .preprocessors import (
    ObjectCapPreprocessor,
    SmartResizePreprocessor,
    smart_resize_params_from_env,
)
from .utils import load_jsonl


class MultiSourceFusionDataset(Dataset):
    """Dataset that fuses a target dataset with auxiliary sources per epoch."""

    def __init__(
        self,
        *,
        fusion_config: FusionConfig,
        base_template: Any,
        user_prompt: str,
        emit_norm: str,
        json_format: str,
        augmenter: Optional[Any],
        bypass_prob: float,
        curriculum_state: Optional[MutableMapping[str, Any]],
        use_summary: bool,
        system_prompt_dense: Optional[str],
        system_prompt_summary: Optional[str],
        seed: int = 42,
        shuffle: bool = True,
        sample_limit: Optional[int] = None,
    ):
        self.config = fusion_config
        self.emit_norm = emit_norm
        self.json_format = json_format
        self.augmenter = augmenter
        self.bypass_prob = bypass_prob
        self.curriculum_state = curriculum_state
        self.use_summary = use_summary
        self.system_prompt_summary = system_prompt_summary
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0
        self._schedule: list[tuple[str, int]] = []
        self._aux_counts: dict[str, int] = {}
        self._template = base_template
        self._curriculum_support: dict[str, bool] = {}

        self.target_dataset = self._build_subdataset(
            spec=self.config.target,
            template=base_template,
            prompt_override=(user_prompt, system_prompt_dense),
            allow_augmentation=self.config.target.supports_augmentation,
            allow_curriculum=self.config.target.supports_curriculum,
            sample_limit=sample_limit,
        )
        self._curriculum_support[self.config.target.name] = (
            self.config.target.supports_curriculum
        )
        self._aux_datasets = {}
        self._aux_poly_indices: dict[str, list[int]] = {}
        for source in self.config.sources:
            dataset = self._build_subdataset(
                spec=source,
                template=base_template,
                prompt_override=None,
                allow_augmentation=source.supports_augmentation,
                allow_curriculum=source.supports_curriculum,
                sample_limit=None,
            )
            self._aux_datasets[source.name] = dataset
            self._curriculum_support[source.name] = source.supports_curriculum
            self._aux_poly_indices[source.name] = self._extract_poly_indices(dataset)

        self.set_epoch(0)

    def _build_subdataset(
        self,
        spec: DatasetSpec,
        template: Any,
        prompt_override: Optional[Tuple[str, Optional[str]]],
        allow_augmentation: bool,
        allow_curriculum: bool,
        sample_limit: Optional[int],
    ) -> DenseCaptionDataset:
        raw_records = self._load_records(spec.train_jsonl, limit=sample_limit)
        system_prompt, registry_user = get_template_prompts(spec.template)
        user_prompt = (
            prompt_override[0] if prompt_override else registry_user
        )
        system_prompt = (
            prompt_override[1] if prompt_override and prompt_override[1] else system_prompt
        )
        records = [
            prepare_record_for_dataset(record, spec) for record in raw_records
        ]
        dataset_template = self._clone_template(template)
        use_augmenter = self.augmenter if allow_augmentation else None
        bypass_prob = self.bypass_prob if allow_augmentation else 0.0
        curriculum_state = (
            self.curriculum_state if allow_augmentation and allow_curriculum else None
        )
        preprocessor = (
            ObjectCapPreprocessor(spec.max_objects_per_image)
            if spec.max_objects_per_image
            else None
        )
        return DenseCaptionDataset(
            base_records=records,
            template=dataset_template,
            user_prompt=user_prompt,
            emit_norm=self.emit_norm,
            json_format=self.json_format,
            augmenter=use_augmenter,
            preprocessor=preprocessor,
            bypass_prob=bypass_prob,
            curriculum_state=curriculum_state,
            use_summary=self.use_summary,
            system_prompt_dense=system_prompt,
            system_prompt_summary=self.system_prompt_summary,
            seed=self._seed_for_dataset(spec.name),
        )

    @staticmethod
    def _clone_template(template: Any) -> Any:
        try:
            return copy.deepcopy(template)
        except Exception:
            return template

    @staticmethod
    def _load_records(path: Path, *, limit: Optional[int]) -> list[MutableMapping[str, Any]]:
        records = load_jsonl(str(path), resolve_relative=True)
        if limit is not None and limit > 0:
            records = records[:limit]

        guard_params = smart_resize_params_from_env()
        if guard_params:
            output_dir_env = os.getenv("SMART_RESIZE_GUARD_OUTPUT_DIR")
            output_dir = (
                Path(output_dir_env).resolve()
                if output_dir_env
                else (path.parent / "_smart_resized_images")
            )
            guard = SmartResizePreprocessor(
                params=guard_params,
                jsonl_dir=path.parent,
                output_dir=output_dir,
                write_images=True,
            )
            guarded: list[MutableMapping[str, Any]] = []
            for rec in records:
                updated = guard(rec)
                if updated is not None:
                    guarded.append(updated)
            records = guarded
        return records

    def _seed_for_dataset(self, name: str) -> int:
        return (self.seed + zlib.adler32(name.encode())) & 0xFFFFFFFF

    def _seed_for_epoch(self, epoch: int) -> int:
        base_seed = self.seed & 0xFFFFFFFF
        mixed = (base_seed ^ ((int(epoch) + 1) * 0x9E3779B1)) & 0xFFFFFFFF
        return mixed

    def _build_schedule(self) -> None:
        rng = random.Random(self._seed_for_epoch(self._epoch))
        self.target_dataset.set_epoch(self._epoch)
        self._schedule = []
        for idx in range(len(self.target_dataset)):
            self._schedule.append((self.config.target.name, idx))

        self._aux_counts = {}
        for source in self.config.sources:
            dataset = self._aux_datasets[source.name]
            dataset.set_epoch(self._epoch)
            quota = round(source.ratio * len(self.target_dataset))
            if quota <= 0 or len(dataset) == 0:
                self._aux_counts[source.name] = 0
                continue
            self._aux_counts[source.name] = quota
            sampled: list[int] = []

            poly_ratio = getattr(source, "poly_min_ratio", None)
            poly_indices = self._aux_poly_indices.get(source.name) or []
            if poly_ratio and poly_indices:
                poly_quota = min(
                    quota, max(1, math.ceil(quota * float(poly_ratio)))
                )
                sampled.extend(rng.choice(poly_indices) for _ in range(poly_quota))
            else:
                poly_quota = 0

            remaining = quota - poly_quota
            if remaining > 0:
                sampled.extend(rng.randrange(len(dataset)) for _ in range(remaining))

            self._schedule.extend(
                (source.name, idx) for idx in sampled
            )

        if self.shuffle:
            rng.shuffle(self._schedule)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._build_schedule()

    def __len__(self) -> int:
        return len(self._schedule)

    def __getitem__(self, index: int) -> Any:
        dataset_name, idx = self._schedule[index]
        if dataset_name == self.config.target.name:
            return self.target_dataset[idx]
        return self._aux_datasets[dataset_name][idx]

    @property
    def aux_quota(self) -> dict[str, int]:
        return dict(self._aux_counts)

    def set_curriculum_state(self, state: MutableMapping[str, Any]) -> None:
        self.curriculum_state = state
        if self._curriculum_support.get(self.config.target.name):
            self._assign_curriculum(self.target_dataset, state)
        for name, dataset in self._aux_datasets.items():
            if self._curriculum_support.get(name):
                self._assign_curriculum(dataset, state)

    @staticmethod
    def _extract_poly_indices(dataset: DenseCaptionDataset) -> list[int]:
        indices: list[int] = []
        for idx, rec in enumerate(dataset.base_records):
            objects = rec.get("objects") or []
            if any("poly" in obj for obj in objects):
                indices.append(idx)
        return indices

    @staticmethod
    def _assign_curriculum(dataset: DenseCaptionDataset, state: MutableMapping[str, Any]) -> None:
        if dataset.preprocessor is not None:
            dataset.preprocessor.curriculum_state = state
