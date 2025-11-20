"""Online fusion dataset that mixes multiple JSONL sources per epoch."""

from __future__ import annotations

import copy
import random
import zlib
from pathlib import Path
from typing import Any, MutableMapping, Optional, Sequence, Tuple

from torch.utils.data import Dataset

from ..config.prompts import get_template_prompts
from .dense_caption import DenseCaptionDataset
from .fusion import (
    FusionConfig,
    DatasetSpec,
    prepare_record_for_dataset,
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
        augment_sources: Optional[Sequence[str]] = None,
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
        self._augment_sources = tuple(
            str(item) for item in augment_sources if item
        ) if augment_sources else None

        self.target_dataset = self._build_subdataset(
            spec=self.config.target,
            template=base_template,
            prompt_override=(user_prompt, system_prompt_dense),
            sample_limit=sample_limit,
        )
        self._aux_datasets = {}
        for source in self.config.sources:
            dataset = self._build_subdataset(
                spec=source,
                template=base_template,
                prompt_override=None,
                sample_limit=None,
            )
            self._aux_datasets[source.name] = dataset

        self.set_epoch(0)

    def _build_subdataset(
        self,
        spec: DatasetSpec,
        template: Any,
        prompt_override: Optional[Tuple[str, Optional[str]]],
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
        return DenseCaptionDataset(
            base_records=records,
            template=dataset_template,
            user_prompt=user_prompt,
            emit_norm=self.emit_norm,
            json_format=self.json_format,
            augmenter=self.augmenter,
            bypass_prob=self.bypass_prob,
            curriculum_state=self.curriculum_state,
            use_summary=self.use_summary,
            system_prompt_dense=system_prompt,
            system_prompt_summary=self.system_prompt_summary,
            seed=self._seed_for_dataset(spec.name),
            augment_sources=self._augment_sources,
        )

    @staticmethod
    def _clone_template(template: Any) -> Any:
        try:
            return copy.deepcopy(template)
        except Exception:
            return template

    @staticmethod
    def _load_records(path: Path, *, limit: Optional[int]) -> list[MutableMapping[str, Any]]:
        records = load_jsonl(str(path))
        if limit is not None and limit > 0:
            records = records[:limit]
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
            sampled = [
                rng.randrange(len(dataset)) for _ in range(quota)
            ]
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
        for dataset in (self.target_dataset, *self._aux_datasets.values()):
            if dataset.preprocessor is not None:
                dataset.preprocessor.curriculum_state = state
