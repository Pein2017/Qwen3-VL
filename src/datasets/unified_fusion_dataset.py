"""Unified fusion dataset that concatenates JSONL files and uses a single template.

This avoids template cloning issues by using one dataset with one template.
Records are tagged with their source dataset for prompt selection.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, MutableMapping, Optional, cast

from torch.utils.data import get_worker_info

from ..config.prompts import USER_PROMPT_SUMMARY, get_template_prompts
from .builders import JSONLinesBuilder
from .contracts import validate_conversation_record
from .dense_caption import LAST_SAMPLE_DEBUG, BaseCaptionDataset
from .fusion import FusionConfig
from .fusion_types import DatasetSpec
from .preprocessors import AugmentationPreprocessor, ObjectCapPreprocessor
from .utils import load_jsonl


@dataclass(frozen=True)
class _PromptResolution:
    user: str
    system: Optional[str]
    source: Literal["default", "domain", "dataset"]


@dataclass
class _DatasetPolicy:
    spec: DatasetSpec
    prompts: _PromptResolution
    augmentation_enabled: bool
    curriculum_enabled: bool
    max_objects_per_image: Optional[int]
    seed: Optional[int]


class FusionCaptionDataset(BaseCaptionDataset):
    """Fusion dataset that concatenates multiple JSONL sources with a unified template."""

    def __init__(
        self,
        *,
        fusion_config: FusionConfig,
        base_template: Any,
        user_prompt: str,
        emit_norm: Literal["none", "norm100", "norm1000"],
        json_format: Literal["standard"],
        augmenter: Optional[Any],
        bypass_prob: float,
        curriculum_state: Optional[MutableMapping[str, Any]],
        use_summary: bool,
        system_prompt_dense: Optional[str],
        system_prompt_summary: Optional[str],
        seed: int = 42,
        shuffle: bool = True,
        sample_limit: Optional[int] = None,
        split: Literal["train", "eval"] = "train",
        target_eval_jsonl: Optional[str] = None,
        include_source_eval: bool = True,
    ):
        self._fusion_config = fusion_config
        self._split: Literal["train", "eval"] = split
        self._augmenter = augmenter
        self.bypass_prob = float(bypass_prob)
        self.curriculum_state = curriculum_state
        self._shuffle = bool(shuffle)
        self._include_source_eval = bool(include_source_eval)
        self._sample_limit = sample_limit
        self._epoch = 0
        self._schedule: list[tuple[str, int]] = []
        self._epoch_counts: dict[str, int] = {}
        self._policies: dict[str, _DatasetPolicy] = {}
        self._record_pools: dict[str, list[dict[str, Any]]] = {}
        self._preprocessors_aug: dict[str, AugmentationPreprocessor] = {}
        self._preprocessors_cap: dict[str, ObjectCapPreprocessor] = {}
        self.epoch_plan: dict[str, dict[str, Any]] = {}
        self._hard_sample_plan: dict[str, Any] | None = None

        self._dataset_order = [
            fusion_config.target.name,
            *[src.name for src in fusion_config.sources],
        ]
        self._target_name = fusion_config.target.name

        default_system_prompt = system_prompt_dense
        # Load pools for the selected split
        target_path: Optional[Path]
        if split == "eval":
            if target_eval_jsonl:
                target_path = Path(target_eval_jsonl)
            else:
                target_path = fusion_config.target.val_jsonl
        else:
            target_path = fusion_config.target.train_jsonl
        if target_path is None:
            raise ValueError("Target split is required for FusionCaptionDataset")

        target_records = self._load_records(
            target_path, limit=sample_limit if split == "train" else sample_limit
        )
        if not target_records:
            raise ValueError("FusionCaptionDataset requires at least one target record")
        self._record_pools[self._target_name] = [
            self._annotate_record(rec, fusion_config.target) for rec in target_records
        ]

        # Build prompt/policy for target
        target_prompts = self._resolve_prompts(
            fusion_config.target,
            default_user_prompt=user_prompt,
            default_system_prompt=default_system_prompt,
        )
        self._policies[self._target_name] = _DatasetPolicy(
            spec=fusion_config.target,
            prompts=target_prompts,
            augmentation_enabled=fusion_config.target.supports_augmentation,
            curriculum_enabled=fusion_config.target.supports_curriculum,
            max_objects_per_image=fusion_config.target.max_objects_per_image,
            seed=fusion_config.target.seed,
        )

        # Sources
        for source in fusion_config.sources:
            policy = self._resolve_prompts(
                source,
                default_user_prompt=user_prompt,
                default_system_prompt=default_system_prompt,
            )
            self._policies[source.name] = _DatasetPolicy(
                spec=source,
                prompts=policy,
                augmentation_enabled=source.supports_augmentation,
                curriculum_enabled=source.supports_curriculum,
                max_objects_per_image=source.max_objects_per_image,
                seed=source.seed,
            )

            # Skip loading for train split when ratio is zero
            if split == "train" and source.ratio <= 0:
                self._record_pools[source.name] = []
                continue

            if split == "eval":
                if not include_source_eval or source.val_jsonl is None:
                    self._record_pools[source.name] = []
                    continue
                source_path = source.val_jsonl
            else:
                source_path = source.train_jsonl

            source_records = self._load_records(
                source_path,
                limit=None,  # Keep auxiliary pools intact
            )
            if split == "train" and source.ratio > 0 and not source_records:
                raise ValueError(
                    f"Fusion source '{source.name}' is empty while ratio={source.ratio}"
                )
            self._record_pools[source.name] = [
                self._annotate_record(rec, source) for rec in source_records
            ]

        # Initialize parent BaseCaptionDataset with a single template instance.
        all_records = [rec for pool in self._record_pools.values() for rec in pool]
        super().__init__(
            base_records=all_records,
            template=base_template,
            user_prompt=user_prompt,
            emit_norm=emit_norm,
            json_format=json_format,
            augmenter=None,
            preprocessor=None,
            bypass_prob=bypass_prob,
            curriculum_state=curriculum_state,
            use_summary=use_summary,
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=system_prompt_summary,
            seed=seed,
            dataset_name=self._target_name,
            allow_empty=True,
        )

        # Build preprocessors after parent init sets base attributes.
        for name, policy in self._policies.items():
            if (
                policy.augmentation_enabled
                and self._split == "train"
                and self._augmenter is not None
            ):
                self._preprocessors_aug[name] = AugmentationPreprocessor(
                    augmenter=self._augmenter,
                    bypass_prob=self.bypass_prob,
                    curriculum_state=(
                        self.curriculum_state if policy.curriculum_enabled else None
                    ),
                )
            if policy.max_objects_per_image is not None:
                self._preprocessors_cap[name] = ObjectCapPreprocessor(
                    policy.max_objects_per_image
                )

        self.set_epoch(0)

    @staticmethod
    def _annotate_record(
        record: MutableMapping[str, Any], spec: DatasetSpec
    ) -> dict[str, Any]:
        """Annotate record with source dataset metadata."""
        annotated = copy.deepcopy(dict(record))
        metadata = annotated.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["_fusion_source"] = spec.name
        metadata["_fusion_template"] = spec.template
        metadata["_fusion_domain"] = spec.domain
        annotated["metadata"] = metadata
        return annotated

    @staticmethod
    def _load_records(
        path: Path, *, limit: Optional[int]
    ) -> list[MutableMapping[str, Any]]:
        records = load_jsonl(str(path), resolve_relative=True)
        records = [cast(MutableMapping[str, Any], rec) for rec in records]
        if limit is not None and limit > 0:
            records = records[:limit]
        validated: list[MutableMapping[str, Any]] = []
        for idx, record in enumerate(records):
            try:
                validated.append(
                    cast(
                        MutableMapping[str, Any],
                        copy.deepcopy(validate_conversation_record(record)),
                    )
                )
            except ValueError as exc:
                raise ValueError(f"Record {idx} in {path} is invalid: {exc}") from exc
        return validated

    @staticmethod
    def _mix_seed(base: int, *parts: int) -> int:
        seed_val = base & 0xFFFFFFFF
        for offset, part in enumerate(parts):
            seed_val ^= ((int(part) + offset + 1) * 0x9E3779B1) & 0xFFFFFFFF
        return seed_val & 0xFFFFFFFF

    def _resolve_prompts(
        self,
        spec: DatasetSpec,
        *,
        default_user_prompt: str,
        default_system_prompt: Optional[str],
    ) -> _PromptResolution:
        domain_system, domain_user = get_template_prompts(spec.template)
        user_prompt = spec.prompt_user or domain_user or default_user_prompt
        system_prompt = spec.prompt_system or domain_system or default_system_prompt

        if spec.prompt_user or spec.prompt_system:
            source = "dataset"
        elif domain_user or domain_system:
            source = "domain"
        else:
            source = "default"
        return _PromptResolution(user=user_prompt, system=system_prompt, source=source)

    def _build_train_schedule(self) -> None:
        target_pool = self._record_pools.get(self._target_name, [])
        target_count_raw = len(target_pool)
        plan = self._hard_sample_plan or {}
        target_epoch_size = int(plan.get("target_epoch_size") or target_count_raw)
        target_weights_map = plan.get("weights") if isinstance(plan, dict) else None
        rng_target = random.Random(
            self._mix_seed(
                self.seed,
                self._epoch,
                self._policies[self._target_name].seed or 0,
                0xA1,
            )
        )
        target_indices_all = list(range(target_count_raw))
        if target_weights_map:
            target_weights = [float(target_weights_map.get(i, 1.0)) for i in target_indices_all]
            sampled_target = rng_target.choices(target_indices_all, weights=target_weights, k=target_epoch_size)
        else:
            target_indices = list(target_indices_all)
            if self._shuffle and len(target_indices) > 1:
                rng_target.shuffle(target_indices)
            if target_epoch_size == len(target_indices):
                sampled_target = target_indices
            elif target_epoch_size < len(target_indices):
                sampled_target = target_indices[:target_epoch_size]
            else:
                extra = rng_target.choices(target_indices, k=target_epoch_size - len(target_indices))
                sampled_target = target_indices + extra

        schedule: list[tuple[str, int]] = [
            (self._target_name, idx) for idx in sampled_target
        ]
        counts: dict[str, int] = {self._target_name: len(sampled_target)}

        for source in self._fusion_config.sources:
            policy = self._policies[source.name]
            pool = self._record_pools.get(source.name, [])
            if source.ratio > 0 and not pool:
                raise ValueError(
                    f"Fusion source '{source.name}' is empty while ratio={source.ratio}"
                )
            quota = round(source.ratio * len(sampled_target))
            if quota <= 0 or not pool:
                counts[source.name] = 0
                continue

            rng_source = random.Random(
                self._mix_seed(self.seed, self._epoch, policy.seed or 0, 0xB7)
            )
            sampled_indices = [rng_source.randrange(len(pool)) for _ in range(quota)]
            counts[source.name] = quota
            schedule.extend((source.name, idx) for idx in sampled_indices)

        if self._shuffle and len(schedule) > 1:
            rng_shuffle = random.Random(self._mix_seed(self.seed, self._epoch, 0xC3))
            rng_shuffle.shuffle(schedule)

        self._schedule = schedule
        self._epoch_counts = counts
        self._update_epoch_plan()

    def _build_eval_schedule(self) -> None:
        schedule: list[tuple[str, int]] = []
        counts: dict[str, int] = {}

        for name in self._dataset_order:
            pool = self._record_pools.get(name, [])
            counts[name] = len(pool)
            schedule.extend((name, idx) for idx in range(len(pool)))

        self._schedule = schedule
        self._epoch_counts = counts
        self._update_epoch_plan(eval_mode=True)

    def _update_epoch_plan(self, eval_mode: bool = False) -> None:
        plan: dict[str, dict[str, Any]] = {}
        for name in self._dataset_order:
            policy = self._policies.get(name)
            if policy is None:
                continue
            plan[name] = {
                "count": self._epoch_counts.get(name, 0),
                "augmentation": bool(
                    policy.augmentation_enabled
                    and self._augmenter is not None
                    and not eval_mode
                ),
                "curriculum": bool(
                    policy.curriculum_enabled
                    and self.curriculum_state is not None
                    and not eval_mode
                ),
                "max_objects_per_image": policy.max_objects_per_image,
                "prompt_source": policy.prompts.source,
            }
        self.epoch_plan = plan

    def set_hard_sample_plan(self, plan: Optional[MutableMapping[str, Any]]) -> None:
        self._hard_sample_plan = dict(plan) if plan is not None else None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._rng = random.Random(self._seed_for_epoch(self._epoch))
        if self._split == "eval":
            self._build_eval_schedule()
        else:
            self._build_train_schedule()

    def __len__(self) -> int:
        return len(self._schedule)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not self._schedule:
            raise IndexError("FusionCaptionDataset is empty")

        dataset_name, base_idx = self._schedule[index % len(self._schedule)]
        pool = self._record_pools.get(dataset_name, [])
        record = copy.deepcopy(pool[base_idx])
        policy = self._policies[dataset_name]

        worker = get_worker_info()
        seed_local = self._rng.randrange(0, 2**32 - 1)
        if worker is not None:
            seed_local ^= ((worker.id + 1) * 0xC2B2AE35) & 0xFFFFFFFF
        rng_local = random.Random(seed_local & 0xFFFFFFFF)

        allow_augmentation = (
            self._split == "train"
            and policy.augmentation_enabled
            and self._augmenter is not None
        )
        allow_curriculum = (
            allow_augmentation and policy.curriculum_enabled and self.curriculum_state
        )

        was_augmented = False
        cap_applied = False
        objects_before = len(record.get("objects") or [])

        aug_pre = self._preprocessors_aug.get(dataset_name)
        if allow_augmentation and aug_pre is not None:
            if hasattr(aug_pre, "rng"):
                aug_pre.rng = rng_local
            if hasattr(aug_pre, "curriculum_state"):
                try:
                    aug_pre.curriculum_state = (
                        self.curriculum_state if allow_curriculum else None
                    )
                except Exception:
                    pass
            processed = aug_pre(record)
            if processed is None:
                raise ValueError(
                    "Preprocessor removed the record; dataset does not duplicate samples"
                )
            record = processed
            was_augmented = True

        cap_pre = self._preprocessors_cap.get(dataset_name)
        if cap_pre is not None and policy.max_objects_per_image is not None:
            if hasattr(cap_pre, "rng"):
                cap_pre.rng = rng_local
            capped = cap_pre(record)
            if capped is None:
                raise ValueError(
                    "Preprocessor removed the record; dataset does not duplicate samples"
                )
            record = capped
            objects_after = len(record.get("objects") or [])
            cap_applied = objects_after < objects_before or (
                objects_before > policy.max_objects_per_image
            )
        else:
            objects_after = len(record.get("objects") or [])

        # Build conversation
        mode = self.mode
        prompts = policy.prompts
        system_prompt = prompts.system
        if mode == "summary" and self.system_prompt_summary is not None:
            system_prompt = self.system_prompt_summary
        builder = JSONLinesBuilder(
            user_prompt=USER_PROMPT_SUMMARY if mode == "summary" else prompts.user,
            emit_norm=self.emit_norm,
            mode=mode,
            json_format=self.json_format,
        )
        merged = builder.build_many([record])
        conversation_messages = copy.deepcopy(merged.get("messages", []) or [])

        if system_prompt:
            conversation_messages = [
                {"role": "system", "content": system_prompt},
                *conversation_messages,
            ]

        original_system = getattr(self.template, "system", None)
        try:
            if system_prompt:
                try:
                    self.template.system = system_prompt
                except Exception:
                    pass
            encoded = self.template.encode(merged, return_length=True)
        finally:
            if system_prompt is not None and original_system is not None:
                try:
                    self.template.system = original_system
                except Exception:
                    pass

        encoded["messages"] = conversation_messages
        for key in ("assistant_payload", "objects", "metadata"):
            if key in merged:
                encoded[key] = copy.deepcopy(merged[key])

        try:
            max_poly = 0
            for obj in record.get("objects") or []:
                if "poly_points" in obj:
                    max_poly = max(max_poly, int(obj.get("poly_points") or 0))
            info = {
                "dataset": dataset_name,
                "base_idx": base_idx,
                "objects": objects_after,
                "max_poly_points": max_poly,
                "width": record.get("width"),
                "height": record.get("height"),
                "mode": mode,
                "prompt_source": prompts.source,
                "augmentation_enabled": was_augmented,
                "object_cap_applied": cap_applied,
                "object_cap_limit": policy.max_objects_per_image,
            }
            input_ids = encoded.get("input_ids")
            if input_ids is not None and hasattr(input_ids, "__len__"):
                try:
                    info["input_length"] = len(input_ids)
                except Exception:
                    pass
            sample_id = self._make_sample_id(dataset_name, base_idx)
            encoded["sample_id"] = sample_id
            encoded["dataset"] = dataset_name
            encoded["base_idx"] = base_idx
            self.last_sample_debug = info
            LAST_SAMPLE_DEBUG.update(info)
        except Exception:
            pass

        return encoded

    @property
    def aux_quota(self) -> dict[str, int]:
        """Per-epoch auxiliary sample counts for debugging/telemetry."""
        return {k: v for k, v in self._epoch_counts.items() if k != self._target_name}

    def set_curriculum_state(self, state: MutableMapping[str, Any]) -> None:
        self.curriculum_state = state
        for name, aug in self._preprocessors_aug.items():
            policy = self._policies.get(name)
            if policy and policy.curriculum_enabled:
                try:
                    aug.curriculum_state = state
                except Exception:
                    pass

    def make_target_eval_dataset(self, *, mine_clean: bool = False) -> BaseCaptionDataset:
        policy = self._policies[self._target_name]
        use_aug = policy.augmentation_enabled and self._augmenter is not None and not mine_clean
        return BaseCaptionDataset(
            base_records=self._record_pools[self._target_name],
            template=self.template,
            user_prompt=policy.prompts.user,
            emit_norm=self.emit_norm,
            json_format=self.json_format,
            augmenter=self._augmenter if use_aug else None,
            bypass_prob=0.0 if mine_clean else self.bypass_prob,
            curriculum_state=None,
            use_summary=self.use_summary,
            system_prompt_dense=self.system_prompt_dense,
            system_prompt_summary=self.system_prompt_summary,
            seed=self.seed,
            dataset_name=self._target_name,
            allow_empty=False,
        )


# Backward compatibility alias
UnifiedFusionDataset = FusionCaptionDataset
