"""Unified fusion dataset that concatenates JSONL files and uses a single template.

This avoids template cloning issues by using one dataset with one template.
Records are tagged with their source dataset for prompt selection.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Optional, cast

from torch.utils.data import get_worker_info

from src.utils import get_logger

from ..config.prompts import USER_PROMPT_SUMMARY, get_template_prompts
from .builders import JSONLinesBuilder
from .contracts import validate_conversation_record
from .dense_caption import LAST_SAMPLE_DEBUG, BaseCaptionDataset
from .fusion import FusionConfig, _compute_target_quotas
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

    _logger = get_logger(__name__)

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
        preprocessor: Optional[Any] = None,
        use_summary: bool,
        system_prompt_dense: Optional[str],
        system_prompt_summary: Optional[str],
        seed: int = 42,
        shuffle: bool = True,
        sample_limit: Optional[int] = None,
        split: Literal["train", "eval"] = "train",
        target_eval_jsonl: Optional[str] = None,
        include_source_eval: bool = False,
    ):
        self._fusion_config = fusion_config
        self._split: Literal["train", "eval"] = split
        self._augmenter = augmenter
        self.bypass_prob = float(bypass_prob)
        self.curriculum_state = curriculum_state
        self._shuffle = bool(shuffle)
        self._include_source_eval = False  # target-only eval per policy
        self._sample_limit = sample_limit
        self._epoch = 0
        self._schedule: list[tuple[str, int]] = []
        self._epoch_counts: dict[str, int] = {}
        self._policies: dict[str, _DatasetPolicy] = {}
        self._record_pools: dict[str, list[dict[str, Any]]] = {}
        self._preprocessors_aug: dict[str, AugmentationPreprocessor] = {}
        self._preprocessors_cap: dict[str, ObjectCapPreprocessor] = {}
        self.epoch_plan: dict[str, dict[str, Any]] = {}

        self._target_names = [t.name for t in fusion_config.targets]
        self._dataset_order = [
            *self._target_names,
            *[src.name for src in fusion_config.sources],
        ]
        self._primary_target = self._target_names[0]

        default_system_prompt = system_prompt_dense
        # Load pools for the selected split
        if split == "eval" and target_eval_jsonl is not None:
            override_target_path = Path(target_eval_jsonl)
        else:
            override_target_path = None

        for target in fusion_config.targets:
            if split == "eval":
                target_path = override_target_path or target.val_jsonl
            else:
                target_path = target.train_jsonl
            if target_path is None:
                raise ValueError(f"Target '{target.name}' split is required")

            target_records = self._load_records(
                target_path, limit=sample_limit if split == "train" else sample_limit
            )
            if not target_records:
                raise ValueError(
                    f"FusionCaptionDataset requires at least one record for target '{target.name}'"
                )
            self._record_pools[target.name] = [
                self._annotate_record(rec, target) for rec in target_records
            ]

            # Build prompt/policy for target
            target_prompts = self._resolve_prompts(
                target,
                default_user_prompt=user_prompt,
                default_system_prompt=default_system_prompt,
            )
            self._policies[target.name] = _DatasetPolicy(
                spec=target,
                prompts=target_prompts,
                augmentation_enabled=target.supports_augmentation,
                curriculum_enabled=target.supports_curriculum,
                max_objects_per_image=None,  # targets stay uncapped
                seed=target.seed,
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
                augmentation_enabled=False,  # sources remain clean
                curriculum_enabled=False,
                max_objects_per_image=source.max_objects_per_image,
                seed=source.seed,
            )

            # Skip loading for train split when ratio is zero
            if split == "train" and source.ratio <= 0:
                self._record_pools[source.name] = []
                continue

            if split == "eval":
                # Target-only eval: ignore source splits even if provided
                self._record_pools[source.name] = []
                continue
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
            preprocessor=preprocessor,
            bypass_prob=bypass_prob,
            curriculum_state=curriculum_state,
            use_summary=use_summary,
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=system_prompt_summary,
            seed=seed,
            dataset_name=self._primary_target,
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
            if (
                policy.max_objects_per_image is not None
                and self._split == "train"
                and policy.spec.domain == "source"
            ):
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
        user_prompt = (
            spec.prompt_user
            if spec.prompt_user is not None
            else (domain_user if domain_user is not None else default_user_prompt)
        )
        system_prompt = (
            spec.prompt_system
            if spec.prompt_system is not None
            else (domain_system if domain_system is not None else default_system_prompt)
        )

        if spec.prompt_user or spec.prompt_system:
            source = "dataset"
        elif domain_user or domain_system:
            source = "domain"
        else:
            source = "default"
        return _PromptResolution(user=user_prompt, system=system_prompt, source=source)

    def _build_train_schedule(self) -> None:
        target_pool_sizes = {
            name: len(self._record_pools.get(name, [])) for name in self._target_names
        }
        target_quotas, _ = _compute_target_quotas(
            self._fusion_config.targets, target_pool_sizes
        )

        schedule: list[tuple[str, int]] = []
        counts: dict[str, int] = {}

        for target in self._fusion_config.targets:
            pool = self._record_pools.get(target.name, [])
            quota = target_quotas.get(target.name, 0)
            if quota <= 0:
                counts[target.name] = 0
                continue
            rng_target = random.Random(
                self._mix_seed(
                    self.seed,
                    self._epoch,
                    self._policies[target.name].seed or 0,
                    0xA1,
                )
            )
            indices = list(range(len(pool)))
            if self._shuffle and len(indices) > 1:
                rng_target.shuffle(indices)
            if quota <= len(pool):
                sampled_target = indices[:quota]
            else:
                sampled_target = indices[:]
                extra_needed = quota - len(pool)
                sampled_target.extend(
                    rng_target.randrange(len(pool)) for _ in range(extra_needed)
                )
            counts[target.name] = len(sampled_target)
            schedule.extend((target.name, idx) for idx in sampled_target)

        total_target_quota = sum(counts.get(name, 0) for name in self._target_names)

        for source in self._fusion_config.sources:
            policy = self._policies[source.name]
            pool = self._record_pools.get(source.name, [])
            if source.ratio > 0 and not pool:
                raise ValueError(
                    f"Fusion source '{source.name}' is empty while ratio={source.ratio}"
                )
            quota = round(source.ratio * total_target_quota)
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
        # Debug: log schedule composition
        try:
            from src.utils import get_logger

            logger = get_logger(__name__)
            logger.debug(
                f"FusionCaptionDataset train schedule (epoch {self._epoch}): {self._epoch_counts}"
            )
            # Count samples per dataset in schedule
            schedule_counts = {}
            for dataset_name, _ in schedule:
                schedule_counts[dataset_name] = schedule_counts.get(dataset_name, 0) + 1
            logger.debug(f"Schedule sample counts: {schedule_counts}")
        except Exception:
            pass
        self._update_epoch_plan()

    def _build_eval_schedule(self) -> None:
        schedule: list[tuple[str, int]] = []
        counts: dict[str, int] = {}

        # Eval is target-only (all targets concatenated, deterministic order)
        for target_name in self._target_names:
            pool = self._record_pools.get(target_name, [])
            counts[target_name] = len(pool)
            schedule.extend((target_name, idx) for idx in range(len(pool)))

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

        schedule = self._schedule
        if index >= len(schedule):
            raise IndexError(
                f"Requested index {index} exceeds current schedule length {len(schedule)}"
            )
        dataset_name, base_idx = schedule[index % len(schedule)]
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
        if (
            self._split == "train"
            and policy.spec.domain == "source"
            and cap_pre is not None
            and policy.max_objects_per_image is not None
        ):
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

        if self.mode == "summary" and self.preprocessor is not None:
            if hasattr(self.preprocessor, "rng"):
                self.preprocessor.rng = rng_local
            processed_summary = self.preprocessor(record)
            if processed_summary is None:
                raise ValueError(
                    "Summary preprocessor removed the record; dataset does not duplicate samples"
                )
            record = processed_summary

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
            info["input_length"] = len(input_ids)
            try:
                # Debug-only: emit per-sample token length by dataset for max-length tuning.
                # Enhanced logging includes domain, object count, and template info for better debugging.
                domain = policy.spec.domain
                template_name = policy.spec.template
                is_text_only = objects_after == 0 and domain == "source"
                domain_label = f"{domain}[{dataset_name}]"

                # Build detailed log message
                log_parts = [
                    f"Sample token length | {domain_label}",
                    f"base_idx={base_idx}",
                    f"len={len(input_ids)}",
                ]

                # Add object count for image-based datasets
                if not is_text_only:
                    log_parts.append(f"objects={objects_after}")
                    if cap_applied:
                        log_parts.append(f"capped_from={objects_before}")

                # Add template info
                log_parts.append(f"template={template_name}")

                # Add augmentation status for targets
                if domain == "target" and was_augmented:
                    log_parts.append("aug=True")

                self._logger.debug(" | ".join(log_parts))
            except Exception:
                pass
        self.last_sample_debug = info
        LAST_SAMPLE_DEBUG.update(info)

        return encoded

    @property
    def aux_quota(self) -> dict[str, int]:
        """Per-epoch auxiliary sample counts for debugging/telemetry."""
        return {
            k: v
            for k, v in self._epoch_counts.items()
            if self._policies.get(k, None) and self._policies[k].spec.domain == "source"
        }

    def set_curriculum_state(self, state: MutableMapping[str, Any]) -> None:
        self.curriculum_state = state
        for name, aug in self._preprocessors_aug.items():
            policy = self._policies.get(name)
            if policy and policy.curriculum_enabled:
                try:
                    aug.curriculum_state = state
                except Exception:
                    pass

    def make_target_eval_dataset(
        self, *, mine_clean: bool = False
    ) -> BaseCaptionDataset:
        primary = self._primary_target
        policy = self._policies[primary]
        use_aug = (
            policy.augmentation_enabled
            and self._augmenter is not None
            and not mine_clean
        )
        return BaseCaptionDataset(
            base_records=self._record_pools[primary],
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
            dataset_name=primary,
            allow_empty=False,
        )


def fusion_pack_group_key(record: Mapping[str, Any]) -> str:
    """Return packing group key (domain) for a fusion sample.

    Ensures packed sequences keep target and source records separate.
    """

    meta = record.get("metadata") if isinstance(record, Mapping) else None
    if isinstance(meta, Mapping):
        domain = meta.get("_fusion_domain")
        if domain is not None:
            return str(domain)
    return "target"


# Backward compatibility alias
UnifiedFusionDataset = FusionCaptionDataset
