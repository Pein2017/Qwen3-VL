"""Unified fusion dataset that concatenates JSONL files and uses a single template.

This avoids template cloning issues by using one dataset with one template.
Records are tagged with their source dataset for prompt selection.
"""

from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast, override

from torch.utils.data import get_worker_info

from src.utils import get_logger, require_mutable_mapping
from src.utils.unstructured import UnstructuredMapping, UnstructuredMutableMapping

from ..config.prompts import (
    SYSTEM_PROMPT_SUMMARY,
    USER_PROMPT_SUMMARY,
    get_template_prompts,
)
from .assistant_prefix import (
    build_assistant_prefix,
    resolve_domain_token,
    resolve_task_token,
)
from .builders import JSONLinesBuilder
from .contracts import ConversationRecord, validate_conversation_record
from .dense_caption import LAST_SAMPLE_DEBUG, BaseCaptionDataset, TemplateProtocol
from .fusion import FusionConfig, _compute_target_quotas, _sample_indices
from .fusion_types import DatasetSpec
from .preprocessors import AugmentationPreprocessor, ObjectCapPreprocessor
from .utils import extract_assistant_text, extract_object_points, load_jsonl


@dataclass(frozen=True)
class _PromptResolution:
    user: str
    system: str | None
    source: Literal["default", "domain", "dataset"]


@dataclass
class _DatasetPolicy:
    spec: DatasetSpec
    prompts: _PromptResolution
    mode: Literal["dense", "summary"]
    augmentation_enabled: bool
    curriculum_enabled: bool
    max_objects_per_image: int | None
    seed: int | None
    sample_without_replacement: bool


class EpochPlanEntry(TypedDict):
    count: int
    mode: Literal["dense", "summary"]
    augmentation: bool
    curriculum: bool
    max_objects_per_image: int | None
    prompt_source: Literal["default", "domain", "dataset"]
    sample_without_replacement: bool
    fallback_to_replacement: bool


class FusionCaptionDataset(BaseCaptionDataset):
    """Fusion dataset that concatenates multiple JSONL sources with a unified template."""

    _logger: object = get_logger(__name__)
    _IRRELEVANT_SOURCE = "irrelevant_summary"
    _IRRELEVANT_ALT_TEMPLATES = ("summary_bbu", "summary_rru")
    _fusion_config: FusionConfig
    _augmenter: object | None
    bypass_prob: float
    curriculum_state: UnstructuredMutableMapping | None
    _shuffle: bool
    _include_source_eval: bool
    _sample_limit: int | None
    _epoch: int
    _assistant_prefix_format: str | None
    _target_names: list[str]
    _dataset_order: list[str]
    _primary_target: str
    _rng: random.Random
    last_sample_debug: UnstructuredMapping

    def __init__(
        self,
        *,
        fusion_config: FusionConfig,
        base_template: object,
        user_prompt: str,
        emit_norm: Literal["none", "norm100", "norm1000"],
        json_format: Literal["standard"],
        augmenter: object | None,
        bypass_prob: float,
        curriculum_state: UnstructuredMutableMapping | None,
        use_summary: bool,
        system_prompt_dense: str | None,
        system_prompt_summary: str | None,
        assistant_prefix_format: str | None = None,
        seed: int = 42,
        shuffle: bool = True,
        sample_limit: int | None = None,
        split: Literal["train", "eval"] = "train",
        target_eval_jsonl: str | None = None,
        include_source_eval: bool = False,
    ):
        self._fusion_config = fusion_config
        self._split: Literal["train", "eval"] = split
        self._augmenter = augmenter
        self.bypass_prob = float(bypass_prob)
        self.curriculum_state = (
            require_mutable_mapping(curriculum_state, context="fusion.curriculum_state")
            if curriculum_state is not None
            else None
        )
        self._shuffle = bool(shuffle)
        self._include_source_eval = False  # target-only eval per policy
        self._sample_limit = sample_limit
        self._epoch = 0
        self._schedule: list[tuple[str, int]] = []
        self._epoch_counts: dict[str, int] = {}
        self._without_replacement_fallbacks: dict[str, bool] = {}
        self._policies: dict[str, _DatasetPolicy] = {}
        self._record_pools: dict[str, list[ConversationRecord]] = {}
        self._preprocessors_aug: dict[str, AugmentationPreprocessor] = {}
        self._preprocessors_cap: dict[str, ObjectCapPreprocessor] = {}
        self.epoch_plan: dict[str, EpochPlanEntry] = {}
        self._assistant_prefix_format = (
            assistant_prefix_format.strip() if assistant_prefix_format else None
        )

        self._target_names = [t.name for t in fusion_config.targets]
        self._dataset_order = [
            *self._target_names,
            *[src.name for src in fusion_config.sources],
        ]
        self._primary_target = self._target_names[0]

        default_mode: Literal["dense", "summary"] = (
            "summary" if use_summary else "dense"
        )
        default_user_prompt_dense = user_prompt
        default_system_prompt_dense = system_prompt_dense
        default_system_prompt_summary = system_prompt_summary

        resolved_modes: dict[str, Literal["dense", "summary"]] = {}
        needs_summary_prompt = default_mode == "summary"
        for spec in [*fusion_config.targets, *fusion_config.sources]:
            mode_val: Literal["dense", "summary"] = (
                spec.mode if spec.mode is not None else default_mode
            )
            resolved_modes[spec.name] = mode_val
            if mode_val == "summary":
                needs_summary_prompt = True

        if needs_summary_prompt and default_system_prompt_summary is None:
            if default_mode == "summary":
                default_system_prompt_summary = getattr(base_template, "system", None)
        if needs_summary_prompt and default_system_prompt_summary is None:
            default_system_prompt_summary = SYSTEM_PROMPT_SUMMARY
        if needs_summary_prompt and default_system_prompt_summary is None:
            raise ValueError(
                "Summary mode requested but no summary system prompt was provided."
            )

        # Load pools for the selected split
        if split == "eval" and target_eval_jsonl is not None:
            override_target_path = Path(target_eval_jsonl)
        else:
            override_target_path = None

        for target in fusion_config.targets:
            mode = resolved_modes.get(target.name, default_mode)
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
                self._annotate_record(rec, target, mode) for rec in target_records
            ]

            # Build prompt/policy for target
            target_prompts = self._resolve_prompts(
                target,
                mode=mode,
                default_user_prompt_dense=default_user_prompt_dense,
                default_system_prompt_dense=default_system_prompt_dense,
                default_system_prompt_summary=default_system_prompt_summary,
            )
            self._policies[target.name] = _DatasetPolicy(
                spec=target,
                prompts=target_prompts,
                mode=mode,
                augmentation_enabled=target.supports_augmentation,
                curriculum_enabled=target.supports_curriculum,
                max_objects_per_image=None,  # targets stay uncapped
                seed=target.seed,
                sample_without_replacement=bool(
                    getattr(target, "sample_without_replacement", False)
                ),
            )

        # Sources
        for source in fusion_config.sources:
            mode = resolved_modes.get(source.name, default_mode)
            policy = self._resolve_prompts(
                source,
                mode=mode,
                default_user_prompt_dense=default_user_prompt_dense,
                default_system_prompt_dense=default_system_prompt_dense,
                default_system_prompt_summary=default_system_prompt_summary,
            )
            self._policies[source.name] = _DatasetPolicy(
                spec=source,
                prompts=policy,
                mode=mode,
                augmentation_enabled=False,  # sources remain clean
                curriculum_enabled=False,
                max_objects_per_image=source.max_objects_per_image,
                seed=source.seed,
                sample_without_replacement=bool(
                    getattr(source, "sample_without_replacement", False)
                ),
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
                self._annotate_record(rec, source, mode) for rec in source_records
            ]

        # Initialize parent BaseCaptionDataset with a single template instance.
        all_records = [rec for pool in self._record_pools.values() for rec in pool]
        super().__init__(
            base_records=all_records,
            template=cast(TemplateProtocol, base_template),
            user_prompt=user_prompt,
            user_prompt_summary=USER_PROMPT_SUMMARY,
            emit_norm=emit_norm,
            json_format=json_format,
            augmenter=None,
            bypass_prob=bypass_prob,
            curriculum_state=curriculum_state,
            use_summary=default_mode == "summary",
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=default_system_prompt_summary
            if needs_summary_prompt
            else system_prompt_summary,
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
        record: ConversationRecord,
        spec: DatasetSpec,
        mode: Literal["dense", "summary"],
    ) -> ConversationRecord:
        """Annotate record with source dataset metadata."""
        annotated = cast(ConversationRecord, copy.deepcopy(dict(record)))
        metadata = annotated.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["_fusion_source"] = spec.name
        metadata["_fusion_template"] = spec.template
        metadata["_fusion_domain"] = spec.domain
        metadata["_fusion_mode"] = mode
        annotated["metadata"] = metadata
        return annotated

    @staticmethod
    def _load_records(path: Path, *, limit: int | None) -> list[ConversationRecord]:
        records = load_jsonl(str(path), resolve_relative=True)
        if limit is not None and limit > 0:
            records = records[:limit]
        validated: list[ConversationRecord] = []
        for idx, record in enumerate(records):
            try:
                validated.append(
                    cast(
                        ConversationRecord,
                        dict(copy.deepcopy(validate_conversation_record(record))),
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
        mode: Literal["dense", "summary"],
        default_user_prompt_dense: str,
        default_system_prompt_dense: str | None,
        default_system_prompt_summary: str | None,
    ) -> _PromptResolution:
        if mode == "summary" and default_system_prompt_summary is None:
            raise ValueError(
                f"Summary mode requested for dataset '{spec.name}' but no summary system prompt provided."
            )

        template_name = spec.template
        if not template_name:
            raise ValueError(
                f"Dataset '{spec.name}' must specify a template; no fallback is allowed."
            )
        if mode == "summary" and "summary" not in template_name.lower():
            raise ValueError(
                f"Dataset '{spec.name}' is in summary mode but template '{template_name}' "
                "is not a summary template. Provide an explicit *summary template."
            )
        if mode == "dense" and "summary" in template_name.lower():
            raise ValueError(
                f"Dataset '{spec.name}' is in dense mode but template '{template_name}' "
                "is a summary template. Use the matching dense template."
            )
        domain_system, domain_user = get_template_prompts(template_name)

        if mode == "summary":
            default_user_prompt = USER_PROMPT_SUMMARY
            default_system_prompt = default_system_prompt_summary
        else:
            default_user_prompt = default_user_prompt_dense
            default_system_prompt = default_system_prompt_dense

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

    def _validate_record_for_mode(
        self,
        record: Mapping[str, object],
        mode: Literal["dense", "summary"],
        dataset_name: str,
    ) -> None:
        """Validate record contents based on the resolved mode."""
        if record.get("messages"):
            # Pre-authored chat records bypass geometry/summary validation.
            return
        if mode == "summary":
            summary = record.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError(
                    f"Fusion dataset '{dataset_name}' is in summary mode but record is missing a non-empty 'summary' string."
                )
            return

        objects = record.get("objects") or []
        if not isinstance(objects, list):
            raise ValueError(
                f"Fusion dataset '{dataset_name}' is in dense mode but 'objects' is not a list."
            )
        if not objects:
            raise ValueError(
                f"Fusion dataset '{dataset_name}' is in dense mode but record has no objects."
            )

        for idx, obj in enumerate(objects, start=1):
            geom_type, points = extract_object_points(obj)
            if not geom_type or not points:
                raise ValueError(
                    f"Fusion dataset '{dataset_name}' dense mode requires geometry; object_{idx} is missing bbox_2d/poly/line points."
                )

    @classmethod
    def _pick_irrelevant_template(
        cls,
        record: Mapping[str, object],
        *,
        dataset_name: str,
        base_idx: int,
        epoch: int | None = None,
    ) -> str:
        images = record.get("images")
        key = None
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, str) and first:
                key = first
        if key is None:
            key = f"{dataset_name}:{base_idx}"
        if epoch is not None:
            key = f"{epoch}:{key}"
        digest = hashlib.md5(key.encode("utf-8")).digest()
        return cls._IRRELEVANT_ALT_TEMPLATES[digest[0] & 1]

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
            policy = self._policies[target.name]
            # Use replacement sampling for targets (consistent with sources when sample_without_replacement=False)
            # This allows duplicates within an epoch and ensures different samples each epoch
            if policy.sample_without_replacement and quota <= len(pool):
                # Without replacement: shuffle and take first quota
                indices = list(range(len(pool)))
                if self._shuffle and len(indices) > 1:
                    rng_target.shuffle(indices)
                sampled_target = indices[:quota]
            else:
                # With replacement: sample quota indices with replacement
                sampled_target = [rng_target.randrange(len(pool)) for _ in range(quota)]
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
                self._without_replacement_fallbacks[source.name] = False
                continue

            rng_source = random.Random(
                self._mix_seed(self.seed, self._epoch, policy.seed or 0, 0xB7)
            )
            sampled_indices, fell_back = _sample_indices(
                len(pool),
                quota,
                rng_source,
                sample_without_replacement=policy.sample_without_replacement,
            )
            # Fallback path uses replacement when quota > pool.
            if policy.sample_without_replacement and fell_back:
                sampled_indices = [
                    rng_source.randrange(len(pool)) for _ in range(quota)
                ]
            counts[source.name] = len(sampled_indices)
            schedule.extend((source.name, idx) for idx in sampled_indices)
            self._without_replacement_fallbacks[source.name] = bool(fell_back)

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
        plan: dict[str, EpochPlanEntry] = {}
        for name in self._dataset_order:
            policy = self._policies.get(name)
            if policy is None:
                continue
            plan[name] = {
                "count": self._epoch_counts.get(name, 0),
                "mode": policy.mode,
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
                "sample_without_replacement": policy.sample_without_replacement,
                "fallback_to_replacement": bool(
                    self._without_replacement_fallbacks.get(name, False)
                ),
            }
        self.epoch_plan = plan

    @override
    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._rng = random.Random(self._seed_for_epoch(self._epoch))
        if self._split == "eval":
            self._build_eval_schedule()
        else:
            self._build_train_schedule()

    @override
    def __len__(self) -> int:
        return len(self._schedule)

    @override
    def __getitem__(self, index: int) -> UnstructuredMutableMapping:
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
        objects_raw = record.get("objects")
        objects_before = len(objects_raw) if isinstance(objects_raw, list) else 0

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
            objects_raw_after = record.get("objects")
            objects_after = (
                len(objects_raw_after) if isinstance(objects_raw_after, list) else 0
            )
            cap_applied = objects_after < objects_before or (
                objects_before > policy.max_objects_per_image
            )
        else:
            objects_raw_else = record.get("objects")
            objects_after = (
                len(objects_raw_else) if isinstance(objects_raw_else, list) else 0
            )

        # Mode-aware validation after preprocessors
        self._validate_record_for_mode(record, policy.mode, dataset_name)

        # Build conversation
        mode = policy.mode
        assistant_prefix = None
        prompts = policy.prompts
        prompt_source = prompts.source
        prompt_template = policy.spec.template
        user_prompt = prompts.user
        system_prompt = prompts.system
        if system_prompt is None:
            system_prompt = (
                self.system_prompt_summary
                if mode == "summary"
                else self.system_prompt_dense
            )
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            record["metadata"] = metadata

        fusion_source = str(metadata.get("_fusion_source") or "").strip().lower()
        is_irrelevant = bool(fusion_source) and fusion_source.startswith("irrelevant")
        builder_mode = "summary" if is_irrelevant else mode

        # Summary mode (and irrelevant streams) store a reference for downstream auditors.
        if mode == "summary" or is_irrelevant:
            summary_ref = record.get("summary")
            if isinstance(summary_ref, str) and summary_ref.strip():
                metadata["summary_ref"] = summary_ref

        # Irrelevant streams always emit a single-line "无关图片" assistant payload
        # (regardless of declared mode) and reuse summary prompts (alternating BBU/RRU).
        if is_irrelevant:
            assistant_prefix = None
            summary = record.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                record["summary"] = "无关图片"

            if self._split == "train":
                alt_template = self._pick_irrelevant_template(
                    record,
                    dataset_name=dataset_name,
                    base_idx=base_idx,
                    epoch=self._epoch,
                )
            else:
                alt_template = self._pick_irrelevant_template(
                    record,
                    dataset_name=dataset_name,
                    base_idx=base_idx,
                )
            alt_system, alt_user = get_template_prompts(alt_template)
            if alt_user:
                user_prompt = alt_user
            if alt_system is not None:
                system_prompt = alt_system
            prompt_source = "domain"
            prompt_template = alt_template
            metadata["_fusion_template"] = alt_template

        if (
            self._assistant_prefix_format
            and policy.spec.domain == "target"
            and not is_irrelevant
        ):
            dataset_token = str(policy.spec.key).strip().lower()

            if mode == "summary":
                template_name = metadata.get("_fusion_template")
                if template_name == "summary_bbu":
                    domain_token = "BBU"
                elif template_name == "summary_rru":
                    domain_token = "RRU"
                else:
                    raise ValueError(
                        "Summary template must be summary_bbu or summary_rru "
                        f"for header construction; got {template_name!r}."
                    )
                assistant_prefix = build_assistant_prefix(
                    fmt=self._assistant_prefix_format,
                    domain=domain_token,
                    task=resolve_task_token(mode),
                    dataset=dataset_token,
                )
            else:
                domain_token = resolve_domain_token(policy.spec.key)
                if domain_token is None:
                    raise ValueError(
                        "assistant_prefix_format configured but unsupported dataset key "
                        f"{policy.spec.key}."
                    )
                assistant_prefix = build_assistant_prefix(
                    fmt=self._assistant_prefix_format,
                    domain=domain_token,
                    task=resolve_task_token(mode),
                    dataset=dataset_token,
                )

        builder = JSONLinesBuilder(
            user_prompt=user_prompt,
            emit_norm=self.emit_norm,
            mode=builder_mode,
            json_format=self.json_format,
            assistant_prefix=assistant_prefix,
        )
        merged = builder.build_many([record])
        conversation_messages = copy.deepcopy(merged.get("messages", []) or [])

        if system_prompt not in (None, ""):
            conversation_messages = [
                {"role": "system", "content": system_prompt},
                *conversation_messages,
            ]

        original_system = getattr(self.template, "system", None)
        try:
            if system_prompt is not None:
                try:
                    setattr(self.template, "system", system_prompt)
                except Exception:
                    pass
            encode_method = getattr(self.template, "encode", None)
            if encode_method is None:
                raise AttributeError("template.encode is not available")
            encoded = encode_method(merged, return_length=True)
        finally:
            if system_prompt is not None:
                try:
                    setattr(self.template, "system", original_system)
                except Exception:
                    pass

        encoded["messages"] = conversation_messages
        encoded["solution"] = extract_assistant_text(conversation_messages) or ""
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
            "prompt_source": prompt_source,
            "prompt_template": prompt_template,
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

                debug_method = getattr(self._logger, "debug", None)
                if debug_method is not None:
                    debug_method(" | ".join(log_parts))
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

    def set_curriculum_state(self, state: MutableMapping[str, object]) -> None:
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
        use_summary_mode = policy.mode == "summary"
        system_prompt_dense = (
            policy.prompts.system
            if policy.mode == "dense"
            else self.system_prompt_dense
        )
        system_prompt_summary = (
            policy.prompts.system
            if policy.mode == "summary"
            else self.system_prompt_summary
        )
        user_prompt_summary = policy.prompts.user if policy.mode == "summary" else None
        return BaseCaptionDataset(
            base_records=self._record_pools[primary],
            template=self.template,
            user_prompt=policy.prompts.user,
            user_prompt_summary=user_prompt_summary,
            emit_norm=self.emit_norm,
            json_format=self.json_format,
            augmenter=self._augmenter if use_aug else None,
            bypass_prob=0.0 if mine_clean else self.bypass_prob,
            curriculum_state=None,
            use_summary=use_summary_mode,
            system_prompt_dense=system_prompt_dense,
            system_prompt_summary=system_prompt_summary,
            seed=self.seed,
            dataset_name=primary,
            allow_empty=False,
        )


def fusion_pack_group_key(record: Mapping[str, object]) -> str:
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
