from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Literal, MutableMapping, Optional, Sequence

from torch.utils.data import Dataset, get_worker_info

from src.config.prompts import USER_PROMPT_SUMMARY

from .builders import JSONLinesBuilder
from .contracts import ConversationRecord, validate_conversation_record
from .preprocessors import AugmentationPreprocessor, SequentialPreprocessor
from .utils import load_jsonl

# Exposed for debugging (e.g., OOM tracing)
LAST_SAMPLE_DEBUG: Dict[str, Any] = {}


class BaseCaptionDataset(Dataset):
    """Base caption dataset without dynamic pairing.

    Each sample corresponds to a single base record. The dataset supports
    optional augmentation, summary/dense mode selection, and epoch-level shuffling
    consistent with the legacy dynamic pairing pipeline.
    """

    def __init__(
        self,
        base_records: Sequence[Any],
        template: Any,
        user_prompt: str,
        emit_norm: Literal["none", "norm100", "norm1000"],
        json_format: Literal["standard"],
        augmenter: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        use_summary: bool = False,
        system_prompt_dense: Optional[str] = None,
        system_prompt_summary: Optional[str] = None,
        bypass_prob: float = 0.0,
        seed: int = 2025,
        curriculum_state: Optional[MutableMapping[str, Any]] = None,
        dataset_name: Optional[str] = None,
        allow_empty: bool = False,
    ):
        self.use_summary = bool(use_summary)
        self.system_prompt_dense = system_prompt_dense
        self.system_prompt_summary = system_prompt_summary
        self.user_prompt = user_prompt
        self.emit_norm: Literal["none", "norm100", "norm1000"] = emit_norm
        self.json_format: Literal["standard"] = json_format
        self.bypass_prob = float(bypass_prob)
        self.seed = int(seed)
        self.template = template
        self.mode: Literal["dense", "summary"] = (
            "summary" if self.use_summary else "dense"
        )

        if self.use_summary:
            if self.system_prompt_summary is None:
                self.system_prompt_summary = getattr(self.template, "system", None)
            if self.system_prompt_summary is None:
                raise ValueError(
                    "system_prompt_summary is required when use_summary is true."
                )
            try:
                setattr(self.template, "system", self.system_prompt_summary)
            except Exception:
                pass
        else:
            if self.system_prompt_dense is None:
                self.system_prompt_dense = getattr(self.template, "system", None)
            if self.system_prompt_dense is not None:
                try:
                    setattr(self.template, "system", self.system_prompt_dense)
                except Exception:
                    pass

        validated_records: List[ConversationRecord] = []
        for idx, record in enumerate(base_records):
            try:
                validated = validate_conversation_record(record)
            except ValueError as exc:
                raise ValueError(f"Base record {idx} is invalid: {exc}") from exc
            validated_records.append(copy.deepcopy(validated))

        if not validated_records and not allow_empty:
            raise ValueError("BaseCaptionDataset requires at least one valid record")

        self.base_records = validated_records

        preprocessors = []
        if preprocessor is not None:
            preprocessors.append(preprocessor)
        if augmenter is not None:
            preprocessors.append(
                AugmentationPreprocessor(
                    augmenter=augmenter,
                    bypass_prob=self.bypass_prob,
                    curriculum_state=curriculum_state,
                )
            )
        if preprocessors:
            self.preprocessor = (
                preprocessors[0]
                if len(preprocessors) == 1
                else SequentialPreprocessor(preprocessors)
            )
            if hasattr(self.preprocessor, "curriculum_state"):
                try:
                    self.preprocessor.curriculum_state = curriculum_state
                except Exception:
                    pass
        else:
            self.preprocessor = None

        self._epoch = 0
        self._rng = random.Random(self._seed_for_epoch(self._epoch))
        self._index_perm = list(range(len(self.base_records)))
        self._hard_sample_plan: Dict[str, Any] | None = None
        self._rebuild_perm_for_epoch()
        self.dataset_name = dataset_name or "dataset"
        self.last_sample_debug: Dict[str, Any] = {}

    @staticmethod
    def _make_sample_id(dataset_name: str, base_idx: int) -> int:
        import zlib

        ns = zlib.crc32(dataset_name.encode("utf-8")) & 0xFFFF
        return (ns << 32) | (int(base_idx) & 0xFFFFFFFF)

    @staticmethod
    def from_jsonl(
        jsonl_path: str,
        template: Any,
        **kwargs,
    ) -> "BaseCaptionDataset":
        records = load_jsonl(jsonl_path, resolve_relative=True)
        # Optional sample limiting for quick smoke tests
        sample_limit = kwargs.pop("sample_limit", None)
        if isinstance(sample_limit, int) and sample_limit > 0:
            records = records[:sample_limit]
        elif isinstance(sample_limit, str) and sample_limit.isdigit():
            records = records[: int(sample_limit)]
        # Backward-compatibility: drop unused arg if present
        if "summary_ratio" in kwargs:
            raise TypeError(
                "summary_ratio is no longer supported; use use_summary instead."
            )
            kwargs.pop("use_detailed_caption", None)
            kwargs.pop("output_variant", None)  # Backward compat
        return BaseCaptionDataset(
            base_records=records,
            template=template,
            **kwargs,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._rng = random.Random(self._seed_for_epoch(self._epoch))
        self._rebuild_perm_for_epoch()

    def _seed_for_epoch(self, epoch: int) -> int:
        base_seed = self.seed & 0xFFFFFFFF
        mixed = (base_seed ^ ((int(epoch) + 1) * 0x9E3779B1)) & 0xFFFFFFFF
        return mixed

    def _rebuild_perm_for_epoch(self) -> None:
        base_len = len(self.base_records)
        plan = self._hard_sample_plan or {}
        target_len = int(plan.get("target_epoch_size") or base_len)
        weights_map = plan.get("weights") if isinstance(plan, MutableMapping) else None

        if weights_map:
            indices = list(range(base_len))
            weights = [float(weights_map.get(i, 1.0)) for i in indices]
            self._index_perm = self._rng.choices(indices, weights=weights, k=target_len)
        else:
            perm = list(range(base_len))
            if len(perm) > 1:
                self._rng.shuffle(perm)
            if target_len == base_len:
                self._index_perm = perm
            elif target_len < base_len:
                self._index_perm = perm[:target_len]
            else:
                extra = self._rng.choices(perm, k=target_len - base_len)
                self._index_perm = perm + extra

    def set_hard_sample_plan(self, plan: Optional[Mapping[str, Any]]) -> None:
        self._hard_sample_plan = dict(plan) if plan is not None else None

    def __len__(self) -> int:
        return len(self.base_records)

    def _create_builder(self, mode: Literal["dense", "summary"]) -> JSONLinesBuilder:
        user_prompt = USER_PROMPT_SUMMARY if mode == "summary" else self.user_prompt
        return JSONLinesBuilder(
            user_prompt=user_prompt,
            emit_norm=self.emit_norm,
            mode=mode,
            json_format=self.json_format,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not self.base_records:
            raise IndexError("BaseCaptionDataset is empty")

        base_idx = self._index_perm[index % len(self._index_perm)]
        record = copy.deepcopy(self.base_records[base_idx])

        worker = get_worker_info()
        seed_local = self._rng.randrange(0, 2**32 - 1)
        if worker is not None:
            seed_local ^= ((worker.id + 1) * 0xC2B2AE35) & 0xFFFFFFFF
        rng_local = random.Random(seed_local & 0xFFFFFFFF)

        if self.preprocessor is not None:
            if hasattr(self.preprocessor, "rng"):
                self.preprocessor.rng = rng_local
            processed = self.preprocessor(record)
            if processed is None:
                raise ValueError(
                    "Preprocessor removed the record; dataset does not duplicate samples"
                )
            record = processed

        mode = self.mode
        builder = self._create_builder(mode)
        merged = builder.build_many([record])

        system_prompt = None
        if mode == "summary" and self.system_prompt_summary:
            system_prompt = self.system_prompt_summary
        elif mode == "dense" and self.system_prompt_dense:
            system_prompt = self.system_prompt_dense

        conversation_messages = copy.deepcopy(merged.get("messages", []) or [])
        if system_prompt is not None:
            conversation_messages = [
                {"role": "system", "content": system_prompt},
                *conversation_messages,
            ]

        original_system = None
        if system_prompt is not None:
            try:
                original_system = getattr(self.template, "system", None)
                self.template.system = system_prompt
            except Exception:
                original_system = None

        try:
            encoded = self.template.encode(merged, return_length=True)
        finally:
            if system_prompt is not None and original_system is not None:
                try:
                    self.template.system = original_system
                except Exception:
                    pass

        # Track last-sample debug info for OOM/root-cause tracing
        try:
            objects = record.get("objects") or []
            max_poly = 0
            for obj in objects:
                if "poly_points" in obj:
                    max_poly = max(max_poly, int(obj.get("poly_points") or 0))
            info = {
                "dataset": self.dataset_name,
                "base_idx": base_idx,
                "objects": len(objects),
                "max_poly_points": max_poly,
                "width": record.get("width"),
                "height": record.get("height"),
                "mode": mode,
            }
            input_ids = encoded.get("input_ids")
            if input_ids is not None and hasattr(input_ids, "__len__"):
                try:
                    info["input_ids_len"] = len(input_ids)
                except Exception:
                    pass
            # attach sample metadata for downstream mining
            sample_id = self._make_sample_id(self.dataset_name, base_idx)
            encoded["sample_id"] = sample_id
            encoded["dataset"] = self.dataset_name
            encoded["base_idx"] = base_idx
            self.last_sample_debug = info
            LAST_SAMPLE_DEBUG.update(info)
        except Exception:
            # Best-effort; do not block training
            pass

        # Attach the original conversation so RLHF/GKD trainers can re-encode.
        encoded["messages"] = conversation_messages
        for key in ("assistant_payload", "objects", "metadata"):
            if key in merged:
                encoded[key] = copy.deepcopy(merged[key])

        return encoded


# Backward compatibility alias
DenseCaptionDataset = BaseCaptionDataset
