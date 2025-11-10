from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Literal, Optional, Sequence

from torch.utils.data import Dataset, get_worker_info

from .builders import JSONLinesBuilder
from .contracts import ConversationRecord, validate_conversation_record
from .preprocessors import AugmentationPreprocessor
from .utils import load_jsonl
from src.config.prompts import USER_PROMPT_SUMMARY


class DenseCaptionDataset(Dataset):
    """Dense caption dataset without dynamic pairing.

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
        json_format: Literal["type_a", "type_b", "type_c", "type_d"],
        augmenter: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        use_summary: bool = False,
        system_prompt_dense: Optional[str] = None,
        system_prompt_summary: Optional[str] = None,
        bypass_prob: float = 0.0,
        seed: int = 2025,
    ):
        self.use_summary = bool(use_summary)
        self.system_prompt_dense = system_prompt_dense
        self.system_prompt_summary = system_prompt_summary
        self.user_prompt = user_prompt
        self.emit_norm: Literal["none", "norm100", "norm1000"] = emit_norm
        self.json_format = json_format
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

        if not validated_records:
            raise ValueError("DenseCaptionDataset requires at least one valid record")

        self.base_records = validated_records

        self.preprocessor = preprocessor
        if augmenter is not None and self.preprocessor is None:
            self.preprocessor = AugmentationPreprocessor(
                augmenter=augmenter, bypass_prob=self.bypass_prob
            )

        self._epoch = 0
        self._rng = random.Random(self._seed_for_epoch(self._epoch))
        self._index_perm = list(range(len(self.base_records)))
        self._rebuild_perm_for_epoch()

    @staticmethod
    def from_jsonl(
        jsonl_path: str,
        template: Any,
        **kwargs,
    ) -> "DenseCaptionDataset":
        records = load_jsonl(jsonl_path)
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
        return DenseCaptionDataset(
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
        self._index_perm = list(range(len(self.base_records)))
        if len(self._index_perm) > 1:
            self._rng.shuffle(self._index_perm)

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
            raise IndexError("DenseCaptionDataset is empty")

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

        # Attach the original conversation so RLHF/GKD trainers can re-encode.
        encoded["messages"] = conversation_messages
        for key in ("assistant_payload", "objects", "metadata"):
            if key in merged:
                encoded[key] = copy.deepcopy(merged[key])

        return encoded
