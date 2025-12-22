from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Literal, MutableMapping, Optional, Sequence

from src.utils import get_logger

from torch.utils.data import Dataset, get_worker_info
from swift.llm.template.base import MaxLengthError

from src.config.prompts import USER_PROMPT_SUMMARY

from .builders import JSONLinesBuilder
from .contracts import ConversationRecord, validate_conversation_record
from .preprocessors import AugmentationPreprocessor, SequentialPreprocessor
from .utils import extract_object_points, load_jsonl

# Exposed for debugging (e.g., OOM tracing)
LAST_SAMPLE_DEBUG: Dict[str, Any] = {}
logger = get_logger(__name__)


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
        user_prompt_summary: Optional[str] = None,
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
        self.user_prompt_summary = (
            user_prompt_summary if user_prompt_summary is not None else USER_PROMPT_SUMMARY
        )
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
        perm = list(range(base_len))
        if len(perm) > 1:
            self._rng.shuffle(perm)
        self._index_perm = perm

    def __len__(self) -> int:
        return len(self.base_records)

    def _create_builder(self, mode: Literal["dense", "summary"]) -> JSONLinesBuilder:
        user_prompt = (
            self.user_prompt_summary if mode == "summary" else self.user_prompt
        )
        return JSONLinesBuilder(
            user_prompt=user_prompt,
            emit_norm=self.emit_norm,
            mode=mode,
            json_format=self.json_format,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not self.base_records:
            raise IndexError("BaseCaptionDataset is empty")

        max_attempts = min(5, len(self.base_records))
        attempt = 0
        while True:
            base_idx = self._index_perm[(index + attempt) % len(self._index_perm)]
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
            if not record.get("messages"):
                if mode == "summary":
                    summary_val = record.get("summary")
                    if not isinstance(summary_val, str) or not summary_val.strip():
                        raise ValueError(
                            f"Dataset '{self.dataset_name}' is in summary mode but record is missing a non-empty 'summary' string."
                        )
                else:
                    objects = record.get("objects") or []
                    if not isinstance(objects, list) or not objects:
                        raise ValueError(
                            f"Dataset '{self.dataset_name}' is in dense mode but record has no objects."
                        )
                    for obj_idx, obj in enumerate(objects, start=1):
                        geom_type, points = extract_object_points(obj)
                        if not geom_type or not points:
                            raise ValueError(
                                f"Dataset '{self.dataset_name}' dense mode requires geometry; object_{obj_idx} is missing bbox_2d/poly/line points."
                            )

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
            except MaxLengthError as exc:
                # Drop over-length sample instead of truncating; try another record.
                logger.warning(
                    "Dropping over-length sample (dataset=%s, base_idx=%s, max_length=%s): %s",
                    self.dataset_name,
                    base_idx,
                    getattr(self.template, "max_length", None),
                    exc,
                )
                if system_prompt is not None and original_system is not None:
                    try:
                        self.template.system = original_system
                    except Exception:
                        pass
                attempt += 1
                if attempt >= max_attempts:
                    raise
                continue
            finally:
                if system_prompt is not None and original_system is not None:
                    try:
                        self.template.system = original_system
                    except Exception:
                        pass
            break

        self._validate_mm_length(encoded, record, base_idx)

        # Track last-sample debug info for OOM/root-cause tracing
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
            info["input_ids_len"] = len(input_ids)
            try:
                # Debug-only: emit per-sample token length to help sizing global_max_length
                logger.debug(
                    "Sample token length | dataset=%s base_idx=%s len=%d",
                    self.dataset_name,
                    base_idx,
                    len(input_ids),
                )
            except Exception:
                pass
        self.last_sample_debug = info
        LAST_SAMPLE_DEBUG.update(info)

        # Attach the original conversation so RLHF/GKD trainers can re-encode.
        encoded["messages"] = conversation_messages
        for key in ("assistant_payload", "objects", "metadata"):
            if key in merged:
                encoded[key] = copy.deepcopy(merged[key])

        return encoded

    def _validate_mm_length(
        self, encoded: Dict[str, Any], record: Dict[str, Any], base_idx: int
    ) -> None:
        input_ids = encoded.get("input_ids")
        image_grid_thw = encoded.get("image_grid_thw")
        if input_ids is None or image_grid_thw is None:
            return

        image_token_id = getattr(self.template, "image_token_id", None)
        if image_token_id is None:
            return

        processor = getattr(self.template, "processor", None)
        image_processor = getattr(processor, "image_processor", None)
        merge_size = getattr(image_processor, "merge_size", None)
        if merge_size is None:
            return

        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        image_token_count = sum(1 for t in input_ids if t == image_token_id)

        if hasattr(image_grid_thw, "tolist"):
            image_grid_thw = image_grid_thw.tolist()

        try:
            expected = 0
            for grid in image_grid_thw:
                if not isinstance(grid, (list, tuple)) or len(grid) < 3:
                    continue
                t, h, w = grid[:3]
                expected += int(t) * int(h) * int(w) // int(merge_size) ** 2
        except Exception:
            return

        if expected != image_token_count:
            meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            src = meta.get("_fusion_source") or meta.get("dataset") or self.dataset_name
            img0 = None
            images = record.get("images") or []
            if images:
                img0 = images[0]
            raise ValueError(
                "Qwen3-VL image token count mismatch: "
                f"dataset={src} base_idx={base_idx} "
                f"input_len={len(input_ids)} image_tokens={image_token_count} "
                f"expected={expected} merge_size={merge_size} "
                f"image_grid_thw={image_grid_thw} "
                f"image={img0} width={record.get('width')} height={record.get('height')}. "
                "This usually means truncation; increase global_max_length or reduce max_pixels/aug scale."
            )


# Backward compatibility alias
DenseCaptionDataset = BaseCaptionDataset
