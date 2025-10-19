from typing import Any, Dict, Optional, Literal

from .dynamic_pair import DynamicPairDataset, DynamicPairingConfig
from .utils import load_jsonl
from .builders import JSONLinesBuilder


class DenseCaptionDataset(DynamicPairDataset):
    """Dense Caption dataset with dynamic pairing and JSON-lines support.
    
    This is a thin wrapper over DynamicPairDataset that provides convenient
    configuration for dense captioning tasks with different output formats.
    """

    def __init__(
        self,
        base_records: Any,
        template: Any,
        user_prompt: str,
        emit_norm: Literal["none", "norm100", "norm1000"],
        config: Optional[DynamicPairingConfig] = None,
        augmenter: Optional[Any] = None,
    ):
        """Initialize dense caption dataset.
        
        Args:
            base_records: List of records or records to use
            template: ms-swift template for encoding
            user_prompt: User prompt text
            emit_norm: Coordinate normalization for text output
            config: Dynamic pairing configuration
            augmenter: Optional augmentation config
        """
        message_builder = JSONLinesBuilder(
            user_prompt=user_prompt,
            emit_norm=emit_norm,
        )

        super().__init__(
            base_records=base_records,
            template=template,
            pair_message_builder=message_builder,
            config=config,
            augmenter=augmenter,
        )
    
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
            records = records[:int(sample_limit)]
        # Backward-compatibility: drop unused arg if present
        kwargs.pop("use_detailed_caption", None)
        return DenseCaptionDataset(
            base_records=records,
            template=template,
            **kwargs,
        )


