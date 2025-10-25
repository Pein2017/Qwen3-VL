from typing import Any, Dict, Optional, Literal
import random

from .dynamic_pair import DynamicPairDataset, DynamicPairingConfig
from .utils import load_jsonl
from .builders import JSONLinesBuilder
from src.config.prompts import USER_PROMPT_SUMMARY


class DenseCaptionDataset(DynamicPairDataset):
    """Dense Caption dataset with dynamic mode selection per pairing group.
    
    This dataset wrapper enables per-group mode selection (dense or summary) by:
    1. Randomly selecting a mode for each pairing group based on summary_ratio
    2. Creating a builder with the selected mode
    3. Injecting the appropriate system prompt into the template
    """

    def __init__(
        self,
        base_records: Any,
        template: Any,
        user_prompt: str,
        emit_norm: Literal["none", "norm100", "norm1000"],
        config: Optional[DynamicPairingConfig] = None,
        augmenter: Optional[Any] = None,
        summary_ratio: Optional[float] = None,
        system_prompt_dense: Optional[str] = None,
        system_prompt_summary: Optional[str] = None,
    ):
        """Initialize dense caption dataset with dynamic mode selection.
        
        Args:
            base_records: List of records or records to use
            template: ms-swift template for encoding
            user_prompt: User prompt text
            emit_norm: Coordinate normalization for text output
            config: Dynamic pairing configuration
            augmenter: Optional augmentation config
            summary_ratio: Probability (0..1) of selecting summary mode per group. 
                          None or 0 = always dense; 1.0 = always summary
            system_prompt_dense: System prompt for dense mode (fallback to template if None)
            system_prompt_summary: System prompt for summary mode (required if summary_ratio > 0)
        """
        self.summary_ratio = summary_ratio if summary_ratio is not None else 0.0
        self.system_prompt_dense = system_prompt_dense
        self.system_prompt_summary = system_prompt_summary
        self.user_prompt = user_prompt
        self.emit_norm = emit_norm
        
        # Validate: if summary mode is possible, summary prompt must be provided
        if not (0.0 <= float(self.summary_ratio) <= 1.0):
            raise ValueError(
                f"summary_ratio must be within [0, 1], got {self.summary_ratio}. "
                f"Set 0 for dense-only or 1 for summary-only."
            )
        if self.summary_ratio > 0 and self.system_prompt_summary is None:
            raise ValueError(
                "system_prompt_summary is required when summary_ratio > 0. "
                "Please provide the summary system prompt."
            )
        
        # Create a placeholder builder; actual builder will be created per group
        self.placeholder_builder = JSONLinesBuilder(
            user_prompt=user_prompt,
            emit_norm=emit_norm,
            mode="dense",
        )

        super().__init__(
            base_records=base_records,
            template=template,
            pair_message_builder=self.placeholder_builder,
            config=config,
            augmenter=augmenter,
        )
        
        self._rng = random.Random(self.config.seed)

    def set_epoch(self, epoch: int) -> None:
        """Ensure per-epoch determinism for mode selection consistent with base class."""
        super().set_epoch(epoch)
        # Re-seed local RNG for mode selection using the same epoch seed policy
        try:
            seed = self._seed_for_epoch(epoch)  # provided by base class
        except Exception:
            seed = int(getattr(self.config, 'seed', 2025)) ^ int(epoch)
        self._rng = random.Random(seed)
    
    def _select_mode_for_group(self) -> Literal["dense", "summary"]:
        """Randomly select mode for current pairing group.
        
        Returns:
            "dense" or "summary" based on summary_ratio
        """
        if self.summary_ratio <= 0:
            return "dense"
        if self.summary_ratio >= 1.0:
            return "summary"
        # Probabilistic selection
        return "summary" if self._rng.random() < self.summary_ratio else "dense"
    
    def _create_builder_for_mode(self, mode: Literal["dense", "summary"]) -> JSONLinesBuilder:
        """Create a builder with appropriate system prompt for the selected mode.
        
        Args:
            mode: The mode to build for
            
        Returns:
            Configured JSONLinesBuilder
        """
        # Use summary-specific user prompt if in summary mode
        user_prompt = USER_PROMPT_SUMMARY if mode == "summary" else self.user_prompt
            
        builder = JSONLinesBuilder(
            user_prompt=user_prompt,
            emit_norm=self.emit_norm,
            mode=mode,
        )
        return builder
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item with dynamic mode selection at pairing group level.
        
        This overrides DynamicPairDataset.__getitem__ to:
        1. Select mode for this group
        2. Create builder with appropriate system prompt
        3. Inject system prompt into template
        4. Build and encode with template
        """
        # Use strong integer seed for worker-safe local RNG
        seed_local = self._rng.randrange(0, 2**32 - 1)
        rng_local = random.Random(seed_local)
        group_size = max(1, int(self.config.images_per_user_turn))

        # Gather records for this group
        if group_size <= 1:
            partner_index = self.pair_selector(index, len(self.base_records), rng_local)
            records = [
                copy.deepcopy(self.base_records[index]),
                copy.deepcopy(self.base_records[partner_index])
            ]
        else:
            start = index * group_size
            records = []
            for i in range(start, min(start + group_size, len(self.base_records))):
                records.append(copy.deepcopy(self.base_records[i]))
            if not records:
                records = [copy.deepcopy(self.base_records[-1])]

        # Apply preprocessing if available (e.g., augmentation)
        if self.preprocessor is not None:
            if hasattr(self.preprocessor, 'rng'):
                self.preprocessor.rng = rng_local
            records = [self.preprocessor(r) for r in records]

        # SELECT MODE FOR THIS GROUP
        mode = self._select_mode_for_group()
        
        # Create builder with the selected mode
        pair_builder = self._create_builder_for_mode(mode)
        
        # Build messages (this uses the builder's mode)
        if hasattr(pair_builder, 'build_many') and callable(getattr(pair_builder, 'build_many')):
            merged = pair_builder.build_many(records)
        else:
            # Fallback: only two records supported
            while len(records) < 2:
                records.append(records[-1])
            merged = pair_builder(records[0], records[1])

        # INJECT SYSTEM PROMPT based on mode
        if mode == "summary" and self.system_prompt_summary:
            # Override template system prompt for summary mode
            system_prompt = self.system_prompt_summary
        elif mode == "dense" and self.system_prompt_dense:
            # Override template system prompt for dense mode
            system_prompt = self.system_prompt_dense
        else:
            system_prompt = None
        
        # If we have an override, temporarily set it on the template
        original_system = None
        if system_prompt is not None:
            try:
                original_system = getattr(self.template, 'system', None)
                self.template.system = system_prompt
            except Exception:
                pass
        
        try:
            # Encode with appropriate system prompt
            encoded = self.template.encode(merged, return_length=True)
        finally:
            # Restore original system prompt
            if original_system is not None and system_prompt is not None:
                try:
                    self.template.system = original_system
                except Exception:
                    pass
        
        return encoded
    
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
        kwargs.pop("output_variant", None)  # Backward compat
        return DenseCaptionDataset(
            base_records=records,
            template=template,
            **kwargs,
        )


# Import copy for deepcopy in __getitem__
import copy


