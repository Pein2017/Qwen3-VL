import random
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast
from torch.utils.data import get_worker_info

from torch.utils.data import Dataset

from .utils import load_jsonl
from .contracts import ConversationRecord, validate_conversation_record


def random_pair_selector(index: int, total_size: int, rng: random.Random) -> int:
    if total_size <= 1:
        return index
    partner = index
    if total_size == 2:
        partner = 1 - index
    else:
        while partner == index:
            partner = rng.randrange(0, total_size)
    return partner


def default_pair_message_builder(
    record_a: ConversationRecord, record_b: ConversationRecord
) -> Dict[str, Any]:
    messages_a: Sequence[Dict[str, Any]] = record_a.get("messages") or []
    messages_b: Sequence[Dict[str, Any]] = record_b.get("messages") or []

    def extract_user_contents(
        messages: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        for turn in messages:
            if turn.get("role") == "user":
                contents = turn.get("content") or []
                text_chunks = [
                    c.get("text")
                    for c in contents
                    if c.get("type") == "text" and c.get("text")
                ]
                user_text = "\n".join(text_chunks) if text_chunks else None
                return contents, user_text
        return [], None

    contents_a, text_a = extract_user_contents(messages_a)
    contents_b, text_b = extract_user_contents(messages_b)

    images_a = [c for c in contents_a if c.get("type") == "image"]
    images_b = [c for c in contents_b if c.get("type") == "image"]

    merged_user_contents: List[Dict[str, Any]] = []
    merged_user_contents.extend(images_a)
    merged_user_contents.extend(images_b)

    merged_text = None
    if text_a and text_b:
        merged_text = f"Image A context:\n{text_a}\n\nImage B context:\n{text_b}"
    elif text_a or text_b:
        merged_text = text_a or text_b
    if merged_text:
        merged_user_contents.append({"type": "text", "text": merged_text})

    assistant_turns = [t for t in messages_a if t.get("role") == "assistant"]
    assistant_text = None
    if assistant_turns:
        assistant_contents = assistant_turns[0].get("content") or []
        for c in assistant_contents:
            if c.get("type") == "text" and c.get("text"):
                assistant_text = c.get("text")
                break

    merged_messages: List[Dict[str, Any]] = [
        {"role": "user", "content": merged_user_contents},
    ]
    if assistant_text:
        merged_messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        )

    return {"messages": merged_messages}


@dataclass
class DynamicPairingConfig:
    seed: int = 2025
    pre_tokenize: bool = False
    # New: how many images/records per user turn (grouping size)
    images_per_user_turn: int = 2


class DynamicPairDataset(Dataset):
    def __init__(
        self,
        base_records: Sequence[Mapping[str, Any]],
        template: Any,
        pair_selector: Callable[[int, int, random.Random], int] = random_pair_selector,
        pair_message_builder: Callable[
            [ConversationRecord, ConversationRecord], Dict[str, Any]
        ] = default_pair_message_builder,
        config: Optional[DynamicPairingConfig] = None,
        augmenter: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        bypass_prob: float = 0.0,
    ) -> None:
        validated_records: List[ConversationRecord] = []
        for idx, record in enumerate(base_records):
            try:
                validated = validate_conversation_record(record)
            except ValueError as exc:
                raise ValueError(f"Base record {idx} is invalid: {exc}") from exc
            validated_records.append(cast(ConversationRecord, copy.deepcopy(validated)))

        self.base_records: List[ConversationRecord] = validated_records
        self.template = template
        self.pair_selector = pair_selector
        self.pair_message_builder = pair_message_builder
        self.config = config or DynamicPairingConfig()
        self.augmenter = augmenter  # Kept for backward compatibility
        self.preprocessor = preprocessor
        self.bypass_prob = float(bypass_prob)
        self._epoch: int = 0
        self._rng = random.Random(self._seed_for_epoch(self._epoch))
        # Build an index permutation for this epoch to enable per-epoch shuffling
        self._index_perm: List[int] = list(range(len(self.base_records)))
        self._rebuild_perm_for_epoch()

        # If augmenter provided but no preprocessor, create augmentation preprocessor
        if self.augmenter is not None and self.preprocessor is None:
            from .preprocessors import AugmentationPreprocessor

            self.preprocessor = AugmentationPreprocessor(
                augmenter=self.augmenter, bypass_prob=self.bypass_prob
            )

    @staticmethod
    def from_jsonl(jsonl_path: str, template: Any, **kwargs) -> "DynamicPairDataset":
        records = load_jsonl(jsonl_path)
        return DynamicPairDataset(records, template, **kwargs)

    def _seed_for_epoch(self, epoch: int) -> int:
        """Derive a 32-bit seed from base seed and epoch (rank-agnostic).

        Keep the same across ranks so that external samplers can shard consistently.
        """
        base = int(getattr(self.config, "seed", 2025)) & 0xFFFFFFFF
        # Mix with an odd constant (golden ratio) for decorrelation across epochs
        mixed = (base ^ ((epoch + 1) * 0x9E3779B1)) & 0xFFFFFFFF
        return mixed

    def _rebuild_perm_for_epoch(self) -> None:
        """Shuffle index permutation deterministically for the current epoch."""
        # Start from identity then shuffle with epoch-seeded RNG
        self._index_perm = list(range(len(self.base_records)))
        if len(self._index_perm) > 1:
            self._rng.shuffle(self._index_perm)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._rng = random.Random(self._seed_for_epoch(self._epoch))
        self._rebuild_perm_for_epoch()

    def __len__(self) -> int:
        # Number of grouped samples when grouping >1
        g = max(1, int(self.config.images_per_user_turn))
        if g <= 1:
            return len(self.base_records)
        return (len(self.base_records) + g - 1) // g

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Build a stable, per-sample RNG that depends on epoch/index and worker id
        seed_local = self._rng.randrange(0, 2**32 - 1)
        seed_local ^= (int(index) * 0x85EBCA6B) & 0xFFFFFFFF
        wi = get_worker_info()
        if wi is not None:
            seed_local ^= ((wi.id + 1) * 0xC2B2AE35) & 0xFFFFFFFF
        rng_local = random.Random(seed_local & 0xFFFFFFFF)
        group_size = max(1, int(self.config.images_per_user_turn))

        if group_size <= 1:
            # Single-record turn: pick current record from permuted order
            base_idx = self._index_perm[index % len(self._index_perm)]
            # Pair with a random partner in base index space
            partner_base_idx = self.pair_selector(
                base_idx, len(self.base_records), rng_local
            )
            records = [
                copy.deepcopy(self.base_records[base_idx]),
                copy.deepcopy(self.base_records[partner_base_idx]),
            ]
        else:
            # Sequential grouping into fixed-size chunks: [0..g-1], [g..2g-1], ...
            start = index * group_size
            records = []
            end = min(start + group_size, len(self.base_records))
            for i in range(start, end):
                perm_i = self._index_perm[i]
                records.append(copy.deepcopy(self.base_records[perm_i]))
            if not records:
                # Fallback to last record if index is out of range due to race
                records = [copy.deepcopy(self.base_records[-1])]

        # Apply preprocessing if available (e.g., augmentation)
        if self.preprocessor is not None:
            if hasattr(self.preprocessor, "rng"):
                self.preprocessor.rng = rng_local
            processed: List[ConversationRecord] = []
            for record in records:
                result = self.preprocessor(record)
                if result is None:
                    continue
                processed.append(result)
            if not processed:
                raise ValueError(
                    "Preprocessor removed all records from the pairing group"
                )
            records = processed

        # Adapt to builder API: support list of records
        pair_builder = self.pair_message_builder
        if hasattr(pair_builder, "build_many") and callable(
            getattr(pair_builder, "build_many")
        ):
            merged = pair_builder.build_many(records)
        else:
            # Fallback: only two records supported
            while len(records) < 2:
                records.append(records[-1])
            merged = pair_builder(records[0], records[1])

        encoded = self.template.encode(merged, return_length=True)
        return encoded


__all__ = [
    "DynamicPairDataset",
    "DynamicPairingConfig",
    "random_pair_selector",
    "default_pair_message_builder",
]
