from __future__ import annotations

from dataclasses import dataclass

from src.datasets.dense_caption import DenseCaptionDataset


@dataclass
class _StubTemplate:
    system: str = "You are helpful."

    def __post_init__(self) -> None:
        self.mode = "train"
        self.encode_calls = 0

    def encode(self, merged, return_length=True):  # noqa: ANN001 - signature mirrors real template
        self.encode_calls += 1
        # Mimic transformers-style encoded payload
        return {
            "input_ids": [1, 2, 3],
            "pixel_values": [42],
            "image_grid_thw": [1, 1, 1],
        }

    def set_mode(self, mode: str) -> None:
        self.mode = mode


def _make_dataset():
    record = {
        "images": [],
        "objects": [
            {
                "desc": "obj_1",
                "bbox": [0.0, 0.0, 10.0, 10.0],
            }
        ],
        "width": 10,
        "height": 10,
    }
    template = _StubTemplate()
    dataset = DenseCaptionDataset(
        base_records=[record],
        template=template,
        user_prompt="Describe the scene.",
        emit_norm="none",
        json_format="type_b",
    )
    return dataset, template


def test_dense_caption_dataset_embeds_conversation_metadata():
    dataset, template = _make_dataset()

    sample = dataset[0]

    assert "messages" in sample
    assert sample["messages"][0]["role"] == "system"
    assert sample["messages"][1]["role"] == "user"
    assert "input_ids" in sample
    assert template.encode_calls == 1
