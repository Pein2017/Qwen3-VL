from pathlib import Path

import pytest
from transformers import AutoTokenizer

from src.data_collators.token_types import compute_token_types, TokenType


def _get_tokenizer():
    model_dir = Path("model_cache/models/Qwen/Qwen3-VL-8B-Instruct")
    if not model_dir.exists():
        pytest.skip(f"Tokenizer model dir not found: {model_dir}")
    return AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True
    )


def _build_labels(tokenizer, text):
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"][0]
    labels = ids.clone()
    return labels


def test_token_types_alignment_bbox_poly_line():
    tokenizer = _get_tokenizer()
    payload = {
        "object_1": {"bbox_2d": [10, 20, 30, 40], "desc": "obj_a"},
        "object_2": {"poly": [0, 0, 10, 0, 10, 10], "desc": "obj_b"},
        "object_3": {
            "line_points": 4,
            "line": [1, 2, 3, 4, 5, 6, 7, 8],
            "desc": "obj_c",
        },
    }
    # Use the same json style as the collator for perfect alignment
    assistant_text = (
        '{"object_1": {"bbox_2d": [10, 20, 30, 40], "desc": "obj_a"}, '
        '"object_2": {"poly": [0, 0, 10, 0, 10, 10], "desc": "obj_b"}, '
        '"object_3": {"line_points": 4, "line": [1, 2, 3, 4, 5, 6, 7, 8], "desc": "obj_c"}}'
        "<|im_end|>\n"
    )
    labels = _build_labels(tokenizer, assistant_text)

    token_types = compute_token_types(
        tokenizer=tokenizer,
        payload=payload,
        labels=labels,
        attention_mask=None,
        suffix_tokens=["<|im_end|>\n"],
    )
    assert token_types is not None
    assert token_types.shape == labels.shape
    supervised = labels != -100
    assert supervised.sum() == (token_types != TokenType.IGNORE).sum()
    assert (token_types == TokenType.DESC).any()
    assert (token_types == TokenType.COORD).any()
    assert (token_types == TokenType.FORMAT).any()


def test_token_types_length_mismatch_aligns():
    tokenizer = _get_tokenizer()
    payload = {"object_1": {"bbox_2d": [1, 2, 3, 4], "desc": "obj"}}
    labels = _build_labels(
        tokenizer, '{"object_1": {"bbox_2d": [1, 2, 3, 4], "desc": "obj"}}'
    )
    truncated = labels[:-1]  # force mismatch

    token_types = compute_token_types(
        tokenizer=tokenizer,
        payload=payload,
        labels=truncated,
        attention_mask=None,
        suffix_tokens=None,
    )
    assert token_types is not None
    assert token_types.shape == truncated.shape
