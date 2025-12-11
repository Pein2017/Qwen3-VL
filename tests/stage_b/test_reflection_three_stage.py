from datetime import datetime, timezone

import pytest

from src.stage_b.config import ReflectionConfig
from src.stage_b.io.guidance import GuidanceRepository
from src.stage_b.reflection.engine import ReflectionEngine
from src.stage_b.types import ExperienceMetadata, MissionGuidance


class _DummyModel:
    device = "cpu"


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return ""

    def __call__(self, *args, **kwargs):
        return {"input_ids": None}


def _mk_engine(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("noop", encoding="utf-8")
    cfg = ReflectionConfig(
        prompt_path=prompt_file,
        batch_size=1,
        max_operations=None,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=64,
        max_reflection_length=512,
    )
    repo = GuidanceRepository(tmp_path / "guidance.json", retention=2)
    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    return ReflectionEngine(
        model=model, tokenizer=tokenizer, config=cfg, guidance_repo=repo
    )


def test_compact_guidance_dedup_and_reindex(tmp_path):
    engine = _mk_engine(tmp_path)
    repo = engine.guidance_repo

    mg = MissionGuidance(
        mission="m",
        experiences={"G0": "规则A", "G1": "规则B", "G2": "规则A"},
        step=1,
        updated_at=datetime.now(timezone.utc),
        metadata={
            "G0": ExperienceMetadata(
                updated_at=datetime.now(timezone.utc),
                reflection_id="r",
                sources=("s",),
                rationale=None,
            )
        },
    )
    repo._write({"m": mg})

    compacted = engine._compact_guidance("m")
    assert compacted is not None
    assert list(compacted.experiences.values()) == ["规则A", "规则B"]
    assert list(compacted.experiences.keys()) == ["G0", "G1"]


def test_compact_guidance_rejects_stage_a_like_text(tmp_path):
    engine = _mk_engine(tmp_path)
    repo = engine.guidance_repo
    mg = MissionGuidance(
        mission="m",
        experiences={"G0": "标签/测试×1"},
        step=1,
        updated_at=datetime.now(timezone.utc),
        metadata={},
    )
    repo._write({"m": mg})
    with pytest.raises(Exception):
        engine._compact_guidance("m")
