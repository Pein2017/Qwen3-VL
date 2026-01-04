from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.stage_b.config import ReflectionConfig
from src.stage_b.reflection import ReflectionEngine
from src.stage_b.types import (
    DeterministicSignals,
    ExperienceBundle,
    ExperienceCandidate,
    ExperienceRecord,
    GroupTicket,
    MissionGuidance,
    StageASummaries,
)


class _FakeModel:
    device = "cpu"


class _FakeTokenizer:
    class _Ids:
        def __init__(self, n: int) -> None:
            self._n = n

        def size(self, dim: int) -> int:
            return self._n

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text: str, **kwargs):
        n_tokens = max(1, len(text) // 6)
        return {"input_ids": self._Ids(n_tokens)}


class _FakeGuidanceRepo:
    def __init__(self) -> None:
        self._guidance = MissionGuidance(
            mission="m",
            experiences={
                "G0": "初始检查清单",
                "S1": "若关键要点无法确认，则判不通过。",
            },
            step=1,
            updated_at=datetime.now(timezone.utc),
        )

    def load(self):
        return {"m": self._guidance}


def _mk_bundle(*, mission: str, ticket_key: str) -> ExperienceBundle:
    group_id, label = ticket_key.split("::", 1)
    ticket = GroupTicket(
        group_id=group_id,
        mission=mission,
        label=label,  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"image_1": "摘要"}),
    )
    cand = ExperienceCandidate(
        candidate_index=0,
        verdict=label,  # type: ignore[arg-type]
        reason="ok",
        signals=DeterministicSignals(label_match=True, self_consistency=None),
    )
    record = ExperienceRecord(
        ticket=ticket,
        candidates=(cand,),
        winning_candidate=0,
        guidance_step=1,
    )
    return ExperienceBundle(mission=mission, records=(record,), reflection_cycle=0, guidance_step=1)


def _engine(tmp_path: Path) -> ReflectionEngine:
    decision = tmp_path / "decision.txt"
    ops = tmp_path / "ops.txt"
    decision.write_text("noop", encoding="utf-8")
    ops.write_text("noop", encoding="utf-8")
    cfg = ReflectionConfig(
        decision_prompt_path=decision,
        ops_prompt_path=ops,
        batch_size=1,
        max_operations=5,
        token_budget=4096,
        max_reflection_length=4096,
        max_new_tokens=16,
    )
    return ReflectionEngine(
        model=_FakeModel(),  # type: ignore[arg-type]
        tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        config=cfg,
        guidance_repo=_FakeGuidanceRepo(),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "payload, error_match",
    [
        (
            {"hypotheses": [{"text": "若无法确认则通过。", "falsifier": "x", "evidence": ["g1::pass"]}]},
            "conflicts with scaffold",
        ),
        (
            {"hypotheses": [{"text": "出现第三态时通过。", "falsifier": "x", "evidence": ["g1::pass"]}]},
            "forbidden",
        ),
        (
            {
                "hypotheses": [
                    {"text": "出现第三态时通过。", "falsifier": "x", "evidence": ["g1::pass"]}
                ]
            },
            "forbidden",
        ),
        (
            {
                "hypotheses": [
                    {
                        "text": "若能确认则通过，否则不通过。",
                        "falsifier": "若关键证据充分，则不应判定为不通过。",
                        "evidence": ["g1::pass"],
                    }
                ]
            },
            "affirmative",
        ),
        (
            {"hypotheses": [{"text": "若能确认则通过。", "falsifier": "x", "evidence": []}]},
            "evidence",
        ),
        (
            {"hypotheses": [{"text": "若能确认则通过。", "falsifier": "x", "evidence": ["UNKNOWN"]}]},
            "learnable",
        ),
        (
            {
                "hypotheses": [
                    {
                        "text": "若能确认则通过。",
                        "falsifier": "x",
                        "evidence": ["g1::pass"],
                        "dimension": "brand",
                    }
                ]
            },
            "brand",
        ),
        (
            {
                "hypotheses": [
                    {
                        "text": "image_1/标签/缺失",
                        "falsifier": "x",
                        "evidence": ["g1::pass"],
                    }
                ]
            },
            "summary",
        ),
    ],
)
def test_hypotheses_validator_rejects_invalid(tmp_path: Path, payload, error_match):
    engine = _engine(tmp_path)
    bundle = _mk_bundle(mission="m", ticket_key="g1::pass")
    engine._generate_json_payload = lambda **kwargs: payload  # type: ignore[assignment]

    with pytest.raises(ValueError, match=error_match):
        engine.run_ops_pass(bundle)
