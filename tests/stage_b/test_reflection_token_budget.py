import re
from dataclasses import dataclass
from typing import Dict, Optional

from src.stage_b.reflection.engine import ReflectionEngine
from src.stage_b.config import ReflectionConfig
from src.stage_b.types import (
    GroupTicket,
    StageASummaries,
    ExperienceCandidate,
    ExperienceRecord,
    DeterministicSignals,
    ExperienceBundle,
)


class _FakeTokenizer:
    class _FakeIds:
        def __init__(self, n: int) -> None:
            self._n = n

        def size(self, dim: int) -> int:  # mimic torch.Tensor.size(1)
            return self._n

    def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = False, **_: Dict) -> Dict:
        # crude token estimate: 1 token per 6 chars, min 1
        n_tokens = max(1, len(text) // 6)
        return {"input_ids": self._FakeIds(n_tokens)}

    @property
    def pad_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 0

    def decode(self, *args, **kwargs):  # not used here
        return ""


class _FakeModel:
    def __init__(self):
        self.device = "cpu"


@dataclass
class _FakeGuidanceRepo:
    experiences: Optional[Dict[str, str]] = None

    def load(self) -> Dict:
        # No existing guidance by default
        return {}

    # apply_reflection not needed for these tests


def _make_record(group_id: str, mission: str, label: str, reason: str, *,
                 win_idx: int = 0, match_true: bool = False, match_false: bool = True) -> ExperienceRecord:
    # Two candidates per record; toggle label_match to craft priorities
    cand0 = ExperienceCandidate(
        candidate_index=0,
        verdict="pass",
        reason=f"{reason}-0",
        confidence=0.5,
        signals=DeterministicSignals(
            label_match=match_true,
            self_consistency=None,
            confidence=0.5,
        ),
        summary="摘要很长很长，包含大量细节，用于占用token。",
        critique="评述也很长，用于制造足够的文本长度。",
    )
    cand1 = ExperienceCandidate(
        candidate_index=1,
        verdict="fail",
        reason=f"{reason}-1",
        confidence=0.4,
        signals=DeterministicSignals(
            label_match=match_false,
            self_consistency=None,
            confidence=0.4,
        ),
        summary="另一份摘要，继续占用token。",
        critique="另一份评述，继续占用token。",
    )
    ticket = GroupTicket(
        group_id=group_id,
        mission=mission,
        label=label,  # type: ignore[arg-type]
        summaries=StageASummaries(per_image={"a.jpg": "清晰，无遮挡。", "b.jpg": "角度不足。"}),
    )
    return ExperienceRecord(
        ticket=ticket,
        candidates=(cand0, cand1),
        winning_candidate=win_idx,
        guidance_step=1,
    )


def _engine_with_budget(budget: int) -> ReflectionEngine:
    cfg = ReflectionConfig(
        prompt_path=__import__("pathlib").Path("configs/prompts/stage_b_reflection_prompt.txt"),
        batch_size=8,
        allow_uncertain=True,
        token_budget=budget,
        temperature=1.0,
        top_p=0.95,
        max_new_tokens=8,
        max_reflection_length=99999,
        eligibility_policy="selected_mismatch_or_all_wrong",
    )
    return ReflectionEngine(
        model=_FakeModel(),  # type: ignore[arg-type]
        tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        config=cfg,
        guidance_repo=_FakeGuidanceRepo(),  # type: ignore[arg-type]
    )


def test_token_budget_trims_records():
    engine = _engine_with_budget(budget=200)  # very small
    # Build 3 records with sizeable text
    recs = [
        _make_record("g1", "MISSION_X", "pass", "R1", win_idx=0, match_true=False, match_false=True),
        _make_record("g2", "MISSION_X", "fail", "R2", win_idx=0, match_true=False, match_false=True),
        _make_record("g3", "MISSION_X", "pass", "R3", win_idx=0, match_true=False, match_false=True),
    ]
    bundle = ExperienceBundle(mission="MISSION_X", records=tuple(recs), reflection_cycle=0, guidance_step=1)

    prompt = engine._build_reflection_prompt(bundle)
    m = re.search(r"批次: (\d+) 组", prompt)
    assert m, f"missing 批次 line in prompt: {prompt[:200]}"
    kept = int(m.group(1))
    assert kept < 3, "expected trimming under tight token budget"
    assert kept >= 1, "should keep at least one record"


def test_prioritization_keeps_contradictions_first():
    engine = _engine_with_budget(budget=200)
    # rec_a: no contradiction (both label_match False)
    rec_a = _make_record("ga", "MISSION_Y", "pass", "NO_CONTRA", win_idx=0, match_true=False, match_false=False)
    # rec_b: contradiction across candidates (True + False)
    rec_b = _make_record("gb", "MISSION_Y", "fail", "HAS_CONTRA", win_idx=0, match_true=True, match_false=False)
    # budget should allow only one block; we expect HAS_CONTRA present
    bundle = ExperienceBundle(mission="MISSION_Y", records=(rec_a, rec_b), reflection_cycle=0, guidance_step=1)

    prompt = engine._build_reflection_prompt(bundle)
    m = re.search(r"批次: (\d+) 组", prompt)
    kept = int(m.group(1)) if m else 0
    assert kept == 1, "expected only one record due to budget"
    assert "HAS_CONTRA" in prompt, "highest-priority contradictory record should be kept"
